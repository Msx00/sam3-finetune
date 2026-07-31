"""Epoch logging and slice/patient segmentation metrics for MoE-SAM3."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import torch
from torch.nn import functional as F

from .moe_lora import AREA_CLASSES, BOUNDARY_CLASSES, MODALITIES


LOSS_NAMES = (
    "total_loss", "sam3_loss", "aux_loss", "modality_loss", "area_loss",
    "area_reg_loss", "boundary_router_loss", "boundary_seg_loss",
    "load_balance_loss", "refine_loss",
)


def _binary_scores(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    prediction = logits.sigmoid() >= 0.5
    target = targets >= 0.5
    intersection = float((prediction & target).sum().item())
    pred_sum = float(prediction.sum().item())
    target_sum = float(target.sum().item())
    union = pred_sum + target_sum - intersection
    dice = 1.0 if pred_sum + target_sum == 0 else 2.0 * intersection / (pred_sum + target_sum)
    iou = 1.0 if union == 0 else intersection / union
    return dice, iou


class EpochStatistics:
    """Accumulate loss, router, expert, SvANet and segmentation statistics."""

    def __init__(self) -> None:
        self.loss_sums = defaultdict(float)
        self.loss_batches = 0
        self.loss_weight = 0
        self.router_correct = defaultdict(int)
        self.router_total = defaultdict(int)
        self.entropy_sum = defaultdict(float)
        self.expert_counts = {
            f"{modality}_area_{label}": 0
            for modality in MODALITIES for label in AREA_CLASSES
        }
        self.expert_counts.update({
            f"{modality}_boundary_{label}": 0
            for modality in MODALITIES for label in BOUNDARY_CLASSES
        })
        self.teacher_used = 0
        self.teacher_total = 0
        self.svanet = defaultdict(float)
        self.svanet_batches = 0
        self.slice_records = []

    def state_dict(self) -> Dict[str, Any]:
        """Return a CPU-only, pickle-safe state for mid-epoch checkpoints."""
        return {
            "loss_sums": dict(self.loss_sums),
            "loss_batches": int(self.loss_batches),
            "loss_weight": int(self.loss_weight),
            "router_correct": dict(self.router_correct),
            "router_total": dict(self.router_total),
            "entropy_sum": dict(self.entropy_sum),
            "expert_counts": dict(self.expert_counts),
            "teacher_used": int(self.teacher_used),
            "teacher_total": int(self.teacher_total),
            "svanet": dict(self.svanet),
            "svanet_batches": int(self.svanet_batches),
            "slice_records": list(self.slice_records),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore statistics accumulated before a mid-epoch interruption."""
        self.loss_sums = defaultdict(float, state.get("loss_sums", {}))
        self.loss_batches = int(state.get("loss_batches", 0))
        self.loss_weight = int(state.get("loss_weight", 0))
        self.router_correct = defaultdict(int, state.get("router_correct", {}))
        self.router_total = defaultdict(int, state.get("router_total", {}))
        self.entropy_sum = defaultdict(float, state.get("entropy_sum", {}))
        restored_experts = dict(state.get("expert_counts", {}))
        for name in self.expert_counts:
            self.expert_counts[name] = int(restored_experts.get(name, 0))
        self.teacher_used = int(state.get("teacher_used", 0))
        self.teacher_total = int(state.get("teacher_total", 0))
        self.svanet = defaultdict(float, state.get("svanet", {}))
        self.svanet_batches = int(state.get("svanet_batches", 0))
        self.slice_records = list(state.get("slice_records", []))

    def update_losses(self, components: Mapping[str, Any], weight: int = 1) -> None:
        """Accumulate batch-mean losses weighted by the effective batch size."""
        weight = max(int(weight), 1)
        for name in LOSS_NAMES:
            value = components.get(name)
            if value is not None:
                scalar = float(
                    value.detach().float().item()
                    if torch.is_tensor(value) else value
                )
                self.loss_sums[name] += scalar * weight
        self.loss_batches += 1
        self.loss_weight += weight

    def update_router(
        self, routes: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor]
    ) -> None:
        for family in ("modality", "area", "boundary"):
            logits = routes[f"{family}_logits"].detach()
            target = targets[family].to(logits.device).long()
            prediction = logits.argmax(dim=-1)
            self.router_correct[family] += int((prediction == target).sum().item())
            self.router_total[family] += int(target.numel())
            probabilities = logits.softmax(dim=-1).clamp_min(1e-8)
            entropy = -(probabilities * probabilities.log()).sum(dim=-1)
            self.entropy_sum[family] += float(entropy.sum().item())

        modality = routes["modality"].detach().argmax(dim=-1)
        area = routes["area"].detach().argmax(dim=-1)
        boundary = routes["boundary"].detach().argmax(dim=-1)
        for modality_index, area_index, boundary_index in zip(
            modality.tolist(), area.tolist(), boundary.tolist()
        ):
            prefix = MODALITIES[modality_index]
            self.expert_counts[f"{prefix}_area_{AREA_CLASSES[area_index]}"] += 1
            self.expert_counts[
                f"{prefix}_boundary_{BOUNDARY_CLASSES[boundary_index]}"
            ] += 1
        for family in ("modality", "area", "boundary"):
            mask = routes.get(f"teacher_{family}_mask")
            if mask is not None:
                self.teacher_used += int(mask.sum().item())
                self.teacher_total += int(mask.numel())

    def update_svanet(self, output: Optional[Mapping[str, Any]]) -> None:
        if not output:
            return
        stats = output.get("batch_stats", {})
        for name in (
            "small_count", "trigger_count", "empty_mask_count",
            "box_fallback_count", "full_image_fallback_count",
        ):
            self.svanet[name] += float(stats.get(name, 0))
        trigger_count = float(stats.get("trigger_count", 0))
        self.svanet["roi_width_weighted"] += float(stats.get("mean_roi_width", 0)) * trigger_count
        self.svanet["roi_height_weighted"] += float(stats.get("mean_roi_height", 0)) * trigger_count
        refine_loss = output.get("refine_loss")
        if refine_loss is not None:
            self.svanet["refine_loss"] += float(
                refine_loss.detach().item() if torch.is_tensor(refine_loss) else refine_loss
            )
        self.svanet_batches += 1

    def update_segmentation(
        self,
        base_logits: torch.Tensor,
        final_logits: torch.Tensor,
        gt_masks: torch.Tensor,
        metadata: Sequence[Mapping[str, Any]],
    ) -> None:
        if final_logits.shape[-2:] != gt_masks.shape[-2:]:
            final_logits = F.interpolate(
                final_logits[:, None].float(), gt_masks.shape[-2:], mode="bilinear",
                align_corners=False,
            )[:, 0]
        if base_logits.shape[-2:] != gt_masks.shape[-2:]:
            base_logits = F.interpolate(
                base_logits[:, None].float(), gt_masks.shape[-2:], mode="bilinear",
                align_corners=False,
            )[:, 0]
        for index, item in enumerate(metadata):
            base_dice, base_iou = _binary_scores(base_logits[index], gt_masks[index])
            final_dice, final_iou = _binary_scores(final_logits[index], gt_masks[index])
            self.slice_records.append({
                "patient_id": int(item["patient_id"]),
                "modality": str(item["modality"]).upper(),
                "area_label": int(item["area_label"]),
                "boundary_label": int(item["boundary_label"]),
                "base_dice": base_dice,
                "base_iou": base_iou,
                "final_dice": final_dice,
                "final_iou": final_iou,
            })

    def merge(self, other: "EpochStatistics") -> None:
        for name, value in other.loss_sums.items(): self.loss_sums[name] += value
        self.loss_batches += other.loss_batches
        self.loss_weight += other.loss_weight
        for name, value in other.router_correct.items(): self.router_correct[name] += value
        for name, value in other.router_total.items(): self.router_total[name] += value
        for name, value in other.entropy_sum.items(): self.entropy_sum[name] += value
        for name, value in other.expert_counts.items(): self.expert_counts[name] += value
        self.teacher_used += other.teacher_used
        self.teacher_total += other.teacher_total
        for name, value in other.svanet.items(): self.svanet[name] += value
        self.svanet_batches += other.svanet_batches
        self.slice_records.extend(other.slice_records)

    @staticmethod
    def _mean(records: Iterable[Mapping[str, float]], key: str) -> float:
        values = [float(record[key]) for record in records]
        return sum(values) / len(values) if values else 0.0

    def segmentation_report(self) -> Dict[str, Any]:
        records = self.slice_records
        report: Dict[str, Any] = {
            "num_slices": len(records),
            "slice_dice": self._mean(records, "final_dice"),
            "slice_iou": self._mean(records, "final_iou"),
            "sam3_base_dice": self._mean(records, "base_dice"),
            "sam3_base_iou": self._mean(records, "base_iou"),
            "final_mask_dice": self._mean(records, "final_dice"),
            "final_mask_iou": self._mean(records, "final_iou"),
        }
        patients: Dict[tuple[str, int], list] = defaultdict(list)
        for record in records:
            patients[(record["modality"], record["patient_id"])].append(record)
        patient_dice = [self._mean(items, "final_dice") for items in patients.values()]
        patient_iou = [self._mean(items, "final_iou") for items in patients.values()]
        report["num_patients"] = len(patients)
        report["patient_macro_dice"] = sum(patient_dice) / len(patient_dice) if patient_dice else 0.0
        report["patient_macro_iou"] = sum(patient_iou) / len(patient_iou) if patient_iou else 0.0
        for modality in MODALITIES:
            selected = [item for item in records if item["modality"] == modality]
            report[f"{modality}_dice"] = self._mean(selected, "final_dice")
        for index, label in enumerate(AREA_CLASSES):
            selected = [item for item in records if item["area_label"] == index]
            report[f"{label}_dice"] = self._mean(selected, "final_dice")
        for index, label in enumerate(BOUNDARY_CLASSES):
            selected = [item for item in records if item["boundary_label"] == index]
            report[f"{label}_dice"] = self._mean(selected, "final_dice")
        return report

    def report(self, configured_teacher_ratio: float = 0.0) -> Dict[str, Any]:
        router = {}
        for family in ("modality", "area", "boundary"):
            total = self.router_total[family]
            router[f"{family}_accuracy"] = self.router_correct[family] / total if total else 0.0
            router[f"{family}_entropy"] = self.entropy_sum[family] / total if total else 0.0
        router["configured_teacher_forcing_ratio"] = float(configured_teacher_ratio)
        router["actual_teacher_forcing_ratio"] = (
            self.teacher_used / self.teacher_total if self.teacher_total else 0.0
        )
        triggers = self.svanet["trigger_count"]
        svanet = {
            "small_count": int(self.svanet["small_count"]),
            "trigger_count": int(triggers),
            "empty_mask_count": int(self.svanet["empty_mask_count"]),
            "box_fallback_count": int(self.svanet["box_fallback_count"]),
            "full_image_fallback_count": int(self.svanet["full_image_fallback_count"]),
            "mean_roi_width": self.svanet["roi_width_weighted"] / triggers if triggers else 0.0,
            "mean_roi_height": self.svanet["roi_height_weighted"] / triggers if triggers else 0.0,
            "refine_loss": self.svanet["refine_loss"] / self.svanet_batches if self.svanet_batches else 0.0,
        }
        return {
            "loss": {
                name: self.loss_sums[name] / self.loss_weight if self.loss_weight else 0.0
                for name in LOSS_NAMES
            },
            "router": router,
            "experts": dict(self.expert_counts),
            "svanet": svanet,
            "segmentation": self.segmentation_report(),
        }
