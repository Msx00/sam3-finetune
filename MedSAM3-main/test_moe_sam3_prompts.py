#!/usr/bin/env python3
"""Evaluate HMOE-SAM3/SvANet checkpoints under multiple prompt modes.

The script is intentionally separate from training.  It loads a staged HMOE-SAM3
checkpoint, evaluates MR, US, or mixed MR+US patient datasets, and reports Dice
and HD95 for:

* all evaluated slices;
* small-target slices, defined by area_label == 0 when area thresholds are
  available, otherwise by ``area_ratio <= --small-area-threshold``.

SAM3 image grounding requires a text query internally. Therefore prompt mode
``none`` accepts only the image from the caller and inserts a fixed task-level
token (default: ``prostate``), with no sample-specific text or box prompt.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

try:
    from scipy import ndimage as ndi
except ImportError:  # HD95 remains present but becomes NaN for non-empty masks.
    ndi = None

from data.patient_dataset import PatientDataset
from models.inference_utils import (
    move_to_device,
    normalized_xyxy_prompts,
    route_predictions,
    select_best_mask_logits,
)
from models.runtime_config import resolve_moe_runtime_config
from models.moe_lora import AREA_CLASSES, BOUNDARY_CLASSES


PROMPT_MODES = ("none", "text", "coarse_box", "box", "text_box")
DATA_MODES = ("mr", "us", "mixed")


def load_evaluation_config(config_path: Path) -> Tuple[Dict[str, Any], Path]:
    """Load a full config or a dedicated evaluation config with base_config."""
    document = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(document, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    base_value = document.get("base_config")
    if not base_value:
        return document, config_path

    unsupported = sorted(set(document) - {"base_config", "evaluation"})
    if unsupported:
        raise ValueError(
            "A dedicated evaluation config may only contain base_config and "
            f"evaluation; unsupported keys: {unsupported}"
        )
    base_path = Path(str(base_value)).expanduser()
    if not base_path.is_absolute():
        base_path = config_path.parent / base_path
    base_path = base_path.resolve()
    if not base_path.is_file():
        raise FileNotFoundError(f"base_config does not exist: {base_path}")
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    if not isinstance(base_config, dict):
        raise ValueError(f"Base configuration must be a mapping: {base_path}")
    effective = dict(base_config)
    effective["evaluation"] = dict(document.get("evaluation") or {})
    return effective, base_path


def _config_get(config: Mapping[str, Any], name: str, default: Any = None) -> Any:
    return (config.get("evaluation") or {}).get(name, default)


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _as_prompt_modes(value: Any) -> List[str]:
    if value is None:
        return list(PROMPT_MODES)
    if isinstance(value, str):
        modes = [item.strip() for item in value.split(",") if item.strip()]
    else:
        modes = [str(item).strip() for item in value if str(item).strip()]
    invalid = [mode for mode in modes if mode not in PROMPT_MODES]
    if invalid:
        raise ValueError(f"Unsupported prompt mode(s): {invalid}; choose from {PROMPT_MODES}")
    return modes


def _split_path(value: Any, split: str) -> Any:
    return value.format(split=split) if isinstance(value, str) else value


def build_runtime(
    config_path: Path,
    config: Mapping[str, Any],
    checkpoint: Path,
    stage: Optional[int],
    device_index: int,
) -> Tuple[SAM3TrainerNative, Dict[str, Any], int, Dict[str, Any]]:
    # Keep heavyweight SAM3/model-building dependencies out of module import so
    # metric/config helpers remain CPU-testable in lightweight environments.
    from train_sam3_lora_native import SAM3TrainerNative

    selected_stage = int(stage or config.get("training", {}).get("stage", 5))
    if torch.cuda.is_available():
        torch.cuda.set_device(device_index)
    moe_config = resolve_moe_runtime_config(config)
    router_config = config.get("router") or {}
    trainer = SAM3TrainerNative(
        str(config_path),
        moe_config=moe_config,
        patient_dataset_config=config.get("dataset"),
        router_config=router_config,
        training_stage=selected_stage,
        svanet_config=config.get("svanet"),
        load_stage_dependencies=False,
        # Evaluation loads exactly --checkpoint below. A training.resume entry
        # in the shared YAML must not restore another checkpoint first.
        load_training_resume=False,
    )
    checkpoint_payload = trainer.stage_manager.load_checkpoint(
        str(checkpoint), allowed_stages={selected_stage}
    )
    trainer.stage_manager.set_module_modes(training=False)
    trainer.model.eval()
    if trainer.svanet_adapter is not None:
        trainer.svanet_adapter.eval()
    if int(getattr(trainer._unwrapped_model, "num_interactive_steps_val", 0)) != 0:
        raise RuntimeError(
            "Deployment evaluation requires num_interactive_steps_val=0 so "
            "SAM3 cannot sample prompts from ground truth"
        )
    return trainer, config, selected_stage, checkpoint_payload


def build_patient_dataset(
    config: Mapping[str, Any],
    split: str,
    data_mode: str,
    max_patients: Optional[int],
    mr_max_patients: Optional[int],
    us_max_patients: Optional[int],
    max_slices_per_patient: Optional[int],
    sampling_mode: str,
    seed: int,
    area_thresholds: Optional[Mapping[str, Any]] = None,
    boundary_thresholds: Optional[Mapping[str, Any]] = None,
) -> Dataset:
    from train_sam3_lora_native import COCOSegmentDataset

    dataset_cfg = config.get("dataset") or {}
    modalities = ("mr", "us") if data_mode == "mixed" else (data_mode,)
    datasets: List[Dataset] = []
    for modality in modalities:
        root = dataset_cfg.get(f"{modality}_root") or dataset_cfg.get(f"{modality}_image_root")
        if not root:
            raise ValueError(f"dataset.{modality}_root is required for data_mode={data_mode!r}")
        coco_json = dataset_cfg.get(f"{modality}_{split}_coco_json") or _split_path(
            dataset_cfg.get(f"{modality}_coco_json"), split
        )
        boxes_json = dataset_cfg.get(f"{modality}_{split}_boxes_json") or _split_path(
            dataset_cfg.get(f"{modality}_boxes_json"), split
        )
        source_max = _coalesce(
            mr_max_patients if modality == "mr" else us_max_patients,
            max_patients,
        )
        base_dataset = COCOSegmentDataset(
            data_dir=root,
            split=split,
            annotation_file=coco_json,
        )
        datasets.append(
            PatientDataset(
                base_dataset=base_dataset,
                modality_root=root,
                split=split,
                modality=modality.upper(),
                num_patients=source_max,
                sampling_mode=sampling_mode,
                seed=seed,
                mask_root=dataset_cfg.get(f"{modality}_mask_root"),
                boxes_json=boxes_json,
                strict_dataset_check=dataset_cfg.get("strict_dataset_check", True),
                area_thresholds=(
                    dict(area_thresholds)
                    if area_thresholds
                    else dataset_cfg.get("area_threshold_file")
                ),
                boundary_thresholds=(
                    dict(boundary_thresholds)
                    if boundary_thresholds
                    else dataset_cfg.get("boundary_threshold_file")
                ),
                return_format="sam3",
                max_slices_per_patient=max_slices_per_patient,
            )
        )
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def iter_patient_datasets(dataset: Dataset) -> Iterator[PatientDataset]:
    if isinstance(dataset, PatientDataset):
        yield dataset
    elif isinstance(dataset, ConcatDataset):
        for child in dataset.datasets:
            yield from iter_patient_datasets(child)


def selected_patient_state(dataset: Dataset) -> Dict[str, Any]:
    selected: Dict[str, List[int]] = {"mr": [], "us": []}
    for patient_dataset in iter_patient_datasets(dataset):
        selected[patient_dataset.modality.lower()] = list(patient_dataset.patient_ids)
    return {
        "mr_patient_ids": selected["mr"],
        "us_patient_ids": selected["us"],
        "num_mr_patients": len(selected["mr"]),
        "num_us_patients": len(selected["us"]),
        "num_patients_total": len(selected["mr"]) + len(selected["us"]),
    }


def preflight_dataset(
    dataset: Dataset,
    prompt_modes: Sequence[str],
    require_ground_truth: bool,
) -> Dict[str, int]:
    num_records = 0
    missing_gt: List[str] = []
    missing_boxes: List[str] = []
    needs_boxes = any(
        mode in {"coarse_box", "box", "text_box"} for mode in prompt_modes
    )
    for patient_dataset in iter_patient_datasets(dataset):
        for record in patient_dataset.records:
            num_records += 1
            sample_name = (
                f"{record.modality}/{record.patient_id}/{record.slice_id}"
            )
            if not record.mask_path or not Path(record.mask_path).is_file():
                missing_gt.append(sample_name)
            if needs_boxes and not record.box_prompt:
                missing_boxes.append(sample_name)
    if num_records != len(dataset):
        raise RuntimeError(
            f"Dataset preflight counted {num_records} records but len(dataset)={len(dataset)}"
        )
    if require_ground_truth and missing_gt:
        preview = ", ".join(missing_gt[:5])
        raise RuntimeError(
            f"Ground-truth masks are required, but {len(missing_gt)} samples have no "
            f"readable mask (first: {preview})"
        )
    if missing_boxes:
        preview = ", ".join(missing_boxes[:5])
        raise RuntimeError(
            f"box/text_box evaluation requires boxes for every sample, but "
            f"{len(missing_boxes)} are missing (first: {preview})"
        )
    return {
        "num_samples": num_records,
        "num_missing_gt": len(missing_gt),
        "num_missing_boxes": len(missing_boxes),
    }


def _normalized_cxcywh_boxes(
    boxes_xyxy: Sequence[Sequence[float]], width: int, height: int
) -> torch.Tensor:
    boxes = torch.tensor(boxes_xyxy, dtype=torch.float32).reshape(-1, 4)
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    return torch.stack(
        (
            (x1 + x2) / (2.0 * width),
            (y1 + y2) / (2.0 * height),
            (x2 - x1) / float(width),
            (y2 - y1) / float(height),
        ),
        dim=-1,
    ).clamp(0.0, 1.0)


def _noise_xyxy_boxes(
    boxes_xyxy: Sequence[Sequence[float]],
    width: int,
    height: int,
    box_noise_std: float,
    box_noise_max: Optional[float],
    generator: torch.Generator,
) -> torch.Tensor:
    """Apply SAM3-style Gaussian noise to absolute XYXY prompt boxes."""
    boxes = torch.tensor(boxes_xyxy, dtype=torch.float32).reshape(-1, 4)
    if box_noise_std <= 0.0 or boxes.numel() == 0:
        return boxes

    output = boxes.clone()
    image_bounds = torch.tensor(
        [float(width), float(height), float(width), float(height)],
        dtype=torch.float32,
    )
    for box_index, box in enumerate(boxes):
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        scale = torch.stack((box_width, box_height, box_width, box_height))
        noise = (
            torch.randn(4, generator=generator, dtype=torch.float32)
            * float(box_noise_std)
            * scale
        )
        if box_noise_max is not None:
            noise = noise.clamp(-float(box_noise_max), float(box_noise_max))
        candidate = (box + noise).clamp_min(0.0)
        candidate = torch.minimum(candidate, image_bounds)
        # Match RandomizeInputBbox's default minimum_box_area=0 behavior:
        # retain the original box if noise collapses or reverses either side.
        if candidate[2] > candidate[0] and candidate[3] > candidate[1]:
            output[box_index] = candidate
    return output


def _coarse_xyxy_boxes(
    boxes_xyxy: Sequence[Sequence[float]],
    width: int,
    height: int,
    expand_min: float,
    expand_max: float,
    jitter_std: float,
    jitter_max: Optional[float],
    generator: torch.Generator,
) -> torch.Tensor:
    """Match the expanded/jittered coarse-box curriculum used in training."""
    boxes = torch.tensor(boxes_xyxy, dtype=torch.float32).reshape(-1, 4)
    if boxes.numel() == 0:
        return boxes
    box_width = (boxes[:, 2] - boxes[:, 0]).clamp_min(1.0)
    box_height = (boxes[:, 3] - boxes[:, 1]).clamp_min(1.0)
    expansion = torch.empty(len(boxes)).uniform_(
        float(expand_min), float(expand_max), generator=generator
    )
    shift_x = torch.randn(len(boxes), generator=generator) * (
        float(jitter_std) * box_width
    )
    shift_y = torch.randn(len(boxes), generator=generator) * (
        float(jitter_std) * box_height
    )
    if jitter_max is not None:
        shift_x.clamp_(-float(jitter_max), float(jitter_max))
        shift_y.clamp_(-float(jitter_max), float(jitter_max))
    output = boxes.clone()
    output[:, 0] -= expansion * box_width
    output[:, 2] += expansion * box_width
    output[:, 1] -= expansion * box_height
    output[:, 3] += expansion * box_height
    output[:, (0, 2)] += shift_x[:, None]
    output[:, (1, 3)] += shift_y[:, None]
    output[:, 0].clamp_(0.0, max(float(width) - 1.0, 0.0))
    output[:, 1].clamp_(0.0, max(float(height) - 1.0, 0.0))
    output[:, 2].clamp_(1.0, float(width))
    output[:, 3].clamp_(1.0, float(height))
    output[:, 2] = torch.maximum(output[:, 2], output[:, 0] + 1.0).clamp_max(width)
    output[:, 3] = torch.maximum(output[:, 3], output[:, 1] + 1.0).clamp_max(height)
    return output


class PromptModeDataset(Dataset):
    """Wrap PatientDataset and force one requested deployment prompt mode."""

    def __init__(
        self,
        dataset: Dataset,
        prompt_mode: str,
        none_text: str,
        box_noise_std: float = 0.0,
        box_noise_max: Optional[float] = None,
        box_noise_seed: int = 42,
        coarse_box_expand_min: float = 0.15,
        coarse_box_expand_max: float = 0.45,
        coarse_box_jitter_std: float = 0.10,
        coarse_box_jitter_max: Optional[float] = 20.0,
        max_samples: Optional[int] = None,
    ) -> None:
        if prompt_mode not in PROMPT_MODES:
            raise ValueError(f"Unsupported prompt mode: {prompt_mode!r}")
        self.dataset = dataset
        self.prompt_mode = prompt_mode
        self.none_text = none_text
        self.box_noise_std = float(box_noise_std)
        self.box_noise_max = (
            None if box_noise_max is None else float(box_noise_max)
        )
        self.box_noise_seed = int(box_noise_seed)
        self.coarse_box_expand_min = float(coarse_box_expand_min)
        self.coarse_box_expand_max = float(coarse_box_expand_max)
        self.coarse_box_jitter_std = float(coarse_box_jitter_std)
        self.coarse_box_jitter_max = (
            None
            if coarse_box_jitter_max is None
            else float(coarse_box_jitter_max)
        )
        self.indices = list(range(len(dataset)))
        if max_samples is not None:
            self.indices = self.indices[: max(0, int(max_samples))]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        source_index = self.indices[index]
        datapoint = self.dataset[source_index]
        metadata = dict(getattr(datapoint, "patient_metadata", None) or {})
        setattr(datapoint, "patient_metadata", metadata)
        text = str(metadata.get("text_prompt") or "object")
        if self.prompt_mode == "none":
            text = self.none_text
        elif self.prompt_mode == "coarse_box":
            text = self.none_text
        elif self.prompt_mode == "box":
            text = "visual"

        box_prompt = metadata.get("box_prompt") or []
        if self.prompt_mode in {"coarse_box", "box", "text_box"} and not box_prompt:
            raise RuntimeError(
                f"{self.prompt_mode} requires box_prompt but sample has none: "
                f"{metadata.get('modality')} {metadata.get('patient_id')} {metadata.get('slice_id')}"
            )

        width = height = None
        if box_prompt:
            with Image.open(metadata["image_path"]) as handle:
                width, height = handle.size
            original_boxes = torch.tensor(
                box_prompt, dtype=torch.float32
            ).reshape(-1, 4)
            noisy_boxes = original_boxes
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.box_noise_seed + source_index)
            if self.prompt_mode == "coarse_box":
                noisy_boxes = _coarse_xyxy_boxes(
                    original_boxes.tolist(),
                    width,
                    height,
                    self.coarse_box_expand_min,
                    self.coarse_box_expand_max,
                    self.coarse_box_jitter_std,
                    self.coarse_box_jitter_max,
                    generator,
                )
            elif (
                self.prompt_mode in {"box", "text_box"}
                and self.box_noise_std > 0.0
            ):
                noisy_boxes = _noise_xyxy_boxes(
                    original_boxes.tolist(),
                    width,
                    height,
                    self.box_noise_std,
                    self.box_noise_max,
                    generator,
                )
            prompt_boxes = _normalized_cxcywh_boxes(
                noisy_boxes.tolist(), width, height
            )
            prompt_labels = torch.ones(len(prompt_boxes), dtype=torch.long)
            metadata["box_prompt_original"] = original_boxes.tolist()
            metadata["box_prompt"] = noisy_boxes.tolist()
            metadata["box_noise_applied"] = bool(
                self.prompt_mode == "coarse_box"
                or (
                    self.prompt_mode in {"box", "text_box"}
                    and self.box_noise_std > 0.0
                )
            )
            metadata["box_transform"] = (
                "expanded_coarse" if self.prompt_mode == "coarse_box" else "gaussian"
            )
        else:
            prompt_boxes = None
            prompt_labels = None

        for query in datapoint.find_queries:
            query.query_text = text
            if self.prompt_mode in {"coarse_box", "box", "text_box"}:
                query.input_bbox = prompt_boxes
                query.input_bbox_label = prompt_labels
            else:
                query.input_bbox = None
                query.input_bbox_label = None
            # Ground truth is evaluated separately from ``mask_path`` after
            # forward. Keep it out of the model's BatchedDatapoint entirely.
            query.object_ids_output = []
            query.semantic_target = None
        metadata["evaluation_prompt_mode"] = self.prompt_mode
        metadata["applied_prompt_text"] = text
        metadata["prompt_text_equals_none"] = text == self.none_text
        for image in datapoint.images:
            image.objects = []
        return datapoint


def make_collate_fn() -> Any:
    from sam3.train.data.collator import collate_fn_api

    def collate(batch: Sequence[Any]) -> Dict[str, Any]:
        metadata = [getattr(item, "patient_metadata", None) for item in batch]
        output = collate_fn_api(list(batch), dict_key="input", with_seg_masks=False)
        output["_patient_metadata"] = metadata
        return output

    return collate


def _has_valid_boxes(find_input: Any) -> bool:
    boxes = getattr(find_input, "input_boxes", None)
    mask = getattr(find_input, "input_boxes_mask", None)
    if boxes is None or mask is None or boxes.numel() == 0:
        return False
    return bool((~mask.bool()).any().item())


def validate_prompt_batch(batch: Dict[str, Any], prompt_mode: str, none_text: str) -> None:
    input_batch = batch["input"]
    if not input_batch.find_inputs:
        raise RuntimeError("Collated batch contains no find_inputs")
    find_input = input_batch.find_inputs[-1]
    has_boxes = _has_valid_boxes(find_input)
    texts = [str(value) for value in input_batch.find_text_batch]
    if any(int(target.num_boxes.sum().item()) != 0 for target in input_batch.find_targets):
        raise RuntimeError("Deployment evaluation BatchedDatapoint contains GT targets")
    if prompt_mode == "none":
        if has_boxes or any(value != none_text for value in texts):
            raise RuntimeError("none prompt mode retained a non-fixed text or box prompt")
    elif prompt_mode == "text":
        if has_boxes or not texts or any(not value.strip() or value == "visual" for value in texts):
            raise RuntimeError("text prompt mode lost text or retained box prompts")
    elif prompt_mode == "coarse_box":
        if not has_boxes or any(value != none_text for value in texts):
            raise RuntimeError("coarse_box mode must use fixed task text and a box")
    elif prompt_mode == "box":
        if not has_boxes or any(value != "visual" for value in texts):
            raise RuntimeError("box prompt mode did not retain box-only prompts")
    elif prompt_mode == "text_box":
        if not has_boxes or not texts or any(not value.strip() or value == "visual" for value in texts):
            raise RuntimeError("text_box prompt mode did not retain both prompts")
    else:
        raise RuntimeError(f"Unsupported prompt mode reached forward: {prompt_mode}")


def load_gt_mask(mask_path: Optional[str], size: Tuple[int, int]) -> Optional[np.ndarray]:
    if not mask_path:
        return None
    path = Path(mask_path)
    if not path.is_file():
        return None
    with Image.open(path) as handle:
        mask = np.asarray(handle.convert("L")) > 0
    height, width = size
    if mask.shape != (height, width):
        resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (width, height), resample=Image.Resampling.NEAREST
        )
        mask = np.asarray(resized) > 0
    return mask


def hd95(pred: np.ndarray, gt: np.ndarray, spacing_yx: Optional[Sequence[float]]) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return math.nan
    if ndi is None:
        raise RuntimeError(
            "SciPy is required to calculate HD95; install scipy in the test environment"
        )
    sampling = tuple(float(v) for v in spacing_yx) if spacing_yx is not None else None
    pred_surface = pred ^ ndi.binary_erosion(pred)
    gt_surface = gt ^ ndi.binary_erosion(gt)
    distance_to_gt = ndi.distance_transform_edt(~gt_surface, sampling=sampling)
    distance_to_pred = ndi.distance_transform_edt(~pred_surface, sampling=sampling)
    distances = np.concatenate(
        (distance_to_gt[pred_surface], distance_to_pred[gt_surface])
    )
    return float(np.percentile(distances, 95))


def mask_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing_yx: Optional[Sequence[float]],
) -> Dict[str, Any]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    pred_n = tp + fp
    gt_n = tp + fn
    denom = 2 * tp + fp + fn
    union = tp + fp + fn
    if pred_n == 0 and gt_n == 0:
        dice = 1.0
        iou = 1.0
        precision = 1.0
        recall = 1.0
        status = "both_empty"
    else:
        dice = 2.0 * tp / denom if denom else 1.0
        iou = tp / union if union else 1.0
        precision = tp / pred_n if pred_n else math.nan
        recall = tp / gt_n if gt_n else math.nan
        status = "gt_only" if pred_n == 0 else "pred_only" if gt_n == 0 else "both_foreground"
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "hd95": hd95(pred, gt, spacing_yx),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pred_foreground": bool(pred_n),
        "gt_foreground": bool(gt_n),
        "status": status,
    }


def _nanmean(values: Iterable[Any]) -> float:
    clean: List[float] = []
    for value in values:
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            clean.append(numeric)
    return float(np.mean(clean)) if clean else math.nan


def _metric_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    metric_names = ("dice", "hd95", "iou", "precision", "recall")
    evaluated = [row for row in rows if row.get("gt_available")]

    def finite_count(name: str) -> int:
        count = 0
        for row in evaluated:
            try:
                if math.isfinite(float(row.get(name))):
                    count += 1
            except (TypeError, ValueError):
                pass
        return count

    summary = {
        "num_samples": len(rows),
        "num_evaluated": len(evaluated),
        "num_patients": len(
            {
                (str(row.get("modality")), str(row.get("patient_id")))
                for row in rows
            }
        ),
        "num_gt_foreground": sum(bool(row.get("gt_foreground")) for row in evaluated),
        "num_gt_empty": sum(not bool(row.get("gt_foreground")) for row in evaluated),
        "num_missing_prediction": sum(bool(row.get("missing_prediction")) for row in rows),
        **{f"num_valid_{name}": finite_count(name) for name in metric_names},
        **{
            name: _nanmean(row.get(name) for row in evaluated)
            for name in metric_names
        },
    }
    for family in ("modality", "area", "boundary", "joint"):
        summary[f"route_{family}_accuracy"] = _nanmean(
            float(bool(row[f"route_{family}_correct"]))
            for row in rows
            if f"route_{family}_correct" in row
        )
    for family in ("modality", "area", "boundary"):
        summary[f"route_{family}_mean_confidence"] = _nanmean(
            row.get(f"route_{family}_confidence") for row in rows
        )
    for family in ("area", "boundary"):
        policies = [
            str(row.get(f"route_{family}_policy"))
            for row in rows
            if row.get(f"route_{family}_policy") is not None
        ]
        for policy in ("shared", "topk", "top1"):
            summary[f"route_{family}_{policy}_ratio"] = (
                sum(value == policy for value in policies) / len(policies)
                if policies else math.nan
            )
    return summary


def _patient_metric_summary(
    rows: Sequence[Mapping[str, Any]],
    prefix: str = "all_",
) -> Dict[str, Any]:
    """Macro-average slice metrics after giving every patient equal weight.

    ``summarize_patients`` first averages the 2D slice metrics within each
    patient. This function then averages those patient-level values, so a
    patient with many slices does not contribute more weight than a patient
    with few slices. This is deliberately not a reconstructed 3D metric.
    """
    evaluated = [
        row for row in rows if int(row.get(f"{prefix}num_evaluated") or 0) > 0
    ]
    metric_names = ("dice", "hd95", "iou", "precision", "recall")

    def values(name: str) -> List[float]:
        clean: List[float] = []
        for row in evaluated:
            value = row.get(f"{prefix}{name}")
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                clean.append(numeric)
        return clean

    metric_values = {name: values(name) for name in metric_names}
    return {
        "num_patients_total": len(rows),
        "num_patients_evaluated": len(evaluated),
        **{
            f"num_valid_{name}": len(metric_values[name])
            for name in metric_names
        },
        **{
            name: (
                float(np.mean(metric_values[name]))
                if metric_values[name]
                else math.nan
            )
            for name in metric_names
        },
    }


def summarize(
    rows: List[Dict[str, Any]],
    patient_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    if patient_rows is None:
        patient_rows = summarize_patients(rows)
    summary: Dict[str, Any] = {"overall": _metric_summary(rows), "by_prompt_mode": {}}
    for prompt_mode in sorted({str(row["prompt_mode"]) for row in rows}):
        mode_rows = [row for row in rows if row["prompt_mode"] == prompt_mode]
        mode_patient_rows = [
            row for row in patient_rows if row["prompt_mode"] == prompt_mode
        ]
        small_rows = [row for row in mode_rows if row.get("is_small_target")]
        summary["by_prompt_mode"][prompt_mode] = {
            "all_samples": _metric_summary(mode_rows),
            "small_samples": _metric_summary(small_rows),
            "patient_macro": _patient_metric_summary(mode_patient_rows, "all_"),
            "small_patient_macro": _patient_metric_summary(
                mode_patient_rows, "small_"
            ),
            "by_modality": {},
        }
        for modality in sorted({str(row.get("modality")) for row in mode_rows}):
            modality_rows = [row for row in mode_rows if str(row.get("modality")) == modality]
            modality_small = [row for row in modality_rows if row.get("is_small_target")]
            modality_patient_rows = [
                row
                for row in mode_patient_rows
                if str(row.get("modality")) == modality
            ]
            summary["by_prompt_mode"][prompt_mode]["by_modality"][modality] = {
                "all_samples": _metric_summary(modality_rows),
                "small_samples": _metric_summary(modality_small),
                "patient_macro": _patient_metric_summary(
                    modality_patient_rows, "all_"
                ),
                "small_patient_macro": _patient_metric_summary(
                    modality_patient_rows, "small_"
                ),
            }
    return summary


def _row_key(row: Mapping[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(row.get("prompt_mode")),
        str(row.get("modality")),
        str(row.get("patient_id")),
        str(row.get("slice_id")),
    )


def _sample_key(row: Mapping[str, Any]) -> Tuple[str, str, str]:
    return (
        str(row.get("modality")),
        str(row.get("patient_id")),
        str(row.get("slice_id")),
    )


def summarize_patients(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_patient: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_patient[
            (str(row.get("prompt_mode")), str(row.get("modality")), str(row.get("patient_id")))
        ].append(row)
    output: List[Dict[str, Any]] = []
    for (prompt_mode, modality, patient_id), items in sorted(by_patient.items()):
        small_items = [row for row in items if row.get("is_small_target")]
        output.append(
            {
                "prompt_mode": prompt_mode,
                "modality": modality,
                "patient_id": patient_id,
                **{f"all_{k}": v for k, v in _metric_summary(items).items()},
                **{f"small_{k}": v for k, v in _metric_summary(small_items).items()},
            }
        )
    return output


def patient_macro_summary_rows(summary: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Flatten overall and per-modality patient-macro summaries for CSV."""
    output: List[Dict[str, Any]] = []
    for prompt_mode, block in summary.get("by_prompt_mode", {}).items():
        scopes = [("overall", block)]
        scopes.extend(
            (str(modality), modality_block)
            for modality, modality_block in block.get("by_modality", {}).items()
        )
        for modality, scope in scopes:
            for target_scope, key in (
                ("all", "patient_macro"),
                ("small", "small_patient_macro"),
            ):
                output.append(
                    {
                        "prompt_mode": prompt_mode,
                        "modality": modality,
                        "target_scope": target_scope,
                        **dict(scope[key]),
                    }
                )
    return output


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def predict_batch(trainer: SAM3TrainerNative, batch: Dict[str, Any]) -> Dict[str, Any]:
    input_batch = move_to_device(batch["input"], trainer.device)
    metadata = batch.get("_patient_metadata") or []
    # Deployment evaluation must not expose modality/area/boundary GT labels to
    # the controller, even though teacher forcing is disabled in eval mode.
    trainer.moe_controller.set_routing_targets(None)
    with torch.inference_mode():
        outputs = trainer.model(input_batch)
        final_output = outputs.output[-1][-1]
        base_logits, query_indices = select_best_mask_logits(final_output)
        routes = trainer.moe_controller.current_routes
        if routes is None:
            raise RuntimeError("MoE router did not produce routes during inference")
        find_input = input_batch.find_inputs[-1]
        image_indices = find_input.img_ids.long()
        images = input_batch.img_batch[image_indices]
        adapter_output = None
        final_logits = base_logits
        if trainer.svanet_adapter is not None:
            adapter_output = trainer.svanet_adapter(
                images=images,
                sam3_logits=base_logits,
                area_logits=routes.get(
                    "area_selected_logits", routes["area_logits"]
                ),
                box_prompts=normalized_xyxy_prompts(find_input),
                locator_logits=routes.get("image_locator_logits"),
            )
            final_logits = adapter_output["final_logits"]
    expected = len(metadata)
    if base_logits.shape[0] != expected or final_logits.shape[0] != expected:
        raise RuntimeError(
            "Prediction/metadata batch mismatch: "
            f"metadata={expected}, base_logits={base_logits.shape[0]}, "
            f"final_logits={final_logits.shape[0]}. The test dataset must expose "
            "one medical category query per image."
        )
    for key in ("modality_logits", "area_logits", "boundary_logits"):
        if key not in routes or routes[key].shape[0] != expected:
            raise RuntimeError(
                f"Router output {key!r} does not align with metadata batch {expected}"
            )
    return {
        "base_logits": base_logits,
        "final_logits": final_logits,
        "query_indices": query_indices,
        "routes": routes,
        "adapter_output": adapter_output,
        "metadata": metadata,
    }


def logits_to_mask(logits: torch.Tensor, height: int, width: int, threshold: float) -> np.ndarray:
    resized = F.interpolate(
        logits[None, None].float(),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return (resized.sigmoid().detach().cpu().numpy() >= threshold)


def is_small_target(metadata: Mapping[str, Any], small_area_threshold: float) -> bool:
    area_ratio = float(metadata.get("area_ratio") or 0.0)
    # An empty mask has no target and must not be reported as a small target.
    if area_ratio <= 0.0:
        return False
    area_label = metadata.get("area_label")
    if area_label is not None and int(area_label) >= 0:
        return int(area_label) == 0
    return area_ratio <= float(small_area_threshold)


def evaluate_prompt_mode(
    trainer: SAM3TrainerNative,
    base_dataset: Dataset,
    prompt_mode: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    dataset = PromptModeDataset(
        base_dataset,
        prompt_mode=prompt_mode,
        none_text=args.none_text,
        box_noise_std=args.box_noise_std,
        box_noise_max=args.box_noise_max,
        box_noise_seed=args.box_noise_seed,
        coarse_box_expand_min=args.coarse_box_expand_min,
        coarse_box_expand_max=args.coarse_box_expand_max,
        coarse_box_jitter_std=args.coarse_box_jitter_std,
        coarse_box_jitter_max=args.coarse_box_jitter_max,
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=trainer.device.type == "cuda",
        drop_last=False,
        collate_fn=make_collate_fn(),
    )
    rows: List[Dict[str, Any]] = []
    mode_mask_dir = output_dir / "masks" / prompt_mode
    if args.save_masks:
        mode_mask_dir.mkdir(parents=True, exist_ok=True)
    for batch_index, batch in enumerate(loader):
        validate_prompt_batch(batch, prompt_mode, args.none_text)
        result = predict_batch(trainer, batch)
        for index, metadata in enumerate(result["metadata"]):
            with Image.open(metadata["image_path"]) as handle:
                width, height = handle.size
            binary = logits_to_mask(result["final_logits"][index], height, width, args.threshold)
            gt = load_gt_mask(metadata.get("mask_path"), (height, width))
            route = {
                f"route_{key}": value
                for key, value in route_predictions(result["routes"], index).items()
            }
            modality_target = str(metadata.get("modality", "")).upper()
            area_target = int(metadata.get("area_label", -1))
            boundary_target = int(metadata.get("boundary_label", -1))
            modality_correct = route.get("route_modality") == modality_target
            area_correct = (
                0 <= area_target < len(AREA_CLASSES)
                and route.get("route_area") == AREA_CLASSES[area_target]
            )
            boundary_correct = (
                0 <= boundary_target < len(BOUNDARY_CLASSES)
                and route.get("route_boundary") == BOUNDARY_CLASSES[boundary_target]
            )
            mask_path = None
            if args.save_masks:
                rel = f"{metadata['modality']}_{metadata['patient_id']}_{metadata['slice_id']}.png"
                mask_path = mode_mask_dir / rel
                Image.fromarray(binary.astype(np.uint8) * 255).save(mask_path)
            row: Dict[str, Any] = {
                "prompt_mode": prompt_mode,
                "modality": metadata.get("modality"),
                "patient_id": metadata.get("patient_id"),
                "slice_id": metadata.get("slice_id"),
                "image_path": metadata.get("image_path"),
                "mask_gt_path": metadata.get("mask_path"),
                "pred_mask_path": str(mask_path) if mask_path else None,
                "text_prompt": metadata.get("text_prompt") if prompt_mode in {"text", "text_box"} else None,
                "applied_prompt_text": metadata.get("applied_prompt_text"),
                "prompt_text_equals_none": metadata.get("prompt_text_equals_none"),
                "box_prompt": json.dumps(metadata.get("box_prompt")) if prompt_mode in {"coarse_box", "box", "text_box"} else None,
                "box_prompt_original": json.dumps(metadata.get("box_prompt_original")) if prompt_mode in {"coarse_box", "box", "text_box"} else None,
                "box_transform": metadata.get("box_transform"),
                "box_noise_applied": bool(metadata.get("box_noise_applied", False)),
                "box_noise_std": args.box_noise_std,
                "box_noise_max": args.box_noise_max,
                "box_noise_seed": args.box_noise_seed,
                "selected_query": int(result["query_indices"][index].item()),
                "is_small_target": is_small_target(metadata, args.small_area_threshold),
                "small_target_rule": (
                    "area_label==0_and_nonempty"
                    if metadata.get("area_label") is not None
                    and int(metadata.get("area_label")) >= 0
                    else f"0<area_ratio<={args.small_area_threshold}"
                ),
                "area_ratio": metadata.get("area_ratio"),
                "area_label": metadata.get("area_label"),
                "boundary_label": metadata.get("boundary_label"),
                "route_modality_correct": modality_correct,
                "route_area_correct": area_correct,
                "route_boundary_correct": boundary_correct,
                "route_joint_correct": (
                    modality_correct and area_correct and boundary_correct
                ),
                "routing_gt_exposed_to_model": False,
                "gt_available": gt is not None,
                "missing_prediction": not bool(binary.any()),
                **route,
            }
            if gt is not None:
                row.update(mask_metrics(binary, gt, args.spacing_yx))
                row["hd95_unit"] = "mm" if args.spacing_yx is not None else "pixel"
            rows.append(row)
        if batch_index == 0 or (batch_index + 1) % args.log_interval == 0:
            print(
                f"[{prompt_mode}] batch {batch_index + 1}/{len(loader)}, "
                f"rows={len(rows)}"
            )
    return sorted(rows, key=_row_key)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate HMOE-SAM3 with none/text/coarse_box/box/text_box prompts."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage", type=int, choices=range(1, 6), default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--split", default=None)
    parser.add_argument("--data-mode", choices=DATA_MODES, default=None)
    parser.add_argument("--prompt-modes", nargs="+", default=None)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--mr-max-patients", type=int, default=None)
    parser.add_argument("--us-max-patients", type=int, default=None)
    parser.add_argument("--max-slices-per-patient", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--patient-sampling-mode", choices=("sequential", "random"), default=None)
    parser.add_argument("--patient-seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--small-area-threshold", type=float, default=None)
    parser.add_argument("--none-text", default=None)
    parser.add_argument("--box-noise-std", type=float, default=None)
    parser.add_argument("--box-noise-max", type=float, default=None)
    parser.add_argument("--box-noise-seed", type=int, default=None)
    parser.add_argument("--coarse-box-expand-min", type=float, default=None)
    parser.add_argument("--coarse-box-expand-max", type=float, default=None)
    parser.add_argument("--coarse-box-jitter-std", type=float, default=None)
    parser.add_argument("--coarse-box-jitter-max", type=float, default=None)
    parser.add_argument("--spacing-yx", type=float, nargs=2, default=None)
    parser.add_argument("--save-masks", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--require-ground-truth",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fail before inference if any selected sample lacks a GT mask (default: true)",
    )
    parser.add_argument("--log-interval", type=int, default=None)
    return parser


def apply_config_defaults(args: argparse.Namespace, config: Mapping[str, Any]) -> argparse.Namespace:
    output_root = Path((config.get("output") or {}).get("output_dir", "outputs"))
    args.output_dir = str(_coalesce(args.output_dir, _config_get(config, "output_dir"), output_root / "prompt_eval"))
    args.split = str(_coalesce(args.split, _config_get(config, "split"), (config.get("dataset") or {}).get("test_split"), "test"))
    args.data_mode = str(_coalesce(args.data_mode, _config_get(config, "data_mode"), "mixed"))
    args.prompt_modes = _as_prompt_modes(_coalesce(args.prompt_modes, _config_get(config, "prompt_modes"), PROMPT_MODES))
    args.max_patients = _coalesce(args.max_patients, _config_get(config, "max_patients"))
    args.mr_max_patients = _coalesce(args.mr_max_patients, _config_get(config, "mr_max_patients"))
    args.us_max_patients = _coalesce(args.us_max_patients, _config_get(config, "us_max_patients"))
    args.max_slices_per_patient = _coalesce(args.max_slices_per_patient, _config_get(config, "max_slices_per_patient"))
    args.max_samples = _coalesce(args.max_samples, _config_get(config, "max_samples"))
    args.patient_sampling_mode = str(_coalesce(args.patient_sampling_mode, _config_get(config, "patient_sampling_mode"), "sequential"))
    args.patient_seed = int(_coalesce(args.patient_seed, _config_get(config, "patient_seed"), 42))
    args.batch_size = int(_coalesce(args.batch_size, _config_get(config, "batch_size"), 1))
    args.num_workers = int(_coalesce(args.num_workers, _config_get(config, "num_workers"), 0))
    args.threshold = float(_coalesce(args.threshold, _config_get(config, "threshold"), 0.5))
    args.small_area_threshold = float(_coalesce(args.small_area_threshold, _config_get(config, "small_area_threshold"), 0.01))
    args.none_text = str(_coalesce(args.none_text, _config_get(config, "none_text"), "prostate"))
    args.box_noise_std = float(
        _coalesce(args.box_noise_std, _config_get(config, "box_noise_std"), 0.0)
    )
    args.box_noise_max = _coalesce(
        args.box_noise_max, _config_get(config, "box_noise_max")
    )
    if args.box_noise_max is not None:
        args.box_noise_max = float(args.box_noise_max)
    args.box_noise_seed = int(
        _coalesce(
            args.box_noise_seed,
            _config_get(config, "box_noise_seed"),
            args.patient_seed,
        )
    )
    curriculum = (config.get("dataset") or {}).get("prompt_curriculum") or {}
    args.coarse_box_expand_min = float(_coalesce(
        args.coarse_box_expand_min,
        _config_get(config, "coarse_box_expand_min"),
        curriculum.get("coarse_box_expand_min"),
        0.15,
    ))
    args.coarse_box_expand_max = float(_coalesce(
        args.coarse_box_expand_max,
        _config_get(config, "coarse_box_expand_max"),
        curriculum.get("coarse_box_expand_max"),
        0.45,
    ))
    args.coarse_box_jitter_std = float(_coalesce(
        args.coarse_box_jitter_std,
        _config_get(config, "coarse_box_jitter_std"),
        curriculum.get("coarse_box_jitter_std"),
        0.10,
    ))
    args.coarse_box_jitter_max = _coalesce(
        args.coarse_box_jitter_max,
        _config_get(config, "coarse_box_jitter_max"),
        curriculum.get("coarse_box_jitter_max"),
        20.0,
    )
    if args.coarse_box_jitter_max is not None:
        args.coarse_box_jitter_max = float(args.coarse_box_jitter_max)
    args.spacing_yx = _coalesce(args.spacing_yx, _config_get(config, "spacing_yx"))
    args.save_masks = bool(_coalesce(args.save_masks, _config_get(config, "save_masks"), True))
    args.require_ground_truth = bool(
        _coalesce(
            args.require_ground_truth,
            _config_get(config, "require_ground_truth"),
            True,
        )
    )
    args.log_interval = int(_coalesce(args.log_interval, _config_get(config, "log_interval"), 10))
    if not args.prompt_modes:
        raise ValueError("At least one prompt mode must be configured")
    for value, name in (
        (args.max_patients, "max_patients"),
        (args.mr_max_patients, "mr_max_patients"),
        (args.us_max_patients, "us_max_patients"),
        (args.max_slices_per_patient, "max_slices_per_patient"),
        (args.max_samples, "max_samples"),
    ):
        if value is not None and int(value) <= 0:
            raise ValueError(f"evaluation.{name} must be positive or null")
    if args.batch_size <= 0 or args.num_workers < 0 or args.log_interval <= 0:
        raise ValueError(
            "batch_size/log_interval must be > 0 and num_workers must be >= 0"
        )
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("evaluation.threshold must be within [0, 1]")
    if not 0.0 < args.small_area_threshold <= 1.0:
        raise ValueError("evaluation.small_area_threshold must be within (0, 1]")
    if args.box_noise_std < 0.0:
        raise ValueError("evaluation.box_noise_std must be >= 0")
    if args.box_noise_max is not None and args.box_noise_max <= 0.0:
        raise ValueError("evaluation.box_noise_max must be positive or null")
    if not 0.0 <= args.coarse_box_expand_min <= args.coarse_box_expand_max:
        raise ValueError("coarse box expansion must satisfy 0 <= min <= max")
    if args.coarse_box_jitter_std < 0.0:
        raise ValueError("coarse_box_jitter_std must be non-negative")
    if args.coarse_box_jitter_max is not None and args.coarse_box_jitter_max <= 0.0:
        raise ValueError("coarse_box_jitter_max must be positive or null")
    if args.spacing_yx is not None:
        args.spacing_yx = [float(value) for value in args.spacing_yx]
        if len(args.spacing_yx) != 2 or any(value <= 0 for value in args.spacing_yx):
            raise ValueError("evaluation.spacing_yx must contain two positive values")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    config, runtime_config_path = load_evaluation_config(config_path)
    args = apply_config_defaults(args, config)
    if ndi is None:
        raise RuntimeError(
            "SciPy is required for HD95. Install it with `pip install scipy` "
            "before running this evaluation."
        )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer, config, stage, checkpoint_payload = build_runtime(
        runtime_config_path, config, checkpoint_path, args.stage, args.device
    )
    base_dataset = build_patient_dataset(
        config,
        split=args.split,
        data_mode=args.data_mode,
        max_patients=args.max_patients,
        mr_max_patients=args.mr_max_patients,
        us_max_patients=args.us_max_patients,
        max_slices_per_patient=args.max_slices_per_patient,
        sampling_mode=args.patient_sampling_mode,
        seed=args.patient_seed,
        # A checkpoint carries the exact train-derived thresholds. Prefer these
        # over potentially stale or relocated paths in the YAML.
        area_thresholds=checkpoint_payload.get("area_thresholds") or None,
        boundary_thresholds=checkpoint_payload.get("boundary_thresholds") or None,
    )
    dataset_preflight = preflight_dataset(
        base_dataset,
        prompt_modes=args.prompt_modes,
        require_ground_truth=args.require_ground_truth,
    )
    patient_selection = selected_patient_state(base_dataset)
    run_config = {
        "evaluation_config": str(config_path),
        "base_config": str(runtime_config_path),
        "checkpoint": str(checkpoint_path),
        "stage": stage,
        "split": args.split,
        "data_mode": args.data_mode,
        "prompt_modes": args.prompt_modes,
        "max_patients": args.max_patients,
        "mr_max_patients": args.mr_max_patients,
        "us_max_patients": args.us_max_patients,
        "max_slices_per_patient": args.max_slices_per_patient,
        "max_samples": args.max_samples,
        "threshold": args.threshold,
        "small_area_threshold": args.small_area_threshold,
        "box_noise_std": args.box_noise_std,
        "box_noise_max": args.box_noise_max,
        "box_noise_seed": args.box_noise_seed,
        "box_noise_note": (
            "Gaussian XYXY coordinate noise is applied only in box/text_box modes; "
            "std is relative to box width/height and max is measured in pixels."
        ),
        "coarse_box": {
            "expand_min": args.coarse_box_expand_min,
            "expand_max": args.coarse_box_expand_max,
            "jitter_std": args.coarse_box_jitter_std,
            "jitter_max": args.coarse_box_jitter_max,
            "note": "Uses the same expanded/jittered transform as training.",
        },
        "small_target_rule": (
            "non-empty area_label==0 when checkpoint/config area thresholds are "
            "available; otherwise 0<area_ratio<=small_area_threshold"
        ),
        "spacing_yx": args.spacing_yx,
        "require_ground_truth": args.require_ground_truth,
        "dataset_preflight": dataset_preflight,
        "selected_patients": patient_selection,
        "none_mode_note": (
            "The caller supplies only an image; SAM3 internally receives fixed "
            f"task token {args.none_text!r} and no sample-specific text/box prompt."
        ),
        "hd95_unit": "mm" if args.spacing_yx is not None else "pixel",
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(_json_ready(run_config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    all_rows: List[Dict[str, Any]] = []
    expected_sample_keys: Optional[List[Tuple[str, str, str]]] = None
    for prompt_mode in args.prompt_modes:
        print(f"[start] prompt_mode={prompt_mode}, samples={len(base_dataset)}")
        rows = evaluate_prompt_mode(trainer, base_dataset, prompt_mode, args, output_dir)
        sample_keys = [_sample_key(row) for row in rows]
        if expected_sample_keys is None:
            expected_sample_keys = sample_keys
        elif sample_keys != expected_sample_keys:
            raise RuntimeError(
                f"Prompt mode {prompt_mode!r} evaluated a different ordered sample set"
            )
        write_csv(output_dir / f"metrics_per_image_{prompt_mode}.csv", rows)
        all_rows.extend(rows)

    all_rows = sorted(all_rows, key=_row_key)
    patient_rows = summarize_patients(all_rows)
    summary = summarize(all_rows, patient_rows)
    summary.update(
        {
            "selected_patients": patient_selection,
            "dataset_preflight": dataset_preflight,
            "num_prompt_modes": len(args.prompt_modes),
            "prompt_modes": list(args.prompt_modes),
            "same_samples_for_all_prompt_modes": True,
            "hd95_unit": "mm" if args.spacing_yx is not None else "pixel",
            "hd95_note": (
                "HD95 is undefined (null in JSON/blank or nan in CSV) when exactly "
                "one of prediction/GT is empty; num_valid_hd95 reports its finite count."
            ),
            "patient_macro_note": (
                "patient_macro first averages 2D slice metrics within each patient, "
                "then gives every patient equal weight. It is not a reconstructed "
                "3D-volume Dice or HD95. Per-modality patient-macro results are "
                "reported under by_prompt_mode.<mode>.by_modality."
            ),
            "none_mode_note": (
                "SAM3 has no native empty-query image-grounding path. Mode 'none' "
                "means the caller supplies only an image; the model inserts fixed "
                f"task token {args.none_text!r} without any sample-specific prompt."
            ),
        }
    )
    write_csv(output_dir / "metrics_per_image.csv", all_rows)
    write_csv(output_dir / "metrics_per_patient.csv", patient_rows)
    write_csv(
        output_dir / "summary_patient_macro.csv",
        patient_macro_summary_rows(summary),
    )
    (output_dir / "summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[done] wrote metrics for {len(all_rows)} prompt/sample rows to {output_dir}")
    for prompt_mode, block in summary["by_prompt_mode"].items():
        all_metrics = block["all_samples"]
        small_metrics = block["small_samples"]
        patient_metrics = block["patient_macro"]
        print(
            f"[summary][{prompt_mode}][slice-macro][overall] "
            f"dice={all_metrics['dice']:.4f}, "
            f"hd95={all_metrics['hd95']:.4f}; "
            f"small dice={small_metrics['dice']:.4f}, hd95={small_metrics['hd95']:.4f}"
        )
        print(
            f"[summary][{prompt_mode}][patient-macro][overall] "
            f"patients={patient_metrics['num_patients_evaluated']}, "
            f"dice={patient_metrics['dice']:.4f}, "
            f"hd95={patient_metrics['hd95']:.4f}"
        )
        for modality, modality_block in block["by_modality"].items():
            modality_metrics = modality_block["patient_macro"]
            print(
                f"[summary][{prompt_mode}][patient-macro][{modality}] "
                f"patients={modality_metrics['num_patients_evaluated']}, "
                f"dice={modality_metrics['dice']:.4f}, "
                f"hd95={modality_metrics['hd95']:.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
