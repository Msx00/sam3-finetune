"""ROI adapter that uses the original, structurally unchanged SvANet."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import ndimage
from torch import nn
from torch.nn import functional as F

from .moe_losses import dice_bce_with_logits


def build_original_svanet(
    svanet_root: str | Path,
    checkpoint: Optional[str] = None,
    backbone_checkpoint: Optional[str] = None,
    input_size: Sequence[int] = (512, 512),
    model_name: str = "resnet50",
    seg_feature_guide: int = 2,
    device: Optional[torch.device | str] = None,
) -> nn.Module:
    """Build SvANet through its own registry; no SvANet layer is modified."""
    root = Path(svanet_root).resolve()
    if not (root / "opts.py").is_file():
        raise FileNotFoundError(f"Invalid SvANet source root: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # SvANet's registry derives import names from cwd and fails on Windows when
    # embedded from a sibling project (it interprets ``C:`` as a module name).
    # Import under SvANet's own cwd, then restore the caller's cwd. This changes
    # no SvANet layer or forward implementation.
    previous_cwd = Path.cwd()
    try:
        os.chdir(root)
        from opts import Opts
        import lib.model.classification.netResNet  # noqa: F401
        import lib.model.segmentation.encDecoder  # noqa: F401
        import lib.model.segmentation.heads.deepLabv3  # noqa: F401
        from lib.model import getModel
    finally:
        os.chdir(previous_cwd)

    opt = Opts().parse([])
    opt.model_name = str(model_name).lower()
    opt.seg_model_name = "encoder_decoder"
    opt.seg_head_name = "deeplabv3"
    opt.seg_feature_guide = int(seg_feature_guide)
    opt.resize_shape = int(input_size[0])
    opt.num_classes = opt.cls_num_classes = opt.seg_num_classes = 2
    opt.device = torch.device(device or "cpu")
    model = getModel(opt=opt)
    if model is None:
        raise RuntimeError("SvANet registry failed to construct the segmentation model")
    if backbone_checkpoint:
        payload = torch.load(
            backbone_checkpoint, map_location="cpu", weights_only=True
        )
        if isinstance(payload, dict) and "state_dict" in payload:
            payload = payload["state_dict"]
        if not isinstance(payload, dict):
            raise ValueError(
                f"Unsupported ResNet backbone checkpoint: {backbone_checkpoint}"
            )
        encoder_state = model.Encoder.state_dict()
        # SvANet names torchvision stem/stages as Conv1/Layer1... and keeps
        # BatchNorm/ReLU/MaxPool inside Conv1. Map by tensor shape and role.
        mapped = _map_torchvision_resnet50_to_svanet(payload, encoder_state)
        missing, unexpected = model.Encoder.load_state_dict(mapped, strict=False)
        if not mapped:
            raise RuntimeError("No ResNet-50 backbone tensors matched SvANet encoder")
        print(
            f"Loaded SvANet ResNet-50 backbone: matched={len(mapped)}, "
            f"encoder_missing={len(missing)}, unexpected={len(unexpected)}"
        )
    if checkpoint:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            for key in ("svanet_state", "model_state", "state_dict", "model"):
                if key in payload and isinstance(payload[key], dict):
                    payload = payload[key]
                    break
        if not isinstance(payload, dict):
            raise ValueError(f"Unsupported SvANet checkpoint format: {checkpoint}")
        state = {key.removeprefix("module."): value for key, value in payload.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                f"Loaded SvANet checkpoint non-strictly: "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
    return model


def _map_torchvision_resnet50_to_svanet(
    source: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Map torchvision ResNet-50 keys without changing the SvANet encoder."""
    mapped: Dict[str, torch.Tensor] = {}
    direct_prefixes = {
        "layer1.": "Layer1.1.", "layer2.": "Layer2.",
        "layer3.": "Layer3.", "layer4.": "Layer4.",
    }
    for source_name, value in source.items():
        candidates = []
        if source_name == "conv1.weight":
            candidates = ["Conv1.Conv.weight"]
        elif source_name.startswith("bn1."):
            suffix = source_name.removeprefix("bn1.")
            candidates = [f"Conv1.Bn.{suffix}"]
        else:
            for old, new in direct_prefixes.items():
                if source_name.startswith(old):
                    candidate = new + source_name.removeprefix(old)
                    for index in (1, 2, 3):
                        candidate = candidate.replace(
                            f".conv{index}.", f".Conv{index}.Conv."
                        ).replace(f".bn{index}.", f".Conv{index}.Bn.")
                    candidate = candidate.replace(
                        ".downsample.0.", ".Downsample.Conv."
                    ).replace(".downsample.1.", ".Downsample.Bn.")
                    candidates = [candidate]
                    break
        for candidate in candidates:
            if candidate in target and target[candidate].shape == value.shape:
                mapped[candidate] = value
                break
    return mapped


def largest_component_bbox(mask: torch.Tensor) -> Optional[Tuple[int, int, int, int]]:
    """Return an exclusive XYXY bbox for the largest 8-connected component."""
    array = mask.detach().to(device="cpu", dtype=torch.bool).numpy()
    labels, count = ndimage.label(array, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return None
    sizes = np.bincount(labels.reshape(-1))
    sizes[0] = 0
    selected = labels == int(sizes.argmax())
    ys, xs = np.nonzero(selected)
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def expand_and_clip_roi(
    box: Sequence[float],
    image_height: int,
    image_width: int,
    expand_ratio: float = 0.25,
    min_roi_size: int = 32,
) -> Tuple[int, int, int, int]:
    """Expand XYXY by ratio, enforce a minimum size, and clip to image bounds."""
    x1, y1, x2, y2 = (float(value) for value in box)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid ROI box: {box}")
    width, height = x2 - x1, y2 - y1
    x1, x2 = x1 - width * expand_ratio, x2 + width * expand_ratio
    y1, y2 = y1 - height * expand_ratio, y2 + height * expand_ratio

    def extent(start: float, end: float, limit: int) -> Tuple[int, int]:
        size = min(max(float(min_roi_size), end - start), float(limit))
        center = 0.5 * (start + end)
        start, end = center - size / 2.0, center + size / 2.0
        if start < 0:
            end -= start
            start = 0.0
        if end > limit:
            start -= end - limit
            end = float(limit)
        return int(np.floor(max(0.0, start))), int(np.ceil(min(float(limit), end)))

    x1i, x2i = extent(x1, x2, image_width)
    y1i, y2i = extent(y1, y2, image_height)
    return x1i, y1i, x2i, y2i


def select_small_triggers(
    area_logits: torch.Tensor,
    training: bool,
    train_trigger: str = "teacher_forcing",
    area_labels: Optional[torch.Tensor] = None,
    teacher_area_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Inference always uses predicted area; training supports all requested modes."""
    predicted = area_logits.argmax(dim=-1) == 0
    if not training:
        return predicted
    mode = str(train_trigger).lower()
    if mode == "pred":
        return predicted
    if area_labels is None:
        raise ValueError(f"area_labels are required for train_trigger={mode!r}")
    ground_truth = area_labels.to(area_logits.device, dtype=torch.long) == 0
    if mode == "gt":
        return ground_truth
    if mode == "gt_or_pred":
        return ground_truth | predicted
    if mode == "teacher_forcing":
        if teacher_area_mask is None:
            raise ValueError("teacher_area_mask is required for teacher_forcing trigger")
        teacher = teacher_area_mask.to(area_logits.device, dtype=torch.bool)
        return torch.where(teacher, ground_truth, predicted)
    raise ValueError(f"Unknown SvANet train_trigger: {train_trigger!r}")


class SvANetROIAdapter(nn.Module):
    """Crop original images, run unchanged SvANet, and paste ROI logits back."""

    def __init__(
        self,
        svanet: nn.Module,
        input_size: Sequence[int] = (512, 512),
        roi_expand_ratio: float = 0.25,
        min_roi_size: int = 32,
        mask_threshold: float = 0.5,
        empty_mask_fallback: str = "box_then_full_image",
        paste_mode: str = "replace_roi",
        outside_roi: str = "zero",
        train_trigger: str = "teacher_forcing",
    ) -> None:
        super().__init__()
        if empty_mask_fallback != "box_then_full_image":
            raise ValueError("Only empty_mask_fallback=box_then_full_image is supported")
        if paste_mode not in {"replace_roi", "blend_with_sam3"}:
            raise ValueError("paste_mode must be replace_roi or blend_with_sam3")
        if outside_roi not in {"zero", "sam3"}:
            raise ValueError("outside_roi must be zero or sam3")
        self.svanet = svanet
        self.input_size = tuple(int(value) for value in input_size)
        self.roi_expand_ratio = float(roi_expand_ratio)
        self.min_roi_size = int(min_roi_size)
        self.mask_threshold = float(mask_threshold)
        self.empty_mask_fallback = empty_mask_fallback
        self.paste_mode = paste_mode
        self.outside_roi = outside_roi
        self.train_trigger = train_trigger
        self.reset_runtime_stats()

    def reset_runtime_stats(self) -> None:
        self.runtime_stats = {
            "trigger_count": 0,
            "empty_mask_count": 0,
            "box_fallback_count": 0,
            "full_image_fallback_count": 0,
        }

    def _zero_refine_loss(self, images: torch.Tensor) -> torch.Tensor:
        parameter = next(self.svanet.parameters(), None)
        if parameter is not None:
            return parameter.sum() * 0.0
        return images.sum() * 0.0

    @staticmethod
    def _foreground_logits(output: Any) -> torch.Tensor:
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise ValueError("SvANet must return Tensor[B,C,H,W] or tuple beginning with it")
        if output.shape[1] == 1:
            return output[:, 0]
        if output.shape[1] == 2:
            return output[:, 1] - output[:, 0]
        raise ValueError(f"SvANet output must have 1 or 2 channels, got {output.shape[1]}")

    def _svanet_forward(self, model_input: torch.Tensor) -> Any:
        """Keep BatchNorm stable when a smoke/medical batch has one small ROI."""
        if model_input.shape[0] != 1 or not self.svanet.training:
            return self.svanet(model_input)
        batch_norms = [
            module for module in self.svanet.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.training
        ]
        for module in batch_norms:
            module.eval()
        try:
            return self.svanet(model_input)
        finally:
            for module in batch_norms:
                module.train()

    @staticmethod
    def _prompt_box(
        boxes: Optional[torch.Tensor],
        image_height: int,
        image_width: int,
        prompt_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[List[float]]:
        if boxes is None or boxes.numel() == 0:
            return None
        candidates = boxes.reshape(-1, 4).detach().cpu().float()
        valid = candidates[(candidates[:, 2] > candidates[:, 0]) & (candidates[:, 3] > candidates[:, 1])]
        if not len(valid):
            return None
        areas = (valid[:, 2] - valid[:, 0]) * (valid[:, 3] - valid[:, 1])
        box = valid[int(areas.argmax())].clone()
        if float(box.max()) <= 1.0:
            box *= torch.tensor([image_width, image_height, image_width, image_height])
        elif prompt_size is not None:
            prompt_height, prompt_width = prompt_size
            box *= torch.tensor(
                [
                    image_width / prompt_width,
                    image_height / prompt_height,
                    image_width / prompt_width,
                    image_height / prompt_height,
                ]
            )
        return box.tolist()

    def forward(
        self,
        images: torch.Tensor,
        sam3_logits: torch.Tensor,
        area_logits: torch.Tensor,
        area_labels: Optional[torch.Tensor] = None,
        box_prompts: Optional[Sequence[Optional[torch.Tensor]]] = None,
        box_prompt_sizes: Optional[Sequence[Optional[Tuple[int, int]]]] = None,
        gt_masks: Optional[torch.Tensor] = None,
        teacher_area_mask: Optional[torch.Tensor] = None,
        use_gt_roi: bool = False,
    ) -> Dict[str, Any]:
        if images.ndim != 4 or sam3_logits.ndim not in {3, 4}:
            raise ValueError("Expected images[B,C,H,W] and sam3_logits[B,H,W]")
        if sam3_logits.ndim == 4:
            if sam3_logits.shape[1] != 1:
                raise ValueError("4D sam3_logits must have one channel")
            sam3_logits = sam3_logits[:, 0]
        batch, _, height, width = images.shape
        if sam3_logits.shape[0] != batch or area_logits.shape[0] != batch:
            raise ValueError("Image, SAM3 logits and area logits batch sizes must match")
        if sam3_logits.shape[-2:] != (height, width):
            sam3_logits = F.interpolate(
                sam3_logits[:, None].float(), (height, width), mode="bilinear", align_corners=False
            )[:, 0].to(images.dtype)
        trigger = select_small_triggers(
            area_logits, self.training, self.train_trigger,
            area_labels=area_labels, teacher_area_mask=teacher_area_mask,
        )
        trigger_indices = trigger.nonzero(as_tuple=False).flatten().tolist()
        prompts = box_prompts or [None] * batch
        prompt_sizes = box_prompt_sizes or [None] * batch
        if len(prompts) != batch:
            raise ValueError("box_prompts must contain one entry per image")
        if len(prompt_sizes) != batch:
            raise ValueError("box_prompt_sizes must contain one entry per image")
        if gt_masks is not None and gt_masks.shape[-2:] != (height, width):
            gt_masks = F.interpolate(
                gt_masks[:, None].float(), (height, width), mode="nearest"
            )[:, 0]

        rois: List[Tuple[int, int, int, int]] = []
        roi_images: List[torch.Tensor] = []
        roi_targets: List[torch.Tensor] = []
        sources: List[str] = []
        for index in trigger_indices:
            source_mask = (
                gt_masks[index].bool()
                if use_gt_roi and gt_masks is not None
                else sam3_logits[index].sigmoid() > self.mask_threshold
            )
            box = largest_component_bbox(source_mask)
            source = "gt_mask" if use_gt_roi and gt_masks is not None else "sam3_mask"
            if box is None:
                self.runtime_stats["empty_mask_count"] += 1
                prompt = self._prompt_box(
                    prompts[index], height, width, prompt_sizes[index]
                )
                if prompt is not None:
                    box, source = tuple(prompt), "box_fallback"
                    self.runtime_stats["box_fallback_count"] += 1
                else:
                    box, source = (0, 0, width, height), "full_image_fallback"
                    self.runtime_stats["full_image_fallback_count"] += 1
            roi = expand_and_clip_roi(
                box, height, width, self.roi_expand_ratio, self.min_roi_size
            )
            x1, y1, x2, y2 = roi
            crop = images[index : index + 1, :, y1:y2, x1:x2]
            if crop.shape[1] == 1:
                crop = crop.repeat(1, 3, 1, 1)
            elif crop.shape[1] != 3:
                crop = crop[:, :3]
            crop = F.interpolate(
                crop.float(), self.input_size, mode="bilinear", align_corners=False
            )[0]
            rois.append(roi)
            roi_images.append(crop)
            sources.append(source)
            if gt_masks is not None:
                target = gt_masks[index : index + 1, None, y1:y2, x1:x2].float()
                roi_targets.append(F.interpolate(target, self.input_size, mode="nearest")[0, 0])

        final_logits = sam3_logits.clone()
        roi_logits_list: List[torch.Tensor] = []
        debug_enabled = bool(os.environ.get("SVANET_DEBUG_MODE", "").strip())
        rank = os.environ.get("RANK", "0")
        if roi_images:
            model_input = torch.stack(roi_images).to(images.device, dtype=images.dtype)
            targets = torch.stack(roi_targets).float() if roi_targets else None
            input_finite = bool(torch.isfinite(model_input).all())
            targets_finite = (
                bool(torch.isfinite(targets).all()) if targets is not None else True
            )
            if debug_enabled:
                target_summary = (
                    f"target_finite={targets_finite} "
                    f"target_min={targets.nan_to_num().min().item():.6f} "
                    f"target_max={targets.nan_to_num().max().item():.6f} "
                    f"target_fg_ratio={targets.nan_to_num().mean().item():.8f}"
                    if targets is not None
                    else "target_finite=True target=unavailable"
                )
                print(
                    f"[SVANET-DEBUG][rank={rank}] "
                    f"roi_count={len(roi_images)} "
                    f"input_shape={tuple(model_input.shape)} "
                    f"input_finite={input_finite} "
                    f"input_min={model_input.nan_to_num().min().item():.6f} "
                    f"input_max={model_input.nan_to_num().max().item():.6f} "
                    f"{target_summary}",
                    flush=True,
                )
            if not input_finite:
                raise FloatingPointError("SvANet ROI input contains NaN/Inf")
            if not targets_finite:
                raise FloatingPointError("SvANet ROI target contains NaN/Inf")
            if targets is not None and (
                bool((targets < 0).any()) or bool((targets > 1).any())
            ):
                raise ValueError(
                    "SvANet ROI target must be binary/in [0, 1], got "
                    f"[{targets.min().item()}, {targets.max().item()}]"
                )
            roi_logits = self._foreground_logits(self._svanet_forward(model_input))
            logits_finite = bool(torch.isfinite(roi_logits).all())
            if debug_enabled:
                print(
                    f"[SVANET-DEBUG][rank={rank}] "
                    f"logits_shape={tuple(roi_logits.shape)} "
                    f"logits_finite={logits_finite} "
                    f"logits_min={roi_logits.nan_to_num().min().item():.6f} "
                    f"logits_max={roi_logits.nan_to_num().max().item():.6f} "
                    f"logits_mean={roi_logits.nan_to_num().mean().item():.6f} "
                    f"logits_abs_max={roi_logits.nan_to_num().abs().max().item():.6f}",
                    flush=True,
                )
            if not logits_finite:
                raise FloatingPointError("SvANet forward produced NaN/Inf logits")
            if roi_logits.shape[-2:] != self.input_size:
                roi_logits = F.interpolate(
                    roi_logits[:, None].float(), self.input_size,
                    mode="bilinear", align_corners=False,
                )[:, 0].to(images.dtype)
            for local_index, image_index in enumerate(trigger_indices):
                x1, y1, x2, y2 = rois[local_index]
                resized = F.interpolate(
                    roi_logits[local_index : local_index + 1, None].float(),
                    (y2 - y1, x2 - x1), mode="bilinear", align_corners=False,
                )[0, 0].to(images.dtype)
                if self.paste_mode == "replace_roi":
                    if self.outside_roi == "zero":
                        final_logits[image_index] = torch.full_like(final_logits[image_index], -20.0)
                    final_logits[image_index, y1:y2, x1:x2] = resized
                else:
                    base = final_logits[image_index, y1:y2, x1:x2]
                    final_logits[image_index, y1:y2, x1:x2] = 0.5 * (base + resized)
                roi_logits_list.append(roi_logits[local_index])
            refine_loss = (
                dice_bce_with_logits(
                    roi_logits,
                    torch.stack(roi_targets),
                    debug_label="refine",
                )
                if gt_masks is not None else self._zero_refine_loss(images)
            )
        else:
            refine_loss = self._zero_refine_loss(images)
            if debug_enabled:
                print(
                    f"[SVANET-DEBUG][rank={rank}] "
                    f"roi_count=0 refine_loss={refine_loss.detach().item():.8f}",
                    flush=True,
                )
            if not torch.isfinite(refine_loss):
                raise FloatingPointError(
                    "Zero-trigger SvANet refine loss is NaN/Inf; "
                    "the SvANet parameters are already contaminated"
                )

        self.runtime_stats["trigger_count"] += len(trigger_indices)
        widths = [roi[2] - roi[0] for roi in rois]
        heights = [roi[3] - roi[1] for roi in rois]
        return {
            "final_logits": final_logits,
            "refine_loss": refine_loss,
            "trigger_mask": trigger,
            "roi_boxes": rois,
            "roi_sources": sources,
            "roi_images": roi_images,
            "roi_gt": roi_targets,
            "svanet_roi_logits": roi_logits_list,
            "batch_stats": {
                "small_count": int(trigger.sum().item()),
                "trigger_count": len(trigger_indices),
                "empty_mask_count": sum(source.endswith("fallback") for source in sources),
                "box_fallback_count": sources.count("box_fallback"),
                "full_image_fallback_count": sources.count("full_image_fallback"),
                "mean_roi_width": float(np.mean(widths)) if widths else 0.0,
                "mean_roi_height": float(np.mean(heights)) if heights else 0.0,
            },
        }
