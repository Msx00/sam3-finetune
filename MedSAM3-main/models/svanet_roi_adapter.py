"""Safe ROI refinement with the original, structurally unchanged SvANet.

The adapter deliberately keeps ROI proposal and logit fusion outside SvANet:
SvANet still receives an ordinary image crop and its architecture is untouched.
This module only decides whether a crop is reliable enough to refine and how to
merge the resulting foreground logits with SAM3.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

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


def _largest_component_candidate(
    probabilities: torch.Tensor,
    threshold: float,
    min_confidence: float,
    min_component_pixels: int,
    max_component_fraction: float,
) -> Tuple[Optional[Tuple[int, int, int, int]], Dict[str, Any]]:
    """Return a bbox only when the largest foreground component is reliable.

    ``min_confidence`` is the mean foreground probability inside the selected
    component.  Pixel-count and image-fraction guards reject isolated spikes and
    degenerate almost-full-image predictions, respectively.  Selection is a
    non-differentiable control decision by design, so it is evaluated on CPU.
    """
    if probabilities.ndim != 2:
        raise ValueError(
            "ROI proposal probabilities must be [H,W], got "
            f"{tuple(probabilities.shape)}"
        )
    probability_array = (
        probabilities.detach().to(device="cpu", dtype=torch.float32).numpy()
    )
    foreground = probability_array > float(threshold)
    labels, count = ndimage.label(
        foreground, structure=np.ones((3, 3), dtype=np.uint8)
    )
    stats: Dict[str, Any] = {
        "confidence": 0.0,
        "component_pixels": 0,
        "component_fraction": 0.0,
        "reason": "empty",
    }
    if count == 0:
        return None, stats

    sizes = np.bincount(labels.reshape(-1))
    sizes[0] = 0
    component_id = int(sizes.argmax())
    selected = labels == component_id
    pixels = int(selected.sum())
    fraction = float(pixels / max(1, selected.size))
    confidence = float(probability_array[selected].mean())
    stats.update(
        confidence=confidence,
        component_pixels=pixels,
        component_fraction=fraction,
    )
    if pixels < int(min_component_pixels):
        stats["reason"] = "too_small"
        return None, stats
    if fraction > float(max_component_fraction):
        stats["reason"] = "too_large"
        return None, stats
    if confidence < float(min_confidence):
        stats["reason"] = "low_confidence"
        return None, stats

    ys, xs = np.nonzero(selected)
    stats["reason"] = "accepted"
    return (
        int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
    ), stats


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
    """Crop original images, run unchanged SvANet, and safely fuse ROI logits.

    The legacy destructive behavior remains available through
    ``paste_mode="replace_roi", outside_roi="zero"``.  New callers default to
    convex logit blending and preserve the SAM3 prediction outside every ROI.
    """

    _FALLBACK_CHAINS = {
        "locator_then_box_then_skip": ("locator", "box", "skip"),
        "locator_then_box_then_full_image": (
            "locator", "box", "full_image",
        ),
        "locator_then_skip": ("locator", "skip"),
        "locator_then_full_image": ("locator", "full_image"),
        "box_then_skip": ("box", "skip"),
        # Kept for checkpoints/configs produced by the original implementation.
        "box_then_full_image": ("box", "full_image"),
        "skip": ("skip",),
        "full_image": ("full_image",),
    }

    def __init__(
        self,
        svanet: nn.Module,
        input_size: Sequence[int] = (512, 512),
        roi_expand_ratio: float = 0.25,
        min_roi_size: int = 32,
        mask_threshold: float = 0.5,
        empty_mask_fallback: str = "locator_then_box_then_skip",
        paste_mode: str = "blend_with_sam3",
        outside_roi: str = "sam3",
        train_trigger: str = "teacher_forcing",
        fusion_weight: float = 0.5,
        residual_scale: float = 0.25,
        sam3_min_confidence: float = 0.0,
        locator_threshold: float = 0.5,
        locator_min_confidence: float = 0.0,
        min_component_pixels: int = 1,
        max_component_fraction: float = 1.0,
        min_area_confidence: float = 0.0,
        confidence_gate_during_training: bool = False,
        roi_chunk_size: int = 0,
        max_roi_per_step: int = 0,
    ) -> None:
        super().__init__()
        if empty_mask_fallback not in self._FALLBACK_CHAINS:
            choices = ", ".join(sorted(self._FALLBACK_CHAINS))
            raise ValueError(
                f"Unknown empty_mask_fallback={empty_mask_fallback!r}; "
                f"choose one of: {choices}"
            )
        if paste_mode not in {"replace_roi", "blend_with_sam3", "residual"}:
            raise ValueError(
                "paste_mode must be replace_roi, blend_with_sam3, or residual"
            )
        if outside_roi not in {"zero", "sam3"}:
            raise ValueError("outside_roi must be zero or sam3")
        if not 0.0 <= float(fusion_weight) <= 1.0:
            raise ValueError("fusion_weight must be in [0, 1]")
        if float(residual_scale) < 0.0:
            raise ValueError("residual_scale must be non-negative")
        for name, value in {
            "mask_threshold": mask_threshold,
            "sam3_min_confidence": sam3_min_confidence,
            "locator_threshold": locator_threshold,
            "locator_min_confidence": locator_min_confidence,
            "max_component_fraction": max_component_fraction,
            "min_area_confidence": min_area_confidence,
        }.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if int(min_component_pixels) < 1:
            raise ValueError("min_component_pixels must be positive")
        if int(roi_chunk_size) < 0:
            raise ValueError("roi_chunk_size must be non-negative (0 = all crops)")
        if int(max_roi_per_step) < 0:
            raise ValueError("max_roi_per_step must be non-negative (0 = unlimited)")
        self.svanet = svanet
        self.input_size = tuple(int(value) for value in input_size)
        self.roi_expand_ratio = float(roi_expand_ratio)
        self.min_roi_size = int(min_roi_size)
        self.mask_threshold = float(mask_threshold)
        self.empty_mask_fallback = empty_mask_fallback
        self.paste_mode = paste_mode
        self.outside_roi = outside_roi
        self.train_trigger = train_trigger
        self.fusion_weight = float(fusion_weight)
        self.residual_scale = float(residual_scale)
        self.sam3_min_confidence = float(sam3_min_confidence)
        self.locator_threshold = float(locator_threshold)
        self.locator_min_confidence = float(locator_min_confidence)
        self.min_component_pixels = int(min_component_pixels)
        self.max_component_fraction = float(max_component_fraction)
        self.min_area_confidence = float(min_area_confidence)
        self.confidence_gate_during_training = bool(
            confidence_gate_during_training
        )
        # SvANet refinement is the memory peak of a training step: every
        # triggered image contributes a full-resolution crop, and running the
        # whole batch through the 202M-parameter encoder/decoder at once can
        # exhaust device memory (observed as an NVML assertion inside the CUDA
        # caching allocator).  ``roi_chunk_size`` splits that forward into
        # sub-batches so peak activation memory stays bounded, and
        # ``max_roi_per_step`` caps how many crops are refined per step.
        self.roi_chunk_size = int(roi_chunk_size)
        self.max_roi_per_step = int(max_roi_per_step)
        self.reset_runtime_stats()

    def reset_runtime_stats(self) -> None:
        self.runtime_stats = {
            "trigger_count": 0,
            "empty_mask_count": 0,
            "box_fallback_count": 0,
            "locator_fallback_count": 0,
            "full_image_fallback_count": 0,
            "unreliable_mask_count": 0,
            "low_area_confidence_skip_count": 0,
            "no_reliable_roi_skip_count": 0,
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

    @contextlib.contextmanager
    def _singleton_batch_norms(self, sub_batch_sizes: Sequence[int]) -> Iterator[None]:
        """Keep BatchNorm usable when a forward sub-batch holds one crop.

        ``F.batch_norm`` rejects a sub-batch of one as soon as a branch has
        pooled the feature map down to 1x1, which SvANet's ASPP global branch
        does for every crop.  ``roi_chunk_size=1`` therefore turns each chunk
        into exactly that forbidden shape.  Whenever a forward pass contains
        such a chunk, the BatchNorm layers run in eval mode for that whole pass
        and reuse their pretrained running statistics, so the pass cannot
        change behaviour half-way through.  The affine weights still receive
        gradients; only the running statistics stay frozen for that pass.
        Passes without single-crop chunks and inference are left untouched.
        """
        if 1 not in sub_batch_sizes or not self.svanet.training:
            yield
            return
        batch_norms = [
            module for module in self.svanet.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.training
        ]
        for module in batch_norms:
            module.eval()
        try:
            yield
        finally:
            for module in batch_norms:
                module.train()

    def _svanet_forward(self, model_input: torch.Tensor) -> Any:
        """Forward ROI crops through SvANet with the memory/batch guards applied."""
        return self._run_svanet(model_input)

    def _run_svanet(self, model_input: torch.Tensor) -> Any:
        """Run SvANet with bounded memory; math is identical to one big call.

        Splitting the batch into sub-batches keeps the graph connected, so
        gradients still accumulate across chunks exactly as in a single
        forward. Chunks are only used while gradients are tracked; inference
        keeps the original single-call path. Sub-batches of one crop run with
        BatchNorm in eval mode (see :meth:`_singleton_batch_norms`).
        """
        chunk_size = self.roi_chunk_size
        batch = model_input.shape[0]
        if chunk_size <= 0 or chunk_size >= batch or not torch.is_grad_enabled():
            chunks = [model_input]
        else:
            chunks = [
                model_input[start : start + chunk_size]
                for start in range(0, batch, chunk_size)
            ]
        with self._singleton_batch_norms([chunk.shape[0] for chunk in chunks]):
            outputs = [self.svanet(chunk) for chunk in chunks]
        if len(outputs) == 1:
            return outputs[0]
        if all(isinstance(output, torch.Tensor) for output in outputs):
            return torch.cat(outputs, dim=0)
        leading = [output[0] for output in outputs]
        if all(isinstance(item, torch.Tensor) for item in leading):
            merged = list(outputs[0])
            merged[0] = torch.cat(leading, dim=0)
            return tuple(merged) if isinstance(outputs[0], tuple) else merged
        raise ValueError(
            "Chunked SvANet forward requires tensor outputs or tuples beginning "
            "with a tensor; set svanet.roi_chunk_size=0 to disable chunking"
        )

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
        locator_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if images.ndim != 4 or sam3_logits.ndim not in {3, 4}:
            raise ValueError("Expected images[B,C,H,W] and sam3_logits[B,H,W]")
        if sam3_logits.ndim == 4:
            if sam3_logits.shape[1] != 1:
                raise ValueError("4D sam3_logits must have one channel")
            sam3_logits = sam3_logits[:, 0]
        batch, _, height, width = images.shape
        if area_logits.ndim != 2 or area_logits.shape[1] != 3:
            raise ValueError(
                "area_logits must have shape [B,3], got "
                f"{tuple(area_logits.shape)}"
            )
        if sam3_logits.shape[0] != batch or area_logits.shape[0] != batch:
            raise ValueError("Image, SAM3 logits and area logits batch sizes must match")
        for name, values in (
            ("area_labels", area_labels),
            ("teacher_area_mask", teacher_area_mask),
        ):
            if values is not None and (values.ndim != 1 or values.shape[0] != batch):
                raise ValueError(
                    f"{name} must have shape [B], got {tuple(values.shape)}"
                )
        if sam3_logits.shape[-2:] != (height, width):
            sam3_logits = F.interpolate(
                sam3_logits[:, None].float(), (height, width), mode="bilinear", align_corners=False
            )[:, 0].to(images.dtype)
        if gt_masks is not None:
            if gt_masks.ndim == 4:
                if gt_masks.shape[1] != 1:
                    raise ValueError("4D gt_masks must have one channel")
                gt_masks = gt_masks[:, 0]
            if gt_masks.ndim != 3 or gt_masks.shape[0] != batch:
                raise ValueError("gt_masks must be [B,H,W] or [B,1,H,W]")
        if locator_logits is not None:
            if locator_logits.ndim == 4:
                if locator_logits.shape[1] != 1:
                    raise ValueError("4D locator_logits must have one channel")
                locator_logits = locator_logits[:, 0]
            if locator_logits.ndim != 3 or locator_logits.shape[0] != batch:
                raise ValueError(
                    "locator_logits must be [B,H,W] or [B,1,H,W]"
                )
            if locator_logits.shape[-2:] != (height, width):
                locator_logits = F.interpolate(
                    locator_logits[:, None].float(),
                    (height, width),
                    mode="bilinear",
                    align_corners=False,
                )[:, 0].to(images.dtype)

        requested_trigger = select_small_triggers(
            area_logits, self.training, self.train_trigger,
            area_labels=area_labels, teacher_area_mask=teacher_area_mask,
        )
        area_small_probability = area_logits.float().softmax(dim=-1)[:, 0]
        confidence_gate = (
            not self.training or self.confidence_gate_during_training
        )
        low_area_confidence = (
            requested_trigger
            & confidence_gate
            & (area_small_probability < self.min_area_confidence)
        )
        trigger = requested_trigger & ~low_area_confidence
        self.runtime_stats["low_area_confidence_skip_count"] += int(
            low_area_confidence.sum().item()
        )
        trigger_indices = trigger.nonzero(as_tuple=False).flatten().tolist()
        # Bound the number of SvANet crops per step. Crops beyond the cap are
        # reported as skipped so callers can still index the refined list.
        trigger_cap = self.max_roi_per_step
        if trigger_cap > 0 and len(trigger_indices) > trigger_cap:
            dropped_indices = trigger_indices[trigger_cap:]
            trigger_indices = trigger_indices[:trigger_cap]
        else:
            dropped_indices = []
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
        source_confidences: List[float] = []
        refine_indices: List[int] = []
        skipped_indices: List[int] = low_area_confidence.nonzero(
            as_tuple=False
        ).flatten().tolist()
        skipped_indices.extend(dropped_indices)
        proposal_diagnostics: List[Dict[str, Any]] = []
        batch_empty_mask_count = 0
        batch_unreliable_mask_count = 0
        for index in trigger_indices:
            diagnostics: Dict[str, Any] = {"image_index": index}
            confidence = 1.0
            if use_gt_roi and gt_masks is not None:
                box = largest_component_bbox(gt_masks[index].bool())
                source = "gt_mask"
                diagnostics["gt_mask"] = {
                    "reason": "accepted" if box is not None else "empty"
                }
            else:
                box, sam3_stats = _largest_component_candidate(
                    sam3_logits[index].float().sigmoid(),
                    threshold=self.mask_threshold,
                    min_confidence=self.sam3_min_confidence,
                    min_component_pixels=self.min_component_pixels,
                    max_component_fraction=self.max_component_fraction,
                )
                diagnostics["sam3_mask"] = sam3_stats
                source = "sam3_mask"
                confidence = float(sam3_stats["confidence"])

            if box is None:
                # Keep the historical empty counter while separately exposing
                # rejected non-empty masks for reliability analysis.
                reason = diagnostics.get(source, {}).get("reason", "empty")
                if reason == "empty":
                    self.runtime_stats["empty_mask_count"] += 1
                    batch_empty_mask_count += 1
                else:
                    self.runtime_stats["unreliable_mask_count"] += 1
                    batch_unreliable_mask_count += 1

                for fallback in self._FALLBACK_CHAINS[
                    self.empty_mask_fallback
                ]:
                    if fallback == "locator":
                        if locator_logits is None:
                            diagnostics["locator_mask"] = {
                                "reason": "unavailable"
                            }
                            continue
                        candidate, locator_stats = _largest_component_candidate(
                            locator_logits[index].float().sigmoid(),
                            threshold=self.locator_threshold,
                            min_confidence=self.locator_min_confidence,
                            min_component_pixels=self.min_component_pixels,
                            max_component_fraction=self.max_component_fraction,
                        )
                        diagnostics["locator_mask"] = locator_stats
                        if candidate is not None:
                            box, source = candidate, "locator_fallback"
                            confidence = float(locator_stats["confidence"])
                            self.runtime_stats["locator_fallback_count"] += 1
                            break
                    elif fallback == "box":
                        prompt = self._prompt_box(
                            prompts[index], height, width, prompt_sizes[index]
                        )
                        if prompt is not None:
                            box, source = tuple(prompt), "box_fallback"
                            confidence = 1.0
                            self.runtime_stats["box_fallback_count"] += 1
                            break
                    elif fallback == "full_image":
                        box, source = (0, 0, width, height), "full_image_fallback"
                        confidence = 0.0
                        self.runtime_stats["full_image_fallback_count"] += 1
                        break
                    elif fallback == "skip":
                        source = "no_reliable_roi_skip"
                        break

            if box is None:
                skipped_indices.append(index)
                diagnostics["decision"] = "skip"
                proposal_diagnostics.append(diagnostics)
                self.runtime_stats["no_reliable_roi_skip_count"] += 1
                continue

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
            source_confidences.append(confidence)
            refine_indices.append(index)
            diagnostics["decision"] = source
            proposal_diagnostics.append(diagnostics)
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
            for local_index, image_index in enumerate(refine_indices):
                x1, y1, x2, y2 = rois[local_index]
                resized = F.interpolate(
                    roi_logits[local_index : local_index + 1, None].float(),
                    (y2 - y1, x2 - x1), mode="bilinear", align_corners=False,
                )[0, 0].to(images.dtype)
                if self.paste_mode == "replace_roi":
                    if self.outside_roi == "zero":
                        final_logits[image_index] = torch.full_like(final_logits[image_index], -20.0)
                    final_logits[image_index, y1:y2, x1:x2] = resized
                elif self.paste_mode == "blend_with_sam3":
                    base = final_logits[image_index, y1:y2, x1:x2]
                    final_logits[image_index, y1:y2, x1:x2] = (
                        (1.0 - self.fusion_weight) * base
                        + self.fusion_weight * resized
                    )
                else:
                    base = final_logits[image_index, y1:y2, x1:x2]
                    final_logits[image_index, y1:y2, x1:x2] = (
                        base + self.residual_scale * resized
                    )
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

        self.runtime_stats["trigger_count"] += len(refine_indices)
        refined_mask = torch.zeros_like(requested_trigger, dtype=torch.bool)
        if refine_indices:
            refined_mask[refine_indices] = True
        widths = [roi[2] - roi[0] for roi in rois]
        heights = [roi[3] - roi[1] for roi in rois]
        return {
            "final_logits": final_logits,
            "refine_loss": refine_loss,
            # ``trigger_mask`` historically drives ROI-list indexing in
            # inference utilities, so it must denote crops that actually ran.
            "trigger_mask": refined_mask,
            "requested_trigger_mask": requested_trigger,
            "confidence_qualified_trigger_mask": trigger,
            "low_area_confidence_mask": low_area_confidence,
            "area_small_probability": area_small_probability,
            "roi_boxes": rois,
            "roi_sources": sources,
            "roi_source_confidences": source_confidences,
            "proposal_diagnostics": proposal_diagnostics,
            "refined_indices": refine_indices,
            "skipped_indices": skipped_indices,
            "roi_images": roi_images,
            "roi_gt": roi_targets,
            "svanet_roi_logits": roi_logits_list,
            "batch_stats": {
                "small_count": int(requested_trigger.sum().item()),
                "trigger_count": len(refine_indices),
                "empty_mask_count": batch_empty_mask_count,
                "unreliable_mask_count": batch_unreliable_mask_count,
                "box_fallback_count": sources.count("box_fallback"),
                "locator_fallback_count": sources.count("locator_fallback"),
                "full_image_fallback_count": sources.count("full_image_fallback"),
                "low_area_confidence_skip_count": int(
                    low_area_confidence.sum().item()
                ),
                "no_reliable_roi_skip_count": sum(
                    diagnostic.get("decision") == "skip"
                    for diagnostic in proposal_diagnostics
                ),
                "mean_roi_width": float(np.mean(widths)) if widths else 0.0,
                "mean_roi_height": float(np.mean(heights)) if heights else 0.0,
            },
        }
