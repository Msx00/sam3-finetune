"""Small inference helpers shared by CLI and debug tools."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from .moe_lora import AREA_CLASSES, BOUNDARY_CLASSES, MODALITIES


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        for field in value.__dataclass_fields__:
            setattr(value, field, move_to_device(getattr(value, field), device))
    return value


def select_best_mask_logits(output: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    masks = output["pred_masks"]
    if masks.ndim != 4:
        raise ValueError(f"pred_masks must be [B,Q,H,W], got {masks.shape}")
    scores = output.get("pred_logits")
    if scores is None:
        indices = torch.zeros(masks.shape[0], device=masks.device, dtype=torch.long)
    else:
        scores = scores.detach()
        while scores.ndim > 2:
            scores = scores.amax(dim=-1)
        if scores.shape[:2] != masks.shape[:2]:
            raise ValueError(
                f"pred_logits {scores.shape} does not align with masks {masks.shape}"
            )
        indices = scores.argmax(dim=1)
    batch = torch.arange(masks.shape[0], device=masks.device)
    return masks[batch, indices], indices


def normalized_xyxy_prompts(find_input: Any) -> Sequence[Optional[torch.Tensor]]:
    boxes_tensor = find_input.input_boxes
    boxes_mask = find_input.input_boxes_mask
    if boxes_tensor.ndim != 3 or boxes_tensor.shape[-1] != 4:
        raise ValueError(
            f"input_boxes must be a 3D tensor ending in 4, got {boxes_tensor.shape}"
        )
    if boxes_mask.ndim != 2:
        raise ValueError(f"input_boxes_mask must be 2D, got {boxes_mask.shape}")

    num_queries, num_boxes = boxes_mask.shape
    batch_first = boxes_tensor.shape[:2] == (num_queries, num_boxes)
    boxes_first = boxes_tensor.shape[:2] == (num_boxes, num_queries)
    if not batch_first and not boxes_first:
        raise ValueError(
            "input_boxes and input_boxes_mask have incompatible shapes: "
            f"{boxes_tensor.shape} versus {boxes_mask.shape}"
        )

    prompts = []
    for query_index in range(num_queries):
        # SAM3 variants use either [B_queries, N_boxes, 4] or
        # [N_boxes, B_queries, 4]. The mask is consistently query-first.
        boxes = (
            boxes_tensor[query_index]
            if batch_first
            else boxes_tensor[:, query_index]
        )
        boxes = boxes[~boxes_mask[query_index]]
        if boxes.numel():
            center, size = boxes[..., :2], boxes[..., 2:]
            boxes = torch.cat((center - size / 2, center + size / 2), dim=-1)
        prompts.append(boxes)
    return prompts


def route_predictions(routes: Dict[str, torch.Tensor], index: int) -> Dict[str, Any]:
    modality = int(routes["modality_logits"][index].argmax().item())
    area = int(routes["area_logits"][index].argmax().item())
    boundary = int(routes["boundary_logits"][index].argmax().item())
    prefix = MODALITIES[modality]
    return {
        "modality": prefix,
        "modality_probs": routes["modality_soft"][index].detach().cpu().tolist(),
        "area": AREA_CLASSES[area],
        "area_probs": routes["area_soft"][index].detach().cpu().tolist(),
        "boundary": BOUNDARY_CLASSES[boundary],
        "boundary_probs": routes["boundary_soft"][index].detach().cpu().tolist(),
        "area_expert": f"{prefix}_area_{AREA_CLASSES[area]}",
        "boundary_expert": f"{prefix}_boundary_{BOUNDARY_CLASSES[boundary]}",
    }
