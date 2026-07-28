"""Losses for Hierarchical LoRA-MoE-SAM3 medical segmentation."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def _as_mask_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
    if tensor.ndim != 3:
        raise ValueError(f"Expected mask tensor [N,H,W] or [N,1,H,W], got {tensor.shape}")
    return tensor


def dice_loss_from_probabilities(
    probabilities: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    probabilities = _as_mask_batch(probabilities).float()
    targets = _as_mask_batch(targets).to(
        device=probabilities.device, dtype=torch.float32
    )
    numerator = 2.0 * (probabilities * targets).flatten(1).sum(dim=1) + eps
    denominator = (
        probabilities.flatten(1).sum(dim=1)
        + targets.flatten(1).sum(dim=1)
        + eps
    )
    return (1.0 - numerator / denominator).mean()


def dice_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
    debug_label: Optional[str] = None,
) -> torch.Tensor:
    """Dice(sigmoid(logits), Y) + BCEWithLogits(logits, Y)."""
    logits = _as_mask_batch(logits)
    targets = _as_mask_batch(targets).to(device=logits.device, dtype=torch.float32)
    if logits.shape[-2:] != targets.shape[-2:]:
        logits = F.interpolate(
            logits[:, None].float(),
            size=targets.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[:, 0].to(dtype=logits.dtype)
    dice = dice_loss_from_probabilities(logits.sigmoid(), targets, eps)
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    total = dice + bce
    if debug_label and os.environ.get("SVANET_DEBUG_MODE", "").strip():
        rank = os.environ.get("RANK", "0")
        print(
            f"[LOSS-DEBUG][rank={rank}][{debug_label}] "
            f"logits_finite={bool(torch.isfinite(logits).all())} "
            f"targets_finite={bool(torch.isfinite(targets).all())} "
            f"dice={dice.detach().item():.8f} "
            f"bce={bce.detach().item():.8f} "
            f"total={total.detach().item():.8f}",
            flush=True,
        )
        if not torch.isfinite(total):
            raise FloatingPointError(
                f"Non-finite {debug_label} Dice+BCE loss: "
                f"dice={dice.detach().item()}, bce={bce.detach().item()}"
            )
    return total


def differentiable_boundary_map(values: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    values = _as_mask_batch(values).float()[:, None]
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("Boundary kernel_size must be a positive odd number")
    pad = kernel_size // 2
    maximum = F.max_pool2d(values, kernel_size, stride=1, padding=pad)
    minimum = -F.max_pool2d(-values, kernel_size, stride=1, padding=pad)
    return (maximum - minimum).clamp_min(0.0)[:, 0]


def boundary_dice_loss(
    pred_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    kernel_size: int = 3,
    eps: float = 1e-6,
) -> torch.Tensor:
    pred_logits = _as_mask_batch(pred_logits)
    gt_mask = _as_mask_batch(gt_mask).to(
        device=pred_logits.device, dtype=torch.float32
    )
    if pred_logits.shape[-2:] != gt_mask.shape[-2:]:
        pred_logits = F.interpolate(
            pred_logits[:, None].float(),
            size=gt_mask.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[:, 0].to(dtype=pred_logits.dtype)
    pred_boundary = differentiable_boundary_map(pred_logits.sigmoid(), kernel_size)
    gt_boundary = differentiable_boundary_map(gt_mask, kernel_size)
    return dice_loss_from_probabilities(pred_boundary, gt_boundary, eps)


def extract_matched_masks(
    output: Dict[str, Any],
    target: Dict[str, Any],
    coarse_mask_p3: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Gather final/P3 logits and GT masks using SAM3 Hungarian indices."""
    pred_masks = output["pred_masks"]
    batch_idx, query_idx, target_idx = output["indices"]
    final_logits = pred_masks[(batch_idx, query_idx)]
    target_masks = target.get("masks")
    if target_masks is None:
        empty = final_logits[:0]
        return empty, empty, None if coarse_mask_p3 is None else coarse_mask_p3[:0, 0]
    target_masks = target_masks if target_idx is None else target_masks[target_idx]
    valid = target.get("is_valid_mask")
    if valid is not None:
        valid = valid if target_idx is None else valid[target_idx]
        final_logits = final_logits[valid]
        target_masks = target_masks[valid]
    aux_logits = None
    if coarse_mask_p3 is not None:
        if query_idx.numel() and int(query_idx.max()) >= coarse_mask_p3.shape[1]:
            raise RuntimeError(
                f"Matched query index exceeds P3 queries: {query_idx.max()} vs "
                f"{coarse_mask_p3.shape[1]}"
            )
        aux_logits = coarse_mask_p3[(batch_idx, query_idx)]
        if valid is not None:
            aux_logits = aux_logits[valid]
    return final_logits, target_masks, aux_logits


class HierarchicalMoELoss(nn.Module):
    """Compose Step-6 losses; refinement is added in the SvANet step."""

    def __init__(self, weights: Dict[str, float], boundary_kernel_size: int = 3) -> None:
        super().__init__()
        self.weights = dict(weights)
        self.boundary_kernel_size = int(boundary_kernel_size)

    def forward(
        self,
        final_logits: torch.Tensor,
        gt_masks: torch.Tensor,
        aux_logits: Optional[torch.Tensor],
        routes: Dict[str, torch.Tensor],
        routing_losses: Dict[str, torch.Tensor],
        area_ratio_gt: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero = routes["area_logits"].new_zeros(())
        if final_logits.shape[0] == 0:
            sam3_loss = zero
            boundary_seg_loss = zero
        else:
            sam3_loss = dice_bce_with_logits(final_logits, gt_masks)
            boundary_seg_loss = boundary_dice_loss(
                final_logits, gt_masks, self.boundary_kernel_size
            )
        aux_loss = (
            dice_bce_with_logits(aux_logits, gt_masks)
            if aux_logits is not None and aux_logits.shape[0] > 0
            else zero
        )
        area_ratio_gt = area_ratio_gt.to(
            device=routes["area_ratio_pred"].device, dtype=torch.float32
        )
        area_reg_loss = F.smooth_l1_loss(
            routes["area_ratio_pred"], area_ratio_gt
        )
        components = {
            "sam3_loss": sam3_loss,
            "aux_loss": aux_loss,
            "modality_loss": routing_losses["modality_loss"],
            "area_loss": routing_losses["area_loss"],
            "area_reg_loss": area_reg_loss,
            "boundary_router_loss": routing_losses["boundary_router_loss"],
            "boundary_seg_loss": boundary_seg_loss,
            "load_balance_loss": routing_losses["load_balance_loss"],
            "refine_loss": zero,
        }
        total = zero
        for name, value in components.items():
            total = total + self.weights.get(name, 0.0) * value
        components["total_loss"] = total
        return total, components
