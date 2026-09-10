"""Image-driven hierarchical routing for SAM3 LoRA experts.

The public ``area_*`` and ``boundary_*`` tensors retain their legacy
``[batch, 3]`` shapes. The corresponding ``*_all`` tensors expose the true
conditional distributions ``p(child | modality, image)`` with shape
``[batch, 2, 3]``.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class _RouterMLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImageLocator(nn.Module):
    """Predict a prompt-independent coarse foreground map from image memory."""

    def __init__(self, embed_dim: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        hidden_dim = max(8, int(hidden_dim))
        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=1),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, local_image_feature: torch.Tensor) -> torch.Tensor:
        return self.head(local_image_feature)


class ModalityRouter(nn.Module):
    """Route a pooled SAM3 image embedding to MR or US."""

    def __init__(self, embed_dim: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        self.classifier = _RouterMLP(embed_dim, hidden_dim, classes=2)

    def forward(self, image_embedding: torch.Tensor) -> torch.Tensor:
        return self.classifier(image_embedding)


def _masked_image_features(
    local_image_feature: torch.Tensor,
    mask_logits: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return global, soft-ROI and foreground-probability image features."""
    probability = mask_logits.sigmoid()
    if probability.ndim != 4 or probability.shape[1] != 1:
        raise ValueError("mask_logits must be [batch,1,H,W]")
    if probability.shape[0] != local_image_feature.shape[0]:
        raise ValueError("mask and image feature batch sizes differ")
    if probability.shape[-2:] != local_image_feature.shape[-2:]:
        probability = F.interpolate(
            probability,
            size=local_image_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    global_feature = local_image_feature.mean(dim=(2, 3))
    weights = probability.flatten(2)
    local_flat = local_image_feature.flatten(2)
    roi_feature = (local_flat * weights).sum(dim=-1)
    roi_feature = roi_feature / weights.sum(dim=-1).clamp_min(1e-6)
    return global_feature, roi_feature, probability


class AreaRouter(nn.Module):
    """Predict ``p(area | modality, image)`` and continuous foreground ratio."""

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 128,
        conditional_hierarchy: bool = True,
    ) -> None:
        super().__init__()
        self.conditional_hierarchy = bool(conditional_hierarchy)
        # Keep the legacy classifier/ration-head tensor shapes checkpoint
        # compatible. A zero-initialized residual head learns modality-specific
        # deviations without discarding a good independent router warm start.
        self.classifier = _RouterMLP(embed_dim + 3, hidden_dim, 3)
        self.conditional_classifier = (
            _RouterMLP(embed_dim + 3, hidden_dim, 2 * 3)
            if self.conditional_hierarchy
            else None
        )
        self.ratio_head = nn.Sequential(
            nn.LayerNorm(embed_dim + 3),
            nn.Linear(embed_dim + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        if self.conditional_classifier is not None:
            nn.init.zeros_(self.conditional_classifier.net[-1].weight)
            nn.init.zeros_(self.conditional_classifier.net[-1].bias)

    def _features(
        self,
        local_image_feature: torch.Tensor,
        mask_logits: torch.Tensor,
    ) -> torch.Tensor:
        global_feature, roi_feature, probability = _masked_image_features(
            local_image_feature, mask_logits
        )
        mask_stats = torch.stack(
            (
                probability.mean(dim=(1, 2, 3)),
                probability.std(dim=(1, 2, 3), unbiased=False),
                probability.amax(dim=(1, 2, 3)),
            ),
            dim=-1,
        )
        fused_image_feature = 0.5 * (global_feature + roi_feature)
        return torch.cat((fused_image_feature, mask_stats), dim=-1)

    def forward(
        self,
        local_image_feature: torch.Tensor,
        mask_logits: torch.Tensor,
    ) -> torch.Tensor:
        logits, _ = self.forward_with_ratio(local_image_feature, mask_logits)
        return logits

    def forward_with_ratio(
        self,
        local_image_feature: torch.Tensor,
        mask_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self._features(local_image_feature, mask_logits)
        shared_logits = self.classifier(features)
        if self.conditional_hierarchy:
            conditional_delta = self.conditional_classifier(features)
            logits = shared_logits[:, None, :] + conditional_delta.reshape(
                shared_logits.shape[0], 2, 3
            )
        else:
            # Independent-routing ablation: both modality branches see the
            # exact same child logits while retaining the V2 output contract.
            logits = shared_logits[:, None, :].expand(-1, 2, -1)
        return logits, self.ratio_head(features).squeeze(-1).sigmoid()


class BoundaryRouter(nn.Module):
    """Predict ``p(boundary | modality, image)`` from locator-aware features."""

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 128,
        conditional_hierarchy: bool = True,
    ) -> None:
        super().__init__()
        self.conditional_hierarchy = bool(conditional_hierarchy)
        self.classifier = _RouterMLP(2 * embed_dim + 4, hidden_dim, 3)
        self.conditional_classifier = (
            _RouterMLP(2 * embed_dim + 4, hidden_dim, 2 * 3)
            if self.conditional_hierarchy
            else None
        )
        if self.conditional_classifier is not None:
            nn.init.zeros_(self.conditional_classifier.net[-1].weight)
            nn.init.zeros_(self.conditional_classifier.net[-1].bias)

    def forward(
        self,
        local_image_feature: torch.Tensor,
        mask_logits: torch.Tensor,
    ) -> torch.Tensor:
        global_feature, roi_feature, probability_4d = _masked_image_features(
            local_image_feature, mask_logits
        )
        probability = probability_4d[:, 0]
        # ``prepend`` keeps the statistics well-defined for a 1-pixel axis.
        dx = torch.diff(probability, dim=-1, prepend=probability[..., :1])
        dy = torch.diff(probability, dim=-2, prepend=probability[:, :1, :])
        gradient_mean = 0.5 * (
            dx.abs().mean(dim=(1, 2)) + dy.abs().mean(dim=(1, 2))
        )
        gradient_std = 0.5 * (
            dx.std(dim=(1, 2), unbiased=False)
            + dy.std(dim=(1, 2), unbiased=False)
        )
        entropy = -(
            probability.clamp_min(1e-6).log() * probability
            + (1 - probability).clamp_min(1e-6).log() * (1 - probability)
        ).mean(dim=(1, 2))
        foreground_ratio = probability.mean(dim=(1, 2))
        mask_stats = torch.stack(
            (gradient_mean, gradient_std, entropy, foreground_ratio), dim=-1
        )
        features = torch.cat((global_feature, roi_feature, mask_stats), dim=-1)
        shared_logits = self.classifier(features)
        if self.conditional_hierarchy:
            conditional_delta = self.conditional_classifier(features)
            return shared_logits[:, None, :] + conditional_delta.reshape(
                shared_logits.shape[0], 2, 3
            )
        return shared_logits[:, None, :].expand(-1, 2, -1)


class HierarchicalRouter(nn.Module):
    """Image-only parent/child router with optional confidence-aware sparsity."""

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 128,
        routing_mode: str = "top1",
        temperature: float = 1.0,
        conditional_hierarchy: bool = True,
        use_image_locator: bool = True,
        locator_hidden_dim: int | None = None,
        confidence_routing: bool = False,
        confidence_low_threshold: float = 0.45,
        confidence_high_threshold: float = 0.75,
        confidence_top_k: int = 2,
        confidence_fallback: str = "shared",
    ) -> None:
        super().__init__()
        if routing_mode not in {"top1", "soft"}:
            raise ValueError("routing_mode must be 'top1' or 'soft'")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 <= confidence_low_threshold <= confidence_high_threshold <= 1:
            raise ValueError("confidence thresholds must satisfy 0 <= low <= high <= 1")
        if not 1 <= int(confidence_top_k) <= 6:
            raise ValueError("confidence_top_k must be between 1 and 6")
        if confidence_fallback != "shared":
            raise ValueError("Only confidence_fallback='shared' is supported")
        self.embed_dim = int(embed_dim)
        self.routing_mode = routing_mode
        self.temperature = float(temperature)
        self.conditional_hierarchy = bool(conditional_hierarchy)
        self.use_image_locator = bool(use_image_locator)
        self.confidence_routing = bool(confidence_routing)
        self.confidence_low_threshold = float(confidence_low_threshold)
        self.confidence_high_threshold = float(confidence_high_threshold)
        self.confidence_top_k = int(confidence_top_k)
        self.confidence_fallback = confidence_fallback

        locator_hidden_dim = hidden_dim if locator_hidden_dim is None else locator_hidden_dim
        self.image_locator = ImageLocator(embed_dim, int(locator_hidden_dim))
        self.modality_router = ModalityRouter(embed_dim, hidden_dim)
        self.area_router = AreaRouter(
            embed_dim, hidden_dim, conditional_hierarchy=self.conditional_hierarchy
        )
        self.boundary_router = BoundaryRouter(
            embed_dim, hidden_dim, conditional_hierarchy=self.conditional_hierarchy
        )

    def _route(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        soft = F.softmax(logits / self.temperature, dim=-1)
        if self.routing_mode == "soft":
            return soft, soft
        hard = F.one_hot(soft.argmax(dim=-1), soft.shape[-1]).to(soft.dtype)
        # Straight-through top-1 routing keeps the router trainable.
        return hard - soft.detach() + soft, soft

    @staticmethod
    def _hierarchical_top1(
        modality_soft: torch.Tensor,
        child_soft_all: torch.Tensor,
        joint_soft: torch.Tensor,
    ) -> torch.Tensor:
        batch = torch.arange(modality_soft.shape[0], device=modality_soft.device)
        modality_index = modality_soft.argmax(dim=-1)
        child_index = child_soft_all[batch, modality_index].argmax(dim=-1)
        flat_index = modality_index * child_soft_all.shape[-1] + child_index
        hard = F.one_hot(flat_index, joint_soft.shape[1] * joint_soft.shape[2])
        hard = hard.reshape_as(joint_soft).to(joint_soft.dtype)
        return hard - joint_soft.detach() + joint_soft

    def _confidence_policy(
        self,
        modality_soft: torch.Tensor,
        child_soft_all: torch.Tensor,
        joint_soft: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = torch.arange(modality_soft.shape[0], device=modality_soft.device)
        parent_index = modality_soft.argmax(dim=-1)
        parent_confidence = modality_soft.amax(dim=-1)
        child_confidence = child_soft_all[batch, parent_index].amax(dim=-1)
        confidence = torch.minimum(parent_confidence, child_confidence)

        if not self.confidence_routing:
            if self.routing_mode == "soft":
                routed = joint_soft
            else:
                routed = self._hierarchical_top1(
                    modality_soft, child_soft_all, joint_soft
                )
            policy = torch.full_like(parent_index, 2)
            return routed, confidence, policy

        top1 = self._hierarchical_top1(modality_soft, child_soft_all, joint_soft)
        flat_soft = joint_soft.flatten(1)
        top_values, top_indices = flat_soft.topk(self.confidence_top_k, dim=-1)
        top_values = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        topk = torch.zeros_like(flat_soft).scatter(1, top_indices, top_values)
        topk = topk.reshape_as(joint_soft)

        high = confidence >= self.confidence_high_threshold
        low = confidence < self.confidence_low_threshold
        medium = ~(high | low)
        sparse_forward = torch.zeros_like(joint_soft)
        # ``top1`` already contains a straight-through term; use only its hard
        # forward value here because one common ST term is added below.
        sparse_forward = torch.where(
            high[:, None, None], top1.detach(), sparse_forward
        )
        sparse_forward = torch.where(
            medium[:, None, None], topk.detach(), sparse_forward
        )
        # Preserve a useful straight-through router gradient while the forward
        # value is exactly zero for the shared-path fallback.
        routed = sparse_forward - joint_soft.detach() + joint_soft
        policy = torch.where(
            high,
            torch.full_like(parent_index, 2),
            torch.where(medium, torch.ones_like(parent_index), torch.zeros_like(parent_index)),
        )
        return routed, confidence, policy

    def _child_outputs(
        self,
        modality_soft: torch.Tensor,
        child_logits_all: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        child_soft_all = F.softmax(child_logits_all / self.temperature, dim=-1)
        joint_soft = modality_soft[:, :, None] * child_soft_all
        marginal_soft = joint_soft.sum(dim=1)
        # A normalized log-probability is a valid legacy [B,3] CE input.
        marginal_logits = marginal_soft.clamp_min(1e-8).log()
        if self.routing_mode == "soft":
            child_conditional = child_soft_all
        else:
            child_hard = F.one_hot(
                child_soft_all.argmax(dim=-1), child_soft_all.shape[-1]
            ).to(child_soft_all.dtype)
            child_conditional = child_hard - child_soft_all.detach() + child_soft_all
        joint, confidence, policy = self._confidence_policy(
            modality_soft, child_soft_all, joint_soft
        )
        return (
            marginal_logits,
            marginal_soft,
            child_soft_all,
            child_conditional,
            joint_soft,
            joint,
            confidence,
            policy,
        )

    def forward(
        self,
        q3: torch.Tensor,
        local_image_feature: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if q3.ndim != 3:
            raise ValueError(f"q3 must be [queries,batch,channels], got {q3.shape}")
        if local_image_feature.ndim != 4:
            raise ValueError("local_image_feature must be [batch,channels,H,W]")
        if q3.shape[1] != local_image_feature.shape[0]:
            raise ValueError("q3 and local_image_feature batch sizes differ")
        if q3.shape[2] != self.embed_dim or local_image_feature.shape[1] != self.embed_dim:
            raise ValueError("router input channel dimension does not match embed_dim")

        # Keep the original query-derived P3 mask solely as a SAM auxiliary
        # output. Expert routing below uses the image-only locator by default.
        local_tokens = local_image_feature.flatten(2).permute(2, 0, 1)
        coarse_mask = torch.einsum("qbc,hbc->bqh", q3, local_tokens)
        coarse_mask = coarse_mask / math.sqrt(self.embed_dim)
        coarse_mask = coarse_mask.reshape(
            q3.shape[1], q3.shape[0], *local_image_feature.shape[-2:]
        )
        image_locator_logits = self.image_locator(local_image_feature)
        if self.use_image_locator:
            routing_mask_logits = image_locator_logits
        else:
            # Prompt-dependent legacy ablation, collapsed to a single map.
            routing_mask_logits = coarse_mask.amax(dim=1, keepdim=True)

        image_embedding = local_image_feature.mean(dim=(2, 3))
        modality_logits = self.modality_router(image_embedding)
        modality, modality_soft = self._route(modality_logits)
        area_logits_all, area_ratio_pred = self.area_router.forward_with_ratio(
            local_image_feature, routing_mask_logits
        )
        # Preserve the DDP reachability workaround from the original code: the
        # external ratio loss is assembled after the wrapped forward returns.
        area_logits_all = area_logits_all + 0.0 * area_ratio_pred[:, None, None]
        boundary_logits_all = self.boundary_router(
            local_image_feature, routing_mask_logits
        )

        area_outputs = self._child_outputs(modality_soft, area_logits_all)
        boundary_outputs = self._child_outputs(modality_soft, boundary_logits_all)
        (
            area_logits,
            area_soft,
            area_soft_all,
            area_conditional,
            area_joint_soft,
            area_joint,
            area_confidence,
            area_policy,
        ) = area_outputs
        (
            boundary_logits,
            boundary_soft,
            boundary_soft_all,
            boundary_conditional,
            boundary_joint_soft,
            boundary_joint,
            boundary_confidence,
            boundary_policy,
        ) = boundary_outputs
        batch = torch.arange(modality_soft.shape[0], device=modality_soft.device)
        selected_modality = modality_soft.argmax(dim=-1)
        area_selected_logits = area_logits_all[batch, selected_modality]
        boundary_selected_logits = boundary_logits_all[batch, selected_modality]

        return {
            "modality_logits": modality_logits,
            "area_logits": area_logits,
            "area_logits_all": area_logits_all,
            "area_selected_logits": area_selected_logits,
            "area_ratio_pred": area_ratio_pred,
            "boundary_logits": boundary_logits,
            "boundary_logits_all": boundary_logits_all,
            "boundary_selected_logits": boundary_selected_logits,
            "modality": modality,
            "area": area_joint.sum(dim=1),
            "boundary": boundary_joint.sum(dim=1),
            "modality_soft": modality_soft,
            "area_soft": area_soft,
            "area_soft_all": area_soft_all,
            "area_selected_soft": area_soft_all[batch, selected_modality],
            "boundary_soft": boundary_soft,
            "boundary_soft_all": boundary_soft_all,
            "boundary_selected_soft": boundary_soft_all[batch, selected_modality],
            "area_conditional": area_conditional,
            "boundary_conditional": boundary_conditional,
            "area_joint_soft": area_joint_soft,
            "boundary_joint_soft": boundary_joint_soft,
            "area_joint": area_joint,
            "boundary_joint": boundary_joint,
            "area_routing_confidence": area_confidence,
            "boundary_routing_confidence": boundary_confidence,
            # 2=top-1, 1=top-k, 0=shared-path fallback.
            "area_routing_policy": area_policy,
            "boundary_routing_policy": boundary_policy,
            "image_locator_logits": image_locator_logits,
            "routing_mask_logits": routing_mask_logits,
            "coarse_mask_p3": coarse_mask,
        }
