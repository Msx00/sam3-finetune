"""Hierarchical modality, area and boundary routers."""

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


class ModalityRouter(nn.Module):
    """Route a pooled SAM3 image embedding to MR or US."""

    def __init__(self, embed_dim: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        self.classifier = _RouterMLP(embed_dim, hidden_dim, classes=2)

    def forward(self, image_embedding: torch.Tensor) -> torch.Tensor:
        return self.classifier(image_embedding)


class AreaRouter(nn.Module):
    """Route q3 and coarse P3 to small, medium or large experts."""

    def __init__(self, embed_dim: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        self.classifier = _RouterMLP(embed_dim + 3, hidden_dim, classes=3)
        self.ratio_head = nn.Sequential(
            nn.LayerNorm(embed_dim + 3),
            nn.Linear(embed_dim + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _features(self, q3: torch.Tensor, coarse_mask: torch.Tensor) -> torch.Tensor:
        query_feature = q3.mean(dim=0)
        probability = coarse_mask.sigmoid()
        mask_stats = torch.stack(
            (
                probability.mean(dim=(1, 2, 3)),
                probability.std(dim=(1, 2, 3), unbiased=False),
                probability.amax(dim=(1, 2, 3)),
            ),
            dim=-1,
        )
        return torch.cat((query_feature, mask_stats), dim=-1)

    def forward(self, q3: torch.Tensor, coarse_mask: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._features(q3, coarse_mask))

    def forward_with_ratio(
        self, q3: torch.Tensor, coarse_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self._features(q3, coarse_mask)
        return self.classifier(features), self.ratio_head(features).squeeze(-1).sigmoid()


class BoundaryRouter(nn.Module):
    """Route q3, P3 and local image features to boundary experts."""

    def __init__(self, embed_dim: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        self.classifier = _RouterMLP(2 * embed_dim + 4, hidden_dim, classes=3)

    def forward(
        self,
        q3: torch.Tensor,
        coarse_mask: torch.Tensor,
        local_image_feature: torch.Tensor,
    ) -> torch.Tensor:
        query_feature = q3.mean(dim=0)
        probability = coarse_mask.sigmoid().amax(dim=1)
        dx = probability[:, :, 1:] - probability[:, :, :-1]
        dy = probability[:, 1:, :] - probability[:, :-1, :]
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

        weights = probability.flatten(1)
        local_flat = local_image_feature.flatten(2)
        local_feature = (local_flat * weights[:, None]).sum(dim=-1)
        local_feature = local_feature / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        features = torch.cat((query_feature, local_feature, mask_stats), dim=-1)
        return self.classifier(features)


class HierarchicalRouter(nn.Module):
    """Create P3 and produce the three routing distributions."""

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 128,
        routing_mode: str = "top1",
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if routing_mode not in {"top1", "soft"}:
            raise ValueError("routing_mode must be 'top1' or 'soft'")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.embed_dim = embed_dim
        self.routing_mode = routing_mode
        self.temperature = float(temperature)
        self.modality_router = ModalityRouter(embed_dim, hidden_dim)
        self.area_router = AreaRouter(embed_dim, hidden_dim)
        self.boundary_router = BoundaryRouter(embed_dim, hidden_dim)

    def _route(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        soft = F.softmax(logits / self.temperature, dim=-1)
        if self.routing_mode == "soft":
            return soft, soft
        hard = F.one_hot(soft.argmax(dim=-1), soft.shape[-1]).to(soft.dtype)
        # Straight-through top-1 routing keeps the router trainable.
        routed = hard - soft.detach() + soft
        return routed, soft

    def forward(
        self,
        q3: torch.Tensor,
        local_image_feature: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if q3.ndim != 3:
            raise ValueError(f"q3 must be [queries,batch,channels], got {q3.shape}")
        if local_image_feature.ndim != 4:
            raise ValueError("local_image_feature must be [batch,channels,H,W]")
        local_tokens = local_image_feature.flatten(2).permute(2, 0, 1)
        coarse_mask = torch.einsum("qbc,hbc->bqh", q3, local_tokens)
        coarse_mask = coarse_mask / math.sqrt(self.embed_dim)
        coarse_mask = coarse_mask.reshape(
            q3.shape[1], q3.shape[0], *local_image_feature.shape[-2:]
        )

        image_embedding = local_image_feature.mean(dim=(2, 3))
        modality_logits = self.modality_router(image_embedding)
        area_logits, area_ratio_pred = self.area_router.forward_with_ratio(q3, coarse_mask)
        boundary_logits = self.boundary_router(
            q3, coarse_mask, local_image_feature
        )
        modality, modality_soft = self._route(modality_logits)
        area, area_soft = self._route(area_logits)
        boundary, boundary_soft = self._route(boundary_logits)
        return {
            "modality_logits": modality_logits,
            "area_logits": area_logits,
            "area_ratio_pred": area_ratio_pred,
            "boundary_logits": boundary_logits,
            "modality": modality,
            "area": area,
            "boundary": boundary,
            "modality_soft": modality_soft,
            "area_soft": area_soft,
            "boundary_soft": boundary_soft,
            "area_joint": modality[:, :, None] * area[:, None, :],
            "boundary_joint": modality[:, :, None] * boundary[:, None, :],
            "coarse_mask_p3": coarse_mask,
        }
