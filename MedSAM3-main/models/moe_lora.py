"""LoRA experts and routed linear adapters used by Hierarchical LoRA-MoE.

The expert pool contains exactly twelve shared low-rank adapters.  The frozen
SAM3 projection remains the base path; a routed area delta and a routed
boundary delta are added to it in decoder layers 4--6.
"""

from __future__ import annotations

import math
import weakref
from typing import Dict, Iterable, Optional

import torch
from torch import nn
from torch.nn import functional as F


MODALITIES = ("MR", "US")
AREA_CLASSES = ("small", "medium", "large")
BOUNDARY_CLASSES = ("clear", "fuzzy", "complex")


class LoRAExpert(nn.Module):
    """A low-rank delta ``scaling * B(A(x))`` rather than a full network."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.A = nn.Parameter(torch.empty(self.rank, self.in_features))
        self.B = nn.Parameter(torch.empty(self.out_features, self.rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.linear(self.dropout(x), self.A)
        return F.linear(hidden, self.B) * self.scaling


class ExpertPool(nn.Module):
    """The shared 12-expert MR/US area and boundary pool."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        names = []
        for modality in MODALITIES:
            names.extend(f"{modality}_area_{name}" for name in AREA_CLASSES)
            names.extend(
                f"{modality}_boundary_{name}" for name in BOUNDARY_CLASSES
            )
        self.experts = nn.ModuleDict(
            {
                name: LoRAExpert(
                    in_features,
                    out_features,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
                for name in names
            }
        )

    @property
    def expert_names(self) -> Iterable[str]:
        return self.experts.keys()

    def forward_family(
        self,
        x: torch.Tensor,
        joint_weights: torch.Tensor,
        family: str,
    ) -> torch.Tensor:
        """Mix one expert family using weights shaped ``[batch, 2, 3]``."""
        if family == "area":
            class_names = AREA_CLASSES
        elif family == "boundary":
            class_names = BOUNDARY_CLASSES
        else:
            raise ValueError(f"Unknown expert family: {family}")
        if joint_weights.ndim != 3 or tuple(joint_weights.shape[1:]) != (2, 3):
            raise ValueError(
                "joint_weights must have shape [batch, 2, 3], got "
                f"{tuple(joint_weights.shape)}"
            )
        if x.shape[0] != joint_weights.shape[0]:
            raise ValueError(
                f"Adapter input batch {x.shape[0]} != route batch "
                f"{joint_weights.shape[0]}"
            )

        delta = torch.zeros(
            (*x.shape[:-1], self.out_features), device=x.device, dtype=x.dtype
        )
        ddp_zero = None
        hard_routing = bool(
            torch.all((joint_weights.detach() == 0) | (joint_weights.detach() == 1))
        )
        for modality_idx, modality in enumerate(MODALITIES):
            for class_idx, class_name in enumerate(class_names):
                expert = self.experts[f"{modality}_{family}_{class_name}"]
                weight = joint_weights[:, modality_idx, class_idx]
                if hard_routing:
                    selected = (weight.detach() != 0).nonzero(as_tuple=False).flatten()
                    if selected.numel() == 0:
                        # Do not execute an unselected top-1 expert. Its
                        # parameters normally receive no gradient from this sample.
                        # Under DDP, keep a zero-valued autograd dependency so
                        # every rank participates in reduction for every expert
                        # without paying for an expert forward pass.
                        if torch.distributed.is_initialized():
                            zero = expert.A.sum() + expert.B.sum()
                            ddp_zero = zero if ddp_zero is None else ddp_zero + zero
                        continue
                    contribution = expert(x.index_select(0, selected))
                    selected_weight = weight.index_select(0, selected).to(x.dtype)
                    selected_weight = selected_weight.view(
                        selected.numel(), *([1] * (x.ndim - 1))
                    )
                    delta = delta.index_add(
                        0, selected, contribution * selected_weight
                    )
                    continue
                weight = weight.to(dtype=x.dtype)
                weight = weight.view(x.shape[0], *([1] * (x.ndim - 1)))
                delta = delta + expert(x) * weight
        if ddp_zero is not None:
            delta = delta + ddp_zero.to(delta.dtype) * 0.0
        return delta


class RoutedMoELinear(nn.Module):
    """Existing projection path plus routed area and boundary LoRA deltas.

    ``base_linear`` may be SAM3's original ``nn.Linear`` or the project's
    existing ``LoRALinear`` wrapper.  Keeping the wrapper here is important:
    the hierarchical experts are additive and must not replace shared LoRA.
    """

    def __init__(self, base_linear: nn.Module, controller: nn.Module) -> None:
        super().__init__()
        required = ("in_features", "out_features", "weight", "bias")
        if any(not hasattr(base_linear, name) for name in required):
            raise TypeError(
                f"Projection {type(base_linear).__name__} does not expose {required}"
            )
        if base_linear.in_features != base_linear.out_features:
            raise ValueError("Routed attention projections must be square")
        self.base_linear = base_linear
        for parameter in self.base_linear.parameters():
            parameter.requires_grad = False
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        # Avoid registering the same controller under every projection wrapper.
        object.__setattr__(self, "_controller_ref", weakref.ref(controller))

    @property
    def weight(self) -> torch.Tensor:
        return self.base_linear.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base_linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base_linear(x)
        controller = self._controller_ref()
        if controller is None:
            raise RuntimeError("Hierarchical MoE controller is no longer available")
        routes: Optional[Dict[str, torch.Tensor]] = controller.current_routes
        if routes is None:
            # Layers 1--3 never use this wrapper. This fallback also makes model
            # inspection/profiling safe before the first full decoder pass.
            return output
        area_delta = controller.expert_pool.forward_family(
            x, routes["area_joint"], family="area"
        )
        boundary_delta = controller.expert_pool.forward_family(
            x, routes["boundary_joint"], family="boundary"
        )
        return output + area_delta + boundary_delta
