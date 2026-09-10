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
        # Projection-specific gates keep a shared expert pool expressive while
        # preventing one global expert delta from overwhelming every attention
        # projection. They live inside ExpertPool so staged training treats
        # them as expert parameters rather than frozen decoder parameters.
        self.area_residual_scales = nn.ParameterDict()
        self.boundary_residual_scales = nn.ParameterDict()
        self._residual_scale_limits: Dict[str, float] = {}
        self._fixed_residual_scale_keys = set()

    def register_projection_scales(
        self,
        projection_key: str,
        initial_value: float = 1.0,
        learnable: bool = True,
        max_abs_value: float = 1.0,
    ) -> None:
        """Register independent area/boundary residual scales for a projection."""
        if not projection_key or "." in projection_key:
            raise ValueError("projection_key must be a non-empty ParameterDict-safe name")
        if projection_key in self.area_residual_scales:
            return
        if max_abs_value <= 0:
            raise ValueError("max_abs_value must be positive")
        if abs(initial_value) > max_abs_value:
            raise ValueError("initial residual scale exceeds max_abs_value")
        initial = torch.tensor(float(initial_value), dtype=torch.float32)
        self.area_residual_scales[projection_key] = nn.Parameter(
            initial.clone(), requires_grad=bool(learnable)
        )
        self.boundary_residual_scales[projection_key] = nn.Parameter(
            initial.clone(), requires_grad=bool(learnable)
        )
        self._residual_scale_limits[projection_key] = float(max_abs_value)
        if not learnable:
            self._fixed_residual_scale_keys.add(projection_key)

    @property
    def fixed_residual_scale_parameters(self) -> Iterable[nn.Parameter]:
        """Scales declared fixed by config, even when a stage is trainable."""
        for key in self._fixed_residual_scale_keys:
            yield self.area_residual_scales[key]
            yield self.boundary_residual_scales[key]

    def projection_scale(self, projection_key: str, family: str) -> torch.Tensor:
        if family == "area":
            scales = self.area_residual_scales
        elif family == "boundary":
            scales = self.boundary_residual_scales
        else:
            raise ValueError(f"Unknown expert family: {family}")
        if projection_key not in scales:
            raise KeyError(f"No residual scale registered for {projection_key}")
        limit = self._residual_scale_limits.get(projection_key, 1.0)
        return scales[projection_key].clamp(min=-limit, max=limit)

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
        for modality_idx, modality in enumerate(MODALITIES):
            for class_idx, class_name in enumerate(class_names):
                expert = self.experts[f"{modality}_{family}_{class_name}"]
                weight = joint_weights[:, modality_idx, class_idx]
                selected = (weight.detach() != 0).nonzero(as_tuple=False).flatten()
                if selected.numel() == 0:
                    # Do not execute unselected sparse experts. Under DDP keep
                    # a zero dependency so every rank reduces every expert.
                    if torch.distributed.is_initialized():
                        zero = expert.A.sum() + expert.B.sum()
                        ddp_zero = zero if ddp_zero is None else ddp_zero + zero
                    continue
                contribution = expert(x.index_select(0, selected))
                selected_weight = weight.index_select(0, selected).to(x.dtype)
                selected_weight = selected_weight.view(
                    selected.numel(), *([1] * (x.ndim - 1))
                )
                delta = delta.index_add(0, selected, contribution * selected_weight)
        if ddp_zero is not None:
            delta = delta + ddp_zero.to(delta.dtype) * 0.0
        return delta


class RoutedMoELinear(nn.Module):
    """Existing projection path plus routed area and boundary LoRA deltas.

    ``base_linear`` may be SAM3's original ``nn.Linear`` or the project's
    existing ``LoRALinear`` wrapper.  Keeping the wrapper here is important:
    the hierarchical experts are additive and must not replace shared LoRA.
    """

    def __init__(
        self,
        base_linear: nn.Module,
        controller: nn.Module,
        projection_key: Optional[str] = None,
        residual_scale_init: float = 1.0,
        learnable_residual_scale: bool = True,
        residual_scale_max: float = 1.0,
    ) -> None:
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
        # Direct construction remains supported by assigning a deterministic
        # key. Injection always supplies a unique layer/attention/projection key.
        if projection_key is None:
            projection_key = f"projection_{len(controller.expert_pool.area_residual_scales)}"
        self.projection_key = str(projection_key).replace(".", "_")
        controller.expert_pool.register_projection_scales(
            self.projection_key,
            initial_value=residual_scale_init,
            learnable=learnable_residual_scale,
            max_abs_value=residual_scale_max,
        )

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
        area_scale = controller.expert_pool.projection_scale(
            self.projection_key, family="area"
        ).to(dtype=area_delta.dtype)
        boundary_scale = controller.expert_pool.projection_scale(
            self.projection_key, family="boundary"
        ).to(dtype=boundary_delta.dtype)
        output = output + area_scale * area_delta + boundary_scale * boundary_delta

        # ``find_unused_parameters=True`` decides which parameters are reachable
        # as soon as DDP's wrapped forward returns.  Router supervision is added
        # only afterwards by the trainer.  If an entire local batch takes the
        # shared-path fallback, ``forward_family`` deliberately executes no
        # expert and would otherwise disconnect that family's route tensor from
        # the returned graph.  DDP would mark the router parameters ready as
        # unused, then fail when the post-forward router loss reaches them.
        #
        # Keep a scalar, zero-valued dependency on both joint routes regardless
        # of sparse execution.  The locator output is included as well: in the
        # ``use_image_locator=False`` ablation it is intentionally absent from
        # both joint routes, but an optional external locator loss must still be
        # safe under DDP.  These anchors do not change forward values.
        routing_anchor = routes["area_joint"].sum() + routes["boundary_joint"].sum()
        locator_logits = routes.get("image_locator_logits")
        if locator_logits is not None:
            routing_anchor = routing_anchor + locator_logits.sum()
        return output + routing_anchor.to(dtype=output.dtype) * 0.0
