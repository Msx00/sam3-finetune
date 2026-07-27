"""Optional injection of hierarchical routed LoRA experts into SAM3 decoder."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .moe_lora import ExpertPool, RoutedMoELinear
from .router import HierarchicalRouter


class HierarchicalMoEController(nn.Module):
    """Own the routers/expert pool and bridge decoder layer 3 to layers 4--6."""

    def __init__(
        self,
        embed_dim: int,
        rank: int,
        alpha: float,
        dropout: float,
        router_hidden_dim: int,
        routing_mode: str,
        temperature: float,
        dac: bool,
    ) -> None:
        super().__init__()
        self.expert_pool = ExpertPool(
            embed_dim,
            embed_dim,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        self.router = HierarchicalRouter(
            embed_dim=embed_dim,
            hidden_dim=router_hidden_dim,
            routing_mode=routing_mode,
            temperature=temperature,
        )
        self.dac = bool(dac)
        self.current_routes: Optional[Dict[str, torch.Tensor]] = None
        self.routing_targets: Optional[Dict[str, torch.Tensor]] = None
        self.teacher_masks: Optional[Dict[str, torch.Tensor]] = None
        self.last_q3: Optional[torch.Tensor] = None
        self.last_local_feature: Optional[torch.Tensor] = None
        self._handles: List[Any] = []

    def clear_routes(self, module: nn.Module, args: Tuple[Any, ...]) -> None:
        self.current_routes = None
        self.last_q3 = None
        self.last_local_feature = None

    def set_routing_targets(
        self,
        targets: Optional[Dict[str, torch.Tensor]],
        teacher_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        self.routing_targets = targets
        self.teacher_masks = teacher_masks

    def set_modality_targets(self, targets: Optional[torch.Tensor]) -> None:
        """Backward-compatible target setter used by older callers."""
        if targets is None:
            self.set_routing_targets(None)
        else:
            self.set_routing_targets({"modality": targets})

    def apply_teacher_routing(
        self, routes: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Replace selected per-sample routes with GT one-hot routes during training."""
        output = dict(routes)
        for family, classes in (("modality", 2), ("area", 3), ("boundary", 3)):
            predicted = routes[family]
            output[f"{family}_predicted"] = predicted
            mask = None if self.teacher_masks is None else self.teacher_masks.get(family)
            target = None if self.routing_targets is None else self.routing_targets.get(family)
            if self.training and mask is not None and target is not None:
                mask = mask.to(device=predicted.device, dtype=torch.bool)
                target = target.to(device=predicted.device, dtype=torch.long)
                if mask.shape != predicted.shape[:1] or target.shape != predicted.shape[:1]:
                    raise RuntimeError(
                        f"{family} teacher target/mask shape does not match route batch"
                    )
                gt_route = F.one_hot(target, classes).to(predicted.dtype)
                output[family] = torch.where(mask[:, None], gt_route, predicted)
                output[f"teacher_{family}_mask"] = mask
            else:
                output[f"teacher_{family}_mask"] = torch.zeros(
                    predicted.shape[0], device=predicted.device, dtype=torch.bool
                )
        output["area_joint"] = (
            output["modality"][:, :, None] * output["area"][:, None, :]
        )
        output["boundary_joint"] = (
            output["modality"][:, :, None] * output["boundary"][:, None, :]
        )
        return output

    def route_after_layer3(
        self,
        module: nn.Module,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        output: Tuple[torch.Tensor, Optional[torch.Tensor]],
    ) -> None:
        q3 = output[0]
        if self.dac and module.training:
            if q3.shape[0] % 2 != 0:
                raise RuntimeError("DAC decoder output must contain two query groups")
            q3 = q3[: q3.shape[0] // 2]
        # SAM3 activation checkpointing converts layer kwargs to positional
        # arguments. Bind both representations back to the layer signature.
        bound = inspect.signature(module.forward).bind_partial(*args, **kwargs)
        memory = bound.arguments.get("memory")
        spatial_shapes = bound.arguments.get("memory_spatial_shapes")
        if memory is None or spatial_shapes is None:
            raise RuntimeError("Layer-3 routing requires decoder image memory and shape")
        if spatial_shapes.numel() < 2:
            raise RuntimeError(f"Invalid spatial_shapes: {spatial_shapes}")
        height = int(spatial_shapes.reshape(-1, 2)[0, 0].item())
        width = int(spatial_shapes.reshape(-1, 2)[0, 1].item())
        spatial_tokens = height * width
        if memory.shape[0] < spatial_tokens:
            raise RuntimeError(
                f"Decoder memory has {memory.shape[0]} tokens, expected {spatial_tokens}"
            )
        local_feature = (
            memory[:spatial_tokens]
            .permute(1, 2, 0)
            .reshape(memory.shape[1], memory.shape[2], height, width)
        )
        routes = self.router(q3, local_feature)
        self.last_q3 = q3
        self.last_local_feature = local_feature
        self.current_routes = self.apply_teacher_routing(routes)

    def install_hooks(self, decoder: nn.Module) -> None:
        if self._handles:
            raise RuntimeError("Hierarchical MoE hooks are already installed")
        self._handles.append(decoder.register_forward_pre_hook(self.clear_routes))
        self._handles.append(
            decoder.layers[2].register_forward_hook(
                self.route_after_layer3, with_kwargs=True
            )
        )

    def routing_supervision_losses(self) -> Dict[str, torch.Tensor]:
        """Router CE losses plus area/boundary-only load balancing."""
        if self.current_routes is None:
            zero = next(self.parameters()).new_zeros(())
            return {
                "modality_loss": zero,
                "area_loss": zero,
                "boundary_router_loss": zero,
                "load_balance_loss": zero,
            }
        load_balance_losses = []
        # Do not force modality balance: real MR/US frequencies may be unequal.
        for key in ("area_soft", "boundary_soft"):
            mean_probability = self.current_routes[key].mean(dim=0)
            target = torch.full_like(mean_probability, 1.0 / mean_probability.numel())
            load_balance_losses.append((mean_probability - target).square().mean())
        zero = self.current_routes["modality_logits"].new_zeros(())
        losses = {
            "modality_loss": zero,
            "area_loss": zero,
            "boundary_router_loss": zero,
            "load_balance_loss": torch.stack(load_balance_losses).sum(),
        }
        if self.routing_targets is None:
            return losses
        mapping = {
            "modality": "modality_loss",
            "area": "area_loss",
            "boundary": "boundary_router_loss",
        }
        for family, loss_name in mapping.items():
            if family not in self.routing_targets:
                continue
            logits = self.current_routes[f"{family}_logits"]
            targets = self.routing_targets[family].to(logits.device, dtype=torch.long)
            if targets.shape != logits.shape[:1]:
                raise RuntimeError(
                    f"{family} target shape {targets.shape} != logits batch {logits.shape[:1]}"
                )
            losses[loss_name] = F.cross_entropy(logits, targets)
        return losses

    def routing_aux_loss(self) -> torch.Tensor:
        """Compatibility sum; new training uses individually weighted losses."""
        return torch.stack(list(self.routing_supervision_losses().values())).sum()

    def teacher_routing_statistics(self) -> Dict[str, Tuple[int, int]]:
        if self.current_routes is None:
            return {}
        statistics = {}
        for family in ("modality", "area", "boundary"):
            mask = self.current_routes.get(f"teacher_{family}_mask")
            if mask is not None:
                statistics[family] = (int(mask.sum().item()), int(mask.numel()))
        return statistics

    def routing_summary(self) -> Dict[str, torch.Tensor]:
        if self.current_routes is None:
            return {}
        return {
            key: value.detach()
            for key, value in self.current_routes.items()
            if key not in {"coarse_mask_p3", "modality_logits", "area_logits", "boundary_logits"}
        }


def _wrap_attention(
    attention: nn.Module,
    controller: HierarchicalMoEController,
    projection_names: List[str],
) -> int:
    replaced = 0
    for projection_name in projection_names:
        if not hasattr(attention, projection_name):
            raise TypeError(
                f"Attention {type(attention).__name__} does not expose {projection_name}; "
                "apply the existing LoRA injection before hierarchical MoE injection"
            )
        projection = getattr(attention, projection_name)
        if isinstance(projection, RoutedMoELinear):
            continue
        setattr(
            attention,
            projection_name,
            RoutedMoELinear(projection, controller),
        )
        replaced += 1
    return replaced


def inject_hierarchical_moe(
    model: nn.Module,
    config: Dict[str, Any],
    verbose: bool = True,
) -> HierarchicalMoEController:
    """Inject routed experts into configured decoder attention projections."""
    if hasattr(model, "moe_controller"):
        raise RuntimeError("Model already contains a hierarchical MoE controller")
    decoder = model.transformer.decoder
    layer_numbers = [int(value) for value in config.get("decoder_layers", [4, 5, 6])]
    if not layer_numbers or len(layer_numbers) != len(set(layer_numbers)):
        raise ValueError("moe.decoder_layers must contain unique one-based layer numbers")
    if min(layer_numbers) < 4 or max(layer_numbers) > len(decoder.layers):
        raise ValueError(
            "moe.decoder_layers must be within [4, number_of_decoder_layers] "
            f"because routing is produced after layer 3, got {layer_numbers}"
        )
    target_modules = list(
        config.get(
            "target_modules",
            [
                f"{attention}.{projection}"
                for attention in ("self_attn", "ca_text", "cross_attn")
                for projection in ("q_proj", "k_proj", "v_proj", "out_proj")
            ],
        )
    )
    valid_attentions = {"self_attn", "ca_text", "cross_attn"}
    valid_projections = {"q_proj", "k_proj", "v_proj", "out_proj"}
    selected: Dict[str, List[str]] = {}
    for value in target_modules:
        parts = str(value).split(".")
        if len(parts) != 2 or parts[0] not in valid_attentions or parts[1] not in valid_projections:
            raise ValueError(
                "Each moe.target_modules entry must be "
                "{self_attn|ca_text|cross_attn}.{q_proj|k_proj|v_proj|out_proj}, "
                f"got {value!r}"
            )
        selected.setdefault(parts[0], []).append(parts[1])
    if not selected:
        raise ValueError("moe.target_modules must not be empty")
    controller = HierarchicalMoEController(
        embed_dim=int(config.get("embed_dim", decoder.d_model)),
        rank=int(config.get("rank", 8)),
        alpha=float(config.get("alpha", 16.0)),
        dropout=float(config.get("dropout", 0.0)),
        router_hidden_dim=int(config.get("router_hidden_dim", 128)),
        routing_mode=str(config.get("routing_mode", "top1")).lower(),
        temperature=float(config.get("temperature", 1.0)),
        dac=bool(decoder.dac),
    )
    # Register once so state_dict/optimizer include router and expert parameters.
    model.add_module("moe_controller", controller)

    replaced = 0
    for layer_number in layer_numbers:
        layer = decoder.layers[layer_number - 1]
        for attention_name, projection_names in selected.items():
            attention = getattr(layer, attention_name, None)
            if attention is None:
                raise TypeError(
                    f"Decoder layer {layer_number} lacks configured attention "
                    f"module {attention_name!r}"
                )
            replaced += _wrap_attention(
                attention, controller, projection_names
            )
    controller.install_hooks(decoder)
    if verbose:
        names = list(controller.expert_pool.expert_names)
        print(
            f"Installed Hierarchical LoRA-MoE on decoder layers {layer_numbers}: "
            f"{replaced} routed attention projections"
        )
        print(f"Shared expert pool ({len(names)} experts): {', '.join(names)}")
    return controller


def save_moe_weights(controller: HierarchicalMoEController, save_path: str) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(controller.state_dict(), path)
    print(f"Saved Hierarchical LoRA-MoE weights to {path}")


def load_moe_weights(
    controller: HierarchicalMoEController, load_path: str, strict: bool = True
) -> None:
    state = torch.load(load_path, map_location="cpu", weights_only=True)
    controller.load_state_dict(state, strict=strict)
