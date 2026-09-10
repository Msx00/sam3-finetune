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
    """Own experts and route later decoder layers from prompt-free image FPN."""

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
        conditional_hierarchy: bool = True,
        use_image_locator: bool = True,
        locator_hidden_dim: Optional[int] = None,
        confidence_routing: bool = False,
        confidence_low_threshold: float = 0.45,
        confidence_high_threshold: float = 0.75,
        confidence_top_k: int = 2,
        confidence_fallback: str = "shared",
        router_regularizer: str = "uniform",
        routing_feature_source: str = "backbone_fpn",
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
            conditional_hierarchy=conditional_hierarchy,
            use_image_locator=use_image_locator,
            locator_hidden_dim=locator_hidden_dim,
            confidence_routing=confidence_routing,
            confidence_low_threshold=confidence_low_threshold,
            confidence_high_threshold=confidence_high_threshold,
            confidence_top_k=confidence_top_k,
            confidence_fallback=confidence_fallback,
        )
        if router_regularizer not in {"uniform", "batch_prior", "none"}:
            raise ValueError(
                "router_regularizer must be 'uniform', 'batch_prior' or 'none'"
            )
        self.router_regularizer = router_regularizer
        if routing_feature_source not in {"backbone_fpn", "decoder_memory"}:
            raise ValueError(
                "routing_feature_source must be 'backbone_fpn' or 'decoder_memory'"
            )
        self.routing_feature_source = routing_feature_source
        self.dac = bool(dac)
        self.current_routes: Optional[Dict[str, torch.Tensor]] = None
        self.routing_targets: Optional[Dict[str, torch.Tensor]] = None
        self.teacher_masks: Optional[Dict[str, torch.Tensor]] = None
        self.last_q3: Optional[torch.Tensor] = None
        self.last_local_feature: Optional[torch.Tensor] = None
        self.backbone_fpn_index: Optional[int] = None
        self._routing_img_ids: Optional[torch.Tensor] = None
        self._raw_backbone_feature: Optional[torch.Tensor] = None
        self._handles: List[Any] = []

    def clear_routes(self, module: nn.Module, args: Tuple[Any, ...]) -> None:
        """Clear per-decoder-call results while retaining captured image state."""
        self.current_routes = None
        self.last_q3 = None
        self.last_local_feature = None

    def begin_model_forward(
        self,
        module: nn.Module,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> None:
        """Reset capture state and record query-to-image IDs for this forward."""
        self.current_routes = None
        self.last_q3 = None
        self.last_local_feature = None
        self._routing_img_ids = None
        self._raw_backbone_feature = None
        if self.routing_feature_source != "backbone_fpn":
            return
        input_batch = kwargs.get("input", kwargs.get("input_batch"))
        if input_batch is None and args:
            input_batch = args[0]
        find_inputs = getattr(input_batch, "find_inputs", None)
        if not find_inputs:
            raise RuntimeError(
                "backbone_fpn routing requires model input.find_inputs[0].img_ids"
            )
        img_ids = getattr(find_inputs[0], "img_ids", None)
        if img_ids is None:
            raise RuntimeError(
                "backbone_fpn routing requires model input.find_inputs[0].img_ids"
            )
        if not torch.is_tensor(img_ids):
            img_ids = torch.as_tensor(img_ids, dtype=torch.long)
        if img_ids.ndim != 1:
            raise RuntimeError("routing img_ids must be a one-dimensional tensor")
        self._routing_img_ids = img_ids.detach().to(dtype=torch.long).clone()

    def capture_backbone_fpn(
        self,
        module: nn.Module,
        args: Tuple[Any, ...],
        output: torch.Tensor,
    ) -> None:
        """Capture one prompt-fusion-free SimpleFPN convolution output."""
        if not torch.is_tensor(output) or output.ndim != 4:
            raise RuntimeError(
                "Selected backbone FPN module must output a [batch,channels,H,W] tensor"
            )
        self._raw_backbone_feature = output

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
        """Apply coherent parent/conditional-child teacher routes per sample."""
        output = dict(routes)
        batch_size = routes["modality"].shape[0]
        device = routes["modality"].device

        def teacher_mask(family: str) -> torch.Tensor:
            mask = None if self.teacher_masks is None else self.teacher_masks.get(family)
            target = None if self.routing_targets is None else self.routing_targets.get(family)
            if not self.training or mask is None or target is None:
                return torch.zeros(batch_size, device=device, dtype=torch.bool)
            mask = mask.to(device=device, dtype=torch.bool)
            target = target.to(device=device, dtype=torch.long)
            if mask.shape != (batch_size,) or target.shape != (batch_size,):
                raise RuntimeError(
                    f"{family} teacher target/mask shape does not match route batch"
                )
            return mask

        modality_mask = teacher_mask("modality")
        output["modality_predicted"] = routes["modality"]
        if modality_mask.any():
            target = self.routing_targets["modality"].to(device=device, dtype=torch.long)
            gt_modality = F.one_hot(target, 2).to(routes["modality"].dtype)
            output["modality"] = torch.where(
                modality_mask[:, None], gt_modality, routes["modality"]
            )
        output["teacher_modality_mask"] = modality_mask
        selected_modality = output["modality"].detach().argmax(dim=-1)
        selected_batch = torch.arange(batch_size, device=device)

        for family in ("area", "boundary"):
            family_mask = teacher_mask(family)
            output[f"{family}_predicted"] = routes[family]
            output[f"{family}_joint_predicted"] = routes[f"{family}_joint"]
            conditional = routes.get(f"{family}_conditional")
            if conditional is None:
                conditional = routes[family][:, None, :].expand(-1, 2, -1)
            if family_mask.any():
                target = self.routing_targets[family].to(device=device, dtype=torch.long)
                gt_child = F.one_hot(target, 3).to(conditional.dtype)
                gt_child = gt_child[:, None, :].expand(-1, 2, -1)
                conditional = torch.where(
                    family_mask[:, None, None], gt_child, conditional
                )
            recomputed_joint = output["modality"][:, :, None] * conditional
            # Preserve confidence fallback/top-k exactly when no teacher signal
            # touched either level of this path.
            touched = modality_mask | family_mask
            output[f"{family}_joint"] = torch.where(
                touched[:, None, None], recomputed_joint, routes[f"{family}_joint"]
            )
            output[family] = output[f"{family}_joint"].sum(dim=1)
            output[f"{family}_conditional"] = conditional
            all_logits = routes.get(f"{family}_logits_all")
            all_soft = routes.get(f"{family}_soft_all")
            if all_logits is not None:
                output[f"{family}_selected_logits"] = all_logits[
                    selected_batch, selected_modality
                ]
            if all_soft is not None:
                output[f"{family}_selected_soft"] = all_soft[
                    selected_batch, selected_modality
                ]
            policy_key = f"{family}_routing_policy"
            if policy_key in routes:
                output[policy_key] = torch.where(
                    touched,
                    torch.full_like(routes[policy_key], 2),
                    routes[policy_key],
                )
            output[f"teacher_{family}_mask"] = family_mask
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
            raise RuntimeError("Router source layer requires decoder image memory and shape")
        if spatial_shapes.numel() < 2:
            raise RuntimeError(f"Invalid spatial_shapes: {spatial_shapes}")
        height = int(spatial_shapes.reshape(-1, 2)[0, 0].item())
        width = int(spatial_shapes.reshape(-1, 2)[0, 1].item())
        spatial_tokens = height * width
        if memory.shape[0] < spatial_tokens:
            raise RuntimeError(
                f"Decoder memory has {memory.shape[0]} tokens, expected {spatial_tokens}"
            )
        if self.routing_feature_source == "decoder_memory":
            local_feature = (
                memory[:spatial_tokens]
                .permute(1, 2, 0)
                .reshape(memory.shape[1], memory.shape[2], height, width)
            )
        else:
            if self._raw_backbone_feature is None:
                raise RuntimeError(
                    "backbone_fpn routing feature was not captured before decoder "
                    "routing; verify the SimpleFPN hook and model forward path"
                )
            if self._routing_img_ids is None:
                raise RuntimeError(
                    "backbone_fpn routing has no img_ids for the current model forward"
                )
            raw_feature = self._raw_backbone_feature
            if raw_feature.shape[-2:] != (height, width):
                raise RuntimeError(
                    "Captured backbone FPN spatial shape "
                    f"{tuple(raw_feature.shape[-2:])} does not match decoder first "
                    f"memory level {(height, width)}"
                )
            if raw_feature.device != q3.device:
                raise RuntimeError(
                    "Captured backbone FPN feature and decoder queries are on "
                    "different devices"
                )
            img_ids = self._routing_img_ids.to(device=raw_feature.device)
            if img_ids.numel() != q3.shape[1]:
                raise RuntimeError(
                    f"routing img_ids batch {img_ids.numel()} does not match decoder "
                    f"query batch {q3.shape[1]}"
                )
            if img_ids.numel() and (
                int(img_ids.min().item()) < 0
                or int(img_ids.max().item()) >= raw_feature.shape[0]
            ):
                raise RuntimeError(
                    "routing img_ids index outside captured backbone FPN batch"
                )
            local_feature = raw_feature.index_select(0, img_ids)
        routes = self.router(q3, local_feature)
        self.last_q3 = q3
        self.last_local_feature = local_feature
        self.current_routes = self.apply_teacher_routing(routes)

    def install_hooks(
        self,
        model: nn.Module,
        decoder: nn.Module,
        source_layer: int = 3,
        backbone_fpn_module: Optional[nn.Module] = None,
        backbone_fpn_index: Optional[int] = None,
    ) -> None:
        """Install model/FPN capture hooks and the decoder routing hook."""
        if self._handles:
            raise RuntimeError("Hierarchical MoE hooks are already installed")
        if not 1 <= int(source_layer) <= len(decoder.layers):
            raise ValueError(
                f"router source layer {source_layer} is outside decoder layers"
            )
        if (
            self.routing_feature_source == "backbone_fpn"
            and (backbone_fpn_module is None or backbone_fpn_index is None)
        ):
            raise RuntimeError(
                "backbone_fpn routing requires a resolved SimpleFPN module"
            )
        self._handles.append(
            model.register_forward_pre_hook(self.begin_model_forward, with_kwargs=True)
        )
        if self.routing_feature_source == "backbone_fpn":
            self.backbone_fpn_index = int(backbone_fpn_index)
            self._handles.append(
                backbone_fpn_module.register_forward_hook(self.capture_backbone_fpn)
            )
        self._handles.append(decoder.register_forward_pre_hook(self.clear_routes))
        self._handles.append(
            decoder.layers[int(source_layer) - 1].register_forward_hook(
                self.route_after_layer3, with_kwargs=True
            )
        )

    def routing_supervision_losses(self) -> Dict[str, torch.Tensor]:
        """Parent/conditional-child CE losses and a configurable regularizer."""
        if self.current_routes is None:
            zero = next(self.parameters()).new_zeros(())
            return {
                "modality_loss": zero,
                "area_loss": zero,
                "boundary_router_loss": zero,
                "load_balance_loss": zero,
            }
        zero = self.current_routes["modality_logits"].new_zeros(())
        losses = {
            "modality_loss": zero,
            "area_loss": zero,
            "boundary_router_loss": zero,
            "load_balance_loss": zero,
        }
        if self.router_regularizer == "uniform":
            regularizers = []
            # Do not force modality balance: real MR/US frequencies may differ.
            for key in ("area_soft", "boundary_soft"):
                mean_probability = self.current_routes[key].mean(dim=0)
                target = torch.full_like(
                    mean_probability, 1.0 / mean_probability.numel()
                )
                regularizers.append((mean_probability - target).square().mean())
            losses["load_balance_loss"] = torch.stack(regularizers).sum()
        elif self.router_regularizer == "batch_prior" and self.routing_targets:
            regularizers = []
            modality_target = self.routing_targets.get("modality")
            for family in ("area", "boundary"):
                family_target = self.routing_targets.get(family)
                if family_target is None:
                    continue
                probabilities = self.current_routes.get(f"{family}_soft_all")
                if probabilities is not None and modality_target is not None:
                    modality = modality_target.to(
                        probabilities.device, dtype=torch.long
                    )
                    batch = torch.arange(probabilities.shape[0], device=probabilities.device)
                    probabilities = probabilities[batch, modality]
                else:
                    probabilities = self.current_routes[f"{family}_soft"]
                target = family_target.to(probabilities.device, dtype=torch.long)
                prior = F.one_hot(target, 3).to(probabilities.dtype).mean(dim=0)
                regularizers.append(
                    (probabilities.mean(dim=0) - prior.detach()).square().mean()
                )
            if regularizers:
                losses["load_balance_loss"] = torch.stack(regularizers).sum()

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
            if family != "modality" and f"{family}_logits_all" in self.current_routes:
                modality_targets = self.routing_targets.get("modality")
                if modality_targets is not None:
                    all_logits = self.current_routes[f"{family}_logits_all"]
                    modality_targets = modality_targets.to(
                        all_logits.device, dtype=torch.long
                    )
                    batch = torch.arange(all_logits.shape[0], device=all_logits.device)
                    logits = all_logits[batch, modality_targets]
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
            if key not in {
                "coarse_mask_p3", "image_locator_logits", "routing_mask_logits",
                "modality_logits", "area_logits", "area_logits_all",
                "boundary_logits", "boundary_logits_all",
            }
        }


_ATTENTION_NAMES = ("self_attn", "ca_text", "cross_attn")
_PROJECTION_NAMES = ("q_proj", "k_proj", "v_proj", "out_proj")


def _unwrap_compiled_module(module: nn.Module) -> nn.Module:
    """Return the original module behind one or more torch.compile wrappers."""
    seen = set()
    while hasattr(module, "_orig_mod") and id(module) not in seen:
        seen.add(id(module))
        module = module._orig_mod
    return module


def _resolve_backbone_fpn_capture(model: nn.Module) -> Tuple[nn.Module, int]:
    """Resolve the raw FPN level corresponding to decoder memory level zero.

    ``SAM3VLBackbone`` first removes ``scalp`` trailing neck outputs, then
    ``Sam3Image`` takes the last ``num_feature_levels`` entries. Therefore the
    first decoder memory level originates at exactly
    ``len(convs) - scalp - num_feature_levels`` in the untrimmed neck.
    """
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise RuntimeError("backbone_fpn routing requires model.backbone")
    backbone = _unwrap_compiled_module(backbone)
    visual = getattr(backbone, "vision_backbone", None)
    if visual is None:
        raise RuntimeError(
            "backbone_fpn routing requires model.backbone.vision_backbone"
        )
    visual = _unwrap_compiled_module(visual)
    convs = getattr(visual, "convs", None)
    if not isinstance(convs, nn.ModuleList) or not convs:
        raise RuntimeError(
            "backbone_fpn routing requires Sam3DualViTDetNeck.convs"
        )
    scalp = int(getattr(backbone, "scalp", 0))
    num_feature_levels = int(getattr(model, "num_feature_levels", 1))
    fpn_index = len(convs) - scalp - num_feature_levels
    if fpn_index < 0 or fpn_index >= len(convs):
        raise RuntimeError(
            "Invalid SAM3 FPN selection: len(convs)="
            f"{len(convs)}, scalp={scalp}, num_feature_levels={num_feature_levels}"
        )
    return convs[fpn_index], fpn_index


def _parse_target_modules(value: Any) -> Dict[str, List[str]]:
    """Normalize bare or ``attention.projection`` target specifications."""
    if value is None:
        return {name: list(_PROJECTION_NAMES) for name in _ATTENTION_NAMES}
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("target_modules must be a non-empty string/list")
    selected = {name: set() for name in _ATTENTION_NAMES}
    for raw_target in value:
        target = str(raw_target).strip()
        parts = target.split(".")
        attention_name = next(
            (part for part in reversed(parts) if part in _ATTENTION_NAMES), None
        )
        projection_name = next(
            (part for part in reversed(parts) if part in _PROJECTION_NAMES), None
        )
        if projection_name is None and attention_name is not None:
            projections = _PROJECTION_NAMES
        elif projection_name is not None:
            projections = (projection_name,)
        else:
            raise ValueError(
                f"Unsupported MoE target '{target}'; expected q_proj or "
                "cross_attn.q_proj style names"
            )
        attentions = _ATTENTION_NAMES if attention_name is None else (attention_name,)
        for name in attentions:
            selected[name].update(projections)
    return {
        name: [projection for projection in _PROJECTION_NAMES if projection in projections]
        for name, projections in selected.items()
        if projections
    }


def _parse_decoder_layers(value: Any, total_layers: int) -> List[int]:
    """Return validated unique 1-based decoder layer numbers."""
    if value is None:
        value = [4, 5, 6]
    if isinstance(value, (int, str)):
        if isinstance(value, str) and "," in value:
            value = [part.strip() for part in value.split(",")]
        else:
            value = [value]
    try:
        layers = sorted({int(layer) for layer in value})
    except (TypeError, ValueError) as error:
        raise ValueError("decoder_layers must contain integer layer numbers") from error
    if not layers:
        raise ValueError("decoder_layers must not be empty")
    invalid = [layer for layer in layers if layer < 1 or layer > total_layers]
    if invalid:
        raise ValueError(
            f"decoder_layers {invalid} outside the decoder's 1..{total_layers} range"
        )
    return layers


def _wrap_attention(
    attention: nn.Module,
    controller: HierarchicalMoEController,
    projection_names: List[str],
    layer_number: int,
    attention_name: str,
    residual_scale_init: float,
    learnable_residual_scale: bool,
    residual_scale_max: float,
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
            RoutedMoELinear(
                projection,
                controller,
                projection_key=(
                    f"layer{layer_number}_{attention_name}_{projection_name}"
                ),
                residual_scale_init=residual_scale_init,
                learnable_residual_scale=learnable_residual_scale,
                residual_scale_max=residual_scale_max,
            ),
        )
        replaced += 1
    return replaced


def inject_hierarchical_moe(
    model: nn.Module,
    config: Dict[str, Any],
    verbose: bool = True,
) -> HierarchicalMoEController:
    """Inject configured routed projections after an image-memory source layer."""
    if hasattr(model, "moe_controller"):
        raise RuntimeError("Model already contains a hierarchical MoE controller")
    decoder = model.transformer.decoder
    layer_value = config.get("decoder_layers", config.get("moe_decoder_layers"))
    decoder_layers = _parse_decoder_layers(layer_value, len(decoder.layers))
    target_value = config.get("target_modules", config.get("moe_target_modules"))
    target_modules = _parse_target_modules(target_value)
    source_layer = int(config.get("router_source_layer", min(decoder_layers) - 1))
    if source_layer < 1 or source_layer >= min(decoder_layers):
        raise ValueError(
            "router_source_layer must be >=1 and precede every routed decoder layer"
        )
    locator_hidden = config.get("locator_hidden_dim")
    routing_feature_source = str(
        config.get("routing_feature_source", "backbone_fpn")
    ).lower()
    backbone_fpn_module = None
    backbone_fpn_index = None
    if routing_feature_source == "backbone_fpn":
        backbone_fpn_module, backbone_fpn_index = _resolve_backbone_fpn_capture(
            model
        )
    controller = HierarchicalMoEController(
        embed_dim=int(config.get("embed_dim", decoder.d_model)),
        rank=int(config.get("rank", 8)),
        alpha=float(config.get("alpha", 16.0)),
        dropout=float(config.get("dropout", 0.0)),
        router_hidden_dim=int(config.get("router_hidden_dim", 128)),
        routing_mode=str(config.get("routing_mode", "top1")).lower(),
        temperature=float(config.get("temperature", 1.0)),
        dac=bool(decoder.dac),
        conditional_hierarchy=bool(config.get("conditional_hierarchy", True)),
        use_image_locator=bool(config.get("use_image_locator", True)),
        locator_hidden_dim=None if locator_hidden is None else int(locator_hidden),
        confidence_routing=bool(config.get("confidence_routing", False)),
        confidence_low_threshold=float(
            config.get("confidence_low_threshold", 0.45)
        ),
        confidence_high_threshold=float(
            config.get("confidence_high_threshold", 0.75)
        ),
        confidence_top_k=int(config.get("confidence_top_k", 2)),
        confidence_fallback=str(config.get("confidence_fallback", "shared")),
        router_regularizer=str(config.get("router_regularizer", "uniform")),
        routing_feature_source=routing_feature_source,
    )
    # Register once so state_dict/optimizer include router and expert parameters.
    model.add_module("moe_controller", controller)

    residual_scale_init = float(config.get("residual_scale_init", 1.0))
    learnable_residual_scale = bool(
        config.get("learnable_residual_scale", True)
    )
    residual_scale_max = float(config.get("residual_scale_max", 1.0))
    replaced = 0
    for layer_number in decoder_layers:
        layer = decoder.layers[layer_number - 1]
        for attention_name, projection_names in target_modules.items():
            attention = getattr(layer, attention_name, None)
            if attention is not None:
                replaced += _wrap_attention(
                    attention,
                    controller,
                    projection_names,
                    layer_number,
                    attention_name,
                    residual_scale_init,
                    learnable_residual_scale,
                    residual_scale_max,
                )
    if replaced == 0:
        raise RuntimeError("No decoder attention projections matched target_modules")
    controller.install_hooks(
        model,
        decoder,
        source_layer=source_layer,
        backbone_fpn_module=backbone_fpn_module,
        backbone_fpn_index=backbone_fpn_index,
    )
    if verbose:
        names = list(controller.expert_pool.expert_names)
        print(
            f"Installed Hierarchical LoRA-MoE on decoder layers "
            f"{decoder_layers} from source layer {source_layer}: "
            f"{replaced} routed projections"
        )
        feature_note = routing_feature_source
        if backbone_fpn_index is not None:
            feature_note += f"[{backbone_fpn_index}]"
        print(f"Routing feature source: {feature_note}")
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
