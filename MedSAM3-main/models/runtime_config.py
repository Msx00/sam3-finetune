"""Resolve one canonical runtime configuration for Hierarchical MoE.

Historically, decoder targets lived under ``model`` while the injector only
received ``moe``.  Keeping the merge here prevents training and evaluation from
silently constructing different architectures from the same YAML file.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping


_MODEL_ALIASES = {
    "moe_decoder_layers": "decoder_layers",
    "moe_target_modules": "target_modules",
    "expert_lora_rank": "rank",
    "expert_lora_alpha": "alpha",
    "expert_lora_dropout": "dropout",
}

_ROUTER_RUNTIME_KEYS = {
    "conditional_hierarchy",
    "use_image_locator",
    "locator_hidden_dim",
    "router_source_layer",
    "routing_feature_source",
    "confidence_routing",
    "confidence_low_threshold",
    "confidence_high_threshold",
    "confidence_top_k",
    "confidence_fallback",
    "router_regularizer",
    "residual_scale_init",
    "residual_scale_max",
    "learnable_residual_scale",
}


def resolve_moe_runtime_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge model/router MoE fields without mutating the loaded YAML."""
    resolved = dict(config.get("moe") or {})
    model = dict(config.get("model") or {})
    router = dict(config.get("router") or {})

    for source, destination in _MODEL_ALIASES.items():
        if source in model and destination not in resolved:
            resolved[destination] = model[source]
    for key in _ROUTER_RUNTIME_KEYS:
        if key in router and key not in resolved:
            resolved[key] = router[key]
    if "routing_type" in router:
        resolved["routing_mode"] = str(router["routing_type"]).lower()
    return resolved
