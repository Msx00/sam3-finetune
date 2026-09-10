"""Tests that train/test build exactly the same MoE structure from YAML."""
from models.runtime_config import resolve_moe_runtime_config


def test_runtime_config_merges_model_and_router_fields_without_mutation() -> None:
    source = {
        "model": {
            "moe_decoder_layers": [4, 5, 6],
            "moe_target_modules": ["cross_attn.q_proj"],
            "expert_lora_rank": 4,
        },
        "moe": {"enabled": True, "rank": 8},
        "router": {
            "routing_type": "soft",
            "conditional_hierarchy": True,
            "confidence_routing": True,
        },
    }
    resolved = resolve_moe_runtime_config(source)
    assert resolved["rank"] == 8  # explicit moe value wins over legacy alias
    assert resolved["decoder_layers"] == [4, 5, 6]
    assert resolved["target_modules"] == ["cross_attn.q_proj"]
    assert resolved["routing_mode"] == "soft"
    assert resolved["conditional_hierarchy"] is True
    assert resolved["confidence_routing"] is True
    assert "decoder_layers" not in source["moe"]
