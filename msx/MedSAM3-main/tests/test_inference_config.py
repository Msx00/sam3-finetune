from models.runtime_config import effective_inference_config


def test_checkpoint_architecture_wins_but_external_paths_are_preserved():
    external = {
        "model": {"sam3_checkpoint": "/new-machine/sam3.pt"},
        "dataset": {"mr_root": "/new-machine/mr"},
        "router": {"routing_type": "top1"},
        "svanet": {"paste_mode": "replace_roi"},
        "training": {"stage": 2, "batch_size": 1},
    }
    checkpoint = {
        "model": {"sam3_checkpoint": "/old-machine/sam3.pt"},
        "dataset": {"mr_root": "/old-machine/mr"},
        "router": {"routing_type": "soft"},
        "svanet": {"paste_mode": "soft_gate", "prefer_prompt_roi": True},
        "training": {"stage": 5, "end_to_end": True, "batch_size": 8},
    }
    effective = effective_inference_config(external, checkpoint)
    assert effective["model"]["sam3_checkpoint"] == "/new-machine/sam3.pt"
    assert effective["dataset"]["mr_root"] == "/new-machine/mr"
    assert effective["router"]["routing_type"] == "soft"
    assert effective["svanet"]["paste_mode"] == "soft_gate"
    assert effective["training"]["stage"] == 5
    assert effective["training"]["end_to_end"] is True
    assert effective["training"]["batch_size"] == 1
