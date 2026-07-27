"""Lightweight runtime configuration helpers shared by train and inference."""

from __future__ import annotations

import copy
from typing import Any, Dict


def effective_inference_config(
    external: Dict[str, Any], checkpoint_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Use checkpoint architecture while retaining current machine data paths."""
    effective = copy.deepcopy(external)
    for section in ("lora", "moe", "router", "svanet", "stages", "loss"):
        if section in checkpoint_config:
            effective[section] = copy.deepcopy(checkpoint_config[section])
    checkpoint_training = checkpoint_config.get("training", {}) or {}
    effective.setdefault("training", {}).update(
        {
            key: copy.deepcopy(checkpoint_training[key])
            for key in ("stage", "end_to_end")
            if key in checkpoint_training
        }
    )
    return effective
