"""Optional Hierarchical LoRA-MoE components for MedSAM3.

Imports are lazy so router/LoRA utilities do not require the optional SvANet
SciPy dependency merely because Python initialized the ``models`` package.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "ExpertPool": (".moe_lora", "ExpertPool"),
    "LoRAExpert": (".moe_lora", "LoRAExpert"),
    "inject_hierarchical_moe": (".moe_injector", "inject_hierarchical_moe"),
    "SvANetROIAdapter": (".svanet_roi_adapter", "SvANetROIAdapter"),
    "build_original_svanet": (".svanet_roi_adapter", "build_original_svanet"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
