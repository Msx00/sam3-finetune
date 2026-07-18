#!/usr/bin/env python3
"""Check sparse top-1 expert and Router gradients without loading full SAM3."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.moe_lora import AREA_CLASSES, BOUNDARY_CLASSES, ExpertPool, MODALITIES
from models.router import HierarchicalRouter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed-dim", type=int, default=256)
    args = parser.parse_args()
    dim = args.embed_dim
    router = HierarchicalRouter(dim, 64, routing_mode="top1")
    pool = ExpertPool(dim, dim, rank=8, alpha=16)
    routes = router(torch.randn(8, 1, dim), torch.randn(1, dim, 12, 12))
    modality = int(routes["modality"].argmax(-1).item())
    area = int(routes["area"].argmax(-1).item())
    boundary = int(routes["boundary"].argmax(-1).item())
    selected = {
        f"{MODALITIES[modality]}_area_{AREA_CLASSES[area]}",
        f"{MODALITIES[modality]}_boundary_{BOUNDARY_CLASSES[boundary]}",
    }
    for name in selected:
        torch.nn.init.normal_(pool.experts[name].B, std=0.02)
    values = torch.randn(1, 6, dim)
    output = pool.forward_family(values, routes["area_joint"], "area")
    output = output + pool.forward_family(values, routes["boundary_joint"], "boundary")
    output.square().mean().backward()
    print("selected experts:", sorted(selected))
    for name, expert in pool.experts.items():
        gradients = [parameter.grad for parameter in expert.parameters()]
        norms = [float(gradient.norm()) for gradient in gradients if gradient is not None]
        print({
            "expert": name,
            "selected": name in selected,
            "gradient_norm": sum(norms),
            "gradient_is_none": all(item is None for item in gradients),
            "parameters": sum(parameter.numel() for parameter in expert.parameters()),
        })
    router_norm = sum(
        float(parameter.grad.norm())
        for parameter in router.parameters() if parameter.grad is not None
    )
    print("router_gradient_norm:", router_norm)


if __name__ == "__main__":
    main()
