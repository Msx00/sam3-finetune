#!/usr/bin/env python3
"""Run one real batch and print all hierarchical MoE tensor interfaces."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infer_moe_sam3 import build_loader, build_runtime, run_batch
from models.inference_utils import route_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    trainer, config, _ = build_runtime(
        args.config, args.checkpoint, args.stage, args.device
    )
    batch = next(iter(build_loader(trainer, config, args.split, 1)))
    result = run_batch(trainer, batch)
    controller = trainer.moe_controller
    routes = result["routes"]
    print("image:", tuple(result["input_batch"].img_batch.shape))
    print("local image embedding:", tuple(controller.last_local_feature.shape))
    print("q3:", tuple(controller.last_q3.shape))
    print("coarse mask P3:", tuple(routes["coarse_mask_p3"].shape))
    for family in ("modality", "area", "boundary"):
        print(f"{family} logits:", routes[f"{family}_logits"].detach().cpu())
        print(f"{family} probs:", routes[f"{family}_soft"].detach().cpu())
    print("selection:", route_predictions(routes, 0))
    print("SAM3 final logits:", tuple(result["base_logits"].shape))
    adapter = result["adapter_output"]
    print("SvANet trigger:", None if adapter is None else adapter["trigger_mask"].tolist())
    print("final mask logits:", tuple(result["final_logits"].shape))


if __name__ == "__main__":
    main()
