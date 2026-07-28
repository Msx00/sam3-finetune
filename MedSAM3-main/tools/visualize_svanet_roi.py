#!/usr/bin/env python3
"""Save the required nine-panel SvANet ROI artifacts for one real sample."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infer_moe_sam3 import build_loader, build_runtime, run_batch, save_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default="outputs/svanet_roi_visualization")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    trainer, config, _ = build_runtime(
        args.config, args.checkpoint, args.stage, args.device
    )
    records = None
    for batch in build_loader(trainer, config, args.split, None):
        result = run_batch(trainer, batch)
        adapter = result["adapter_output"]
        if adapter is not None and bool(adapter["trigger_mask"].any()):
            records = save_batch(result, Path(args.output_dir))
            break
    if records is None:
        raise RuntimeError("No predicted-small sample triggered SvANet in this split")
    print(f"Saved ROI visualization: {records[0]['final_mask']}")


if __name__ == "__main__":
    main()
