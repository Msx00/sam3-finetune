#!/usr/bin/env python3
"""Compute area labels from TRAIN masks for the already selected patients."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.area_labels import (
    compute_area_ratio,
    compute_area_thresholds,
    save_area_thresholds,
)
from data.sample_index import resolve_sample_path


def _patient_ids(manifest: Dict[str, Any], modality: str) -> List[int]:
    flat_key = f"{modality.lower()}_patient_ids"
    if flat_key in manifest:
        return [int(value) for value in manifest[flat_key]]
    return [int(value) for value in manifest.get("patients", {}).get(modality, [])]


def _mask_root(dataset: Dict[str, Any], modality: str) -> Path:
    value = dataset.get(f"{modality.lower()}_mask_root")
    if not value:
        raise ValueError(f"dataset.{modality.lower()}_mask_root is required")
    return Path(value)


def collect_train_ratios(
    dataset: Dict[str, Any], manifest: Dict[str, Any], strict: bool
) -> List[float]:
    split = str(dataset.get("train_split", "train"))
    ratios: List[float] = []
    for modality in ("MR", "US"):
        root = _mask_root(dataset, modality)
        split_root = root if root.name.lower() == split.lower() else root / split
        for patient_id in _patient_ids(manifest, modality):
            patient_dir = split_root / str(patient_id)
            if not patient_dir.is_dir():
                message = f"Selected {modality} patient directory is missing: {patient_dir}"
                if strict:
                    raise FileNotFoundError(message)
                print(f"[Warning] {message}; patient skipped")
                continue
            for mask_file in sorted(patient_dir.glob("slice_*.png")):
                resolved = resolve_sample_path(
                    split_root,
                    f"{patient_id}/{mask_file.name}",
                    split,
                    strict=strict,
                )
                if resolved is not None:
                    ratios.append(compute_area_ratio(resolved))
    return ratios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute train-split area_ratio quantiles for selected patients"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--selected-patients", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with Path(args.selected_patients).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    dataset = config["dataset"]
    ratios = collect_train_ratios(
        dataset, manifest, strict=bool(dataset.get("strict_dataset_check", True))
    )
    mr_ids = _patient_ids(manifest, "MR")
    us_ids = _patient_ids(manifest, "US")
    thresholds, counts = compute_area_thresholds(
        ratios,
        dataset.get("area_threshold_mode", {"type": "quantile"}),
        mr_ids,
        us_ids,
    )
    save_area_thresholds(thresholds, args.output)
    print(f"Saved area thresholds: {Path(args.output).resolve()}")
    print(f"Samples: {thresholds['num_samples']}")
    print(
        f"small={counts['small']}, medium={counts['medium']}, "
        f"large={counts['large']}"
    )


if __name__ == "__main__":
    main()

