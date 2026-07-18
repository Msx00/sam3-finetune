#!/usr/bin/env python3
"""Compute TRAIN-only automatic boundary pseudo-label thresholds for MR/US."""

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

from data.boundary_labels import (
    BoundaryScores,
    compute_boundary_scores,
    compute_boundary_thresholds,
    save_boundary_thresholds,
)
from data.sample_index import resolve_sample_path


def _patient_ids(manifest: Dict[str, Any], modality: str) -> List[int]:
    key = f"{modality}_patient_ids"
    if key in manifest:
        return [int(value) for value in manifest[key]]
    return [int(value) for value in manifest.get("patients", {}).get(modality.upper(), [])]


def _root(dataset: Dict[str, Any], modality: str, kind: str) -> Path:
    keys = (
        [f"{modality}_{kind}_root", f"{modality}_root"]
        if kind == "image"
        else [f"{modality}_mask_root"]
    )
    value = next((dataset.get(key) for key in keys if dataset.get(key)), None)
    if not value:
        raise ValueError(f"Missing dataset root; tried {keys}")
    return Path(value)


def collect_train_scores(
    dataset: Dict[str, Any], manifest: Dict[str, Any], band_width: int
) -> Dict[str, List[BoundaryScores]]:
    split = str(dataset.get("train_split", "train"))
    strict = bool(dataset.get("strict_dataset_check", True))
    scores: Dict[str, List[BoundaryScores]] = {"mr": [], "us": []}
    for modality in ("mr", "us"):
        image_root = _root(dataset, modality, "image")
        mask_root = _root(dataset, modality, "mask")
        image_split = image_root if image_root.name.lower() == split else image_root / split
        mask_split = mask_root if mask_root.name.lower() == split else mask_root / split
        for patient_id in _patient_ids(manifest, modality):
            patient_dir = mask_split / str(patient_id)
            if not patient_dir.is_dir():
                message = f"Selected {modality.upper()} mask directory is missing: {patient_dir}"
                if strict:
                    raise FileNotFoundError(message)
                print(f"[Warning] {message}; patient skipped")
                continue
            for mask_file in sorted(patient_dir.glob("slice_*.png")):
                key = f"{patient_id}/{mask_file.name}"
                mask_path = resolve_sample_path(mask_split, key, split, strict=strict)
                image_path = resolve_sample_path(image_split, key, split, strict=strict)
                if mask_path is not None and image_path is not None:
                    scores[modality].append(
                        compute_boundary_scores(image_path, mask_path, band_width)
                    )
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute train-split MR/US automatic boundary pseudo-label thresholds"
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
    mode = dataset.get("boundary_threshold_mode", {})
    band_width = int(mode.get("boundary_band_width", 3))
    scores = collect_train_scores(dataset, manifest, band_width)
    thresholds, counts = compute_boundary_thresholds(
        scores,
        contrast_quantile=float(mode.get("contrast_quantile", 0.33)),
        complexity_quantile=float(mode.get("complexity_quantile", 0.67)),
        boundary_band_width=band_width,
    )
    save_boundary_thresholds(thresholds, args.output)
    print(f"Saved boundary thresholds: {Path(args.output).resolve()}")
    for modality in ("mr", "us"):
        item = thresholds[modality]
        label_counts = counts[modality]
        print(
            f"{modality.upper()}: clear={label_counts['clear']}, "
            f"fuzzy={label_counts['fuzzy']}, complex={label_counts['complex']}, "
            f"fallback={item['fallback_count']}"
        )


if __name__ == "__main__":
    main()

