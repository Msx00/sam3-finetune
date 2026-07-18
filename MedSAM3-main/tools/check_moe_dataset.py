#!/usr/bin/env python3
"""Load a few patient-aware samples and print all MoE labels and paths."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.patient_dataset import PatientDataset
from train_sam3_lora_native import COCOSegmentDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Check MoE medical dataset")
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg = config["dataset"]
    split = cfg.get("train_split", "train")
    total = 0
    for modality in ("mr", "us"):
        root = cfg.get(f"{modality}_root") or cfg.get(f"{modality}_image_root")
        if not root:
            continue
        coco = cfg.get(f"{modality}_{split}_coco_json") or cfg.get(
            f"{modality}_coco_json", ""
        ).format(split=split)
        boxes = cfg.get(f"{modality}_{split}_boxes_json") or cfg.get(
            f"{modality}_boxes_json", ""
        ).format(split=split)
        base = COCOSegmentDataset(root, split=split, annotation_file=coco)
        dataset = PatientDataset(
            base_dataset=base,
            modality_root=root,
            split=split,
            modality=modality.upper(),
            num_patients=cfg.get(f"num_{modality}_patients"),
            patient_ids=(cfg.get("patient_ids") or {}).get(modality),
            sampling_mode=(cfg.get("patient_sampling") or {}).get("mode", "random"),
            seed=int((cfg.get("patient_sampling") or {}).get("seed", 42)),
            mask_root=cfg.get(f"{modality}_mask_root"),
            boxes_json=boxes,
            strict_dataset_check=cfg.get("strict_dataset_check", True),
            area_thresholds=cfg.get("area_threshold_file"),
            boundary_thresholds=cfg.get("boundary_threshold_file"),
            return_format="dict",
            max_slices_per_patient=cfg.get("max_slices_per_patient"),
        )
        print(f"{modality.upper()} patients: {dataset.patient_ids}")
        print(f"{modality.upper()} slices: {len(dataset)}")
        total += len(dataset)
        for index in range(min(args.samples, len(dataset))):
            sample = dataset[index]
            print({
                "image_path": sample["image_path"],
                "mask_path": sample["mask_path"],
                "image_shape": tuple(sample["image"].shape),
                "mask_shape": tuple(sample["mask_gt"].shape),
                "box": sample["box_prompt"].tolist(),
                "patient_id": sample["patient_id"],
                "slice_id": sample["slice_id"],
                "modality_label": sample["modality_label"],
                "area_ratio": sample["area_ratio"],
                "area_label": sample["area_label"],
                "boundary_contrast": sample["boundary_contrast"],
                "boundary_complexity": sample["boundary_complexity"],
                "boundary_label": sample["boundary_label"],
            })
    print(f"Total smoke slices: {total}")


if __name__ == "__main__":
    main()
