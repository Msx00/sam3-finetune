#!/usr/bin/env python3
"""Draw the XYXY box prompts configured for Stage 5 on their source images."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable

import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.sample_index import load_boxes_index


COLORS = (
    "#ff2d2d",
    "#00d26a",
    "#1e90ff",
    "#ffd400",
    "#d14cff",
    "#00d9e8",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw Stage-5 XYXY box prompts on original images."
    )
    parser.add_argument("--config", required=True, help="Stage-5 YAML path")
    parser.add_argument(
        "--modality",
        choices=("mr", "us", "both"),
        default="both",
        help="Dataset modality to inspect",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="train",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of randomly selected images per modality; 0 means all",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--all-patients",
        action="store_true",
        help="Ignore selected-patient manifest and sample from the whole split",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/box_prompt_check",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("dataset"), dict):
        raise ValueError(f"Missing dataset section in config: {path}")
    return config


def configured_paths(dataset_cfg: dict, modality: str, split: str) -> tuple[Path, Path]:
    image_root_value = (
        dataset_cfg.get(f"{modality}_image_root")
        or dataset_cfg.get(f"{modality}_root")
    )
    boxes_value = (
        dataset_cfg.get(f"{modality}_{split}_boxes_json")
        or dataset_cfg.get(f"{modality}_boxes_json")
    )
    if isinstance(boxes_value, str):
        boxes_value = boxes_value.format(split=split)
    if not image_root_value:
        raise ValueError(f"No image root configured for modality={modality}")
    if not boxes_value:
        raise ValueError(
            f"No boxes JSON configured for modality={modality}, split={split}"
        )
    return Path(image_root_value), Path(boxes_value)


def resolve_image(image_root: Path, split: str, sample_key: str) -> Path | None:
    candidates = [
        image_root / split / sample_key,
        image_root / sample_key,
    ]
    if image_root.name.lower() == split.lower():
        candidates.insert(0, image_root / sample_key)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    color: str,
) -> None:
    font = ImageFont.load_default()
    left, top, right, bottom = draw.textbbox(xy, text, font=font)
    padding = 3
    draw.rectangle(
        (left - padding, top - padding, right + padding, bottom + padding),
        fill=color,
    )
    draw.text(xy, text, fill="white", font=font)


def visualize_one(
    image_path: Path,
    boxes: Iterable[Iterable[float]],
    output_path: Path,
) -> list[str]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)
    line_width = max(2, round(min(width, height) / 150))
    warnings: list[str] = []

    for box_index, values in enumerate(boxes):
        x1, y1, x2, y2 = (float(value) for value in values)
        color = COLORS[box_index % len(COLORS)]
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            warnings.append(
                f"box[{box_index}] is out of bounds: "
                f"{[x1, y1, x2, y2]}, image_size={[width, height]}"
            )
        if x2 <= x1 or y2 <= y1:
            warnings.append(
                f"box[{box_index}] has non-positive size: {[x1, y1, x2, y2]}"
            )

        # Clip only for drawing. The warning above preserves evidence of bad input.
        draw_x1 = min(max(x1, 0.0), width - 1)
        draw_y1 = min(max(y1, 0.0), height - 1)
        draw_x2 = min(max(x2, 0.0), width - 1)
        draw_y2 = min(max(y2, 0.0), height - 1)
        draw.rectangle(
            (draw_x1, draw_y1, draw_x2, draw_y2),
            outline=color,
            width=line_width,
        )
        label_y = max(3, int(draw_y1) + 3)
        draw_label(
            draw,
            (max(3, int(draw_x1) + 3), label_y),
            f"#{box_index} [{x1:g}, {y1:g}, {x2:g}, {y2:g}]",
            color,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return warnings


def process_modality(
    dataset_cfg: dict,
    modality: str,
    split: str,
    num_samples: int,
    seed: int,
    output_root: Path,
    selected_patient_ids: set[int] | None,
) -> dict:
    image_root, boxes_path = configured_paths(dataset_cfg, modality, split)
    boxes_index = load_boxes_index(boxes_path, strict=True)
    keys = sorted(boxes_index)
    if selected_patient_ids is not None:
        keys = [
            key for key in keys if int(key.split("/", maxsplit=1)[0]) in selected_patient_ids
        ]
    if num_samples < 0:
        raise ValueError("--num-samples must be >= 0")
    if num_samples and num_samples < len(keys):
        keys = sorted(random.Random(seed).sample(keys, num_samples))

    result = {
        "modality": modality.upper(),
        "split": split,
        "image_root": str(image_root),
        "boxes_json": str(boxes_path),
        "requested": len(keys),
        "selected_patient_filter": (
            sorted(selected_patient_ids) if selected_patient_ids is not None else None
        ),
        "saved": 0,
        "missing_images": [],
        "warnings": [],
    }
    modality_output = output_root / modality / split
    for key in keys:
        image_path = resolve_image(image_root, split, key)
        if image_path is None:
            result["missing_images"].append(key)
            continue
        safe_name = key.replace("/", "__").replace("\\", "__")
        output_path = modality_output / safe_name
        item_warnings = visualize_one(image_path, boxes_index[key], output_path)
        result["saved"] += 1
        for warning in item_warnings:
            result["warnings"].append(f"{key}: {warning}")

    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset_cfg = config["dataset"]
    modalities = ("mr", "us") if args.modality == "both" else (args.modality,)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    selected_by_modality: dict[str, set[int] | None] = {
        modality: None for modality in modalities
    }
    if not args.all_patients:
        manifest_key = (
            "selected_val_patients_json" if args.split == "val"
            else "selected_patients_json" if args.split == "train"
            else None
        )
        manifest_value = dataset_cfg.get(manifest_key) if manifest_key else None
        if manifest_value and Path(manifest_value).is_file():
            with Path(manifest_value).open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            for modality in modalities:
                field = f"{modality}_patient_ids"
                selected_by_modality[modality] = {
                    int(value) for value in manifest.get(field, [])
                }
            print(f"Using selected-patient manifest: {manifest_value}")
        elif manifest_value:
            print(
                f"Selected-patient manifest not found: {manifest_value}; "
                "sampling from the whole split."
            )

    reports = [
        process_modality(
            dataset_cfg=dataset_cfg,
            modality=modality,
            split=args.split,
            num_samples=args.num_samples,
            seed=args.seed,
            output_root=output_root,
            selected_patient_ids=selected_by_modality[modality],
        )
        for modality in modalities
    ]
    report_path = output_root / f"report_{args.split}.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(reports, handle, ensure_ascii=False, indent=2)

    for report in reports:
        print(
            f"[{report['modality']}/{report['split']}] "
            f"saved={report['saved']}/{report['requested']}, "
            f"missing={len(report['missing_images'])}, "
            f"warnings={len(report['warnings'])}"
        )
    print(f"Images: {output_root.resolve()}")
    print(f"Report: {report_path.resolve()}")

    if any(report["missing_images"] or report["warnings"] for report in reports):
        raise SystemExit(
            "Visualization completed, but path/coordinate problems were found. "
            f"Inspect {report_path}."
        )


if __name__ == "__main__":
    main()
