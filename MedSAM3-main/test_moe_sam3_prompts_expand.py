#!/usr/bin/env python3
"""Run prompt evaluation after enlarging XYXY box prompts around their centers.

This is a thin extension of ``test_moe_sam3_prompts.py``.  All model loading,
dataset selection, inference, metrics, and box-noise behavior remain in the
original module.  The only added operation is:

    raw box -> center-preserving enlargement -> existing box noise -> inference
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from torch.utils.data import Dataset

import test_moe_sam3_prompts as base


_original_build_parser = base.build_parser
_original_apply_config_defaults = base.apply_config_defaults
_OriginalPromptModeDataset = base.PromptModeDataset
_box_expand_scale = 1.2


def expand_boxes_xyxy(
    boxes: Sequence[Sequence[float]],
    width: int,
    height: int,
    scale: float,
) -> list[list[float]]:
    """Scale XYXY boxes about their centers and clip them to image bounds."""
    expanded: list[list[float]] = []
    for values in boxes:
        if len(values) != 4:
            raise ValueError(f"Expected an XYXY box with 4 values, got {values!r}")
        x1, y1, x2, y2 = (float(value) for value in values)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid XYXY box: {values!r}")

        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        half_width = (x2 - x1) * scale / 2.0
        half_height = (y2 - y1) * scale / 2.0
        expanded.append(
            [
                max(0.0, center_x - half_width),
                max(0.0, center_y - half_height),
                min(float(width), center_x + half_width),
                min(float(height), center_y + half_height),
            ]
        )
    return expanded


class _ExpandedBoxDataset(Dataset):
    """Insert box enlargement immediately after the source sample is read."""

    def __init__(self, dataset: Dataset, scale: float) -> None:
        self.dataset = dataset
        self.scale = float(scale)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        datapoint = self.dataset[index]
        metadata = dict(getattr(datapoint, "patient_metadata", None) or {})
        raw_boxes = metadata.get("box_prompt_unexpanded")
        if raw_boxes is None:
            raw_boxes = metadata.get("box_prompt") or []

        if raw_boxes:
            with base.Image.open(metadata["image_path"]) as handle:
                width, height = handle.size
            expanded_boxes = expand_boxes_xyxy(
                raw_boxes,
                width=width,
                height=height,
                scale=self.scale,
            )
            metadata["box_prompt_unexpanded"] = [
                [float(value) for value in box] for box in raw_boxes
            ]
            metadata["box_prompt"] = expanded_boxes
            metadata["box_expand_scale"] = self.scale
            metadata["box_expansion_applied"] = self.scale != 1.0

        setattr(datapoint, "patient_metadata", metadata)
        return datapoint


class PromptModeDataset(_OriginalPromptModeDataset):
    """Use the original prompt wrapper with an enlarged-box source dataset."""

    def __init__(self, dataset: Dataset, prompt_mode: str, *args: Any, **kwargs: Any) -> None:
        if prompt_mode in {"box", "text_box"}:
            dataset = _ExpandedBoxDataset(dataset, _box_expand_scale)
        super().__init__(dataset, prompt_mode, *args, **kwargs)


def build_parser():
    parser = _original_build_parser()
    parser.description = (
        "Evaluate HMOE-SAM3 after center-preserving box-prompt enlargement."
    )
    parser.add_argument(
        "--box-expand-scale",
        type=float,
        default=None,
        help=(
            "Width/height multiplier applied before box noise; 1.0 disables "
            "enlargement (default: evaluation.box_expand_scale or 1.2)."
        ),
    )
    return parser


def apply_config_defaults(args, config: Mapping[str, Any]):
    global _box_expand_scale
    args = _original_apply_config_defaults(args, config)
    configured = base._config_get(config, "box_expand_scale")
    args.box_expand_scale = float(
        args.box_expand_scale
        if args.box_expand_scale is not None
        else (configured if configured is not None else 1.2)
    )
    if args.box_expand_scale < 1.0:
        raise ValueError("evaluation.box_expand_scale must be >= 1.0")
    _box_expand_scale = args.box_expand_scale
    return args


base.build_parser = build_parser
base.apply_config_defaults = apply_config_defaults
base.PromptModeDataset = PromptModeDataset


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _finite_mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return sum(values) / len(values) if values else math.nan


def _finite_count(rows: Sequence[Mapping[str, Any]], key: str) -> int:
    count = 0
    for row in rows:
        try:
            count += int(math.isfinite(float(row.get(key))))
        except (TypeError, ValueError):
            pass
    return count


def write_patient_dice_hd95_outputs(output_dir: Path) -> None:
    """Derive patient metrics directly from the always-written image CSV."""
    image_rows = _read_csv(output_dir / "metrics_per_image.csv")
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in image_rows:
        grouped[
            (str(row["prompt_mode"]), str(row["modality"]), str(row["patient_id"]))
        ].append(row)

    all_patient_rows: list[dict[str, Any]] = []
    small_patient_rows: list[dict[str, Any]] = []
    for (prompt_mode, modality, patient_id), rows in sorted(grouped.items()):
        evaluated = [
            row for row in rows if _truthy(row.get("gt_available", True))
        ]
        small = [row for row in evaluated if _truthy(row.get("is_small_target"))]
        all_patient_rows.append(
            {
                "prompt_mode": prompt_mode,
                "modality": modality,
                "patient_id": patient_id,
                "num_slices": len(rows),
                "num_evaluated_slices": len(evaluated),
                "num_valid_hd95_slices": _finite_count(evaluated, "hd95"),
                "dice": _finite_mean(evaluated, "dice"),
                "hd95": _finite_mean(evaluated, "hd95"),
            }
        )
        if small:
            small_patient_rows.append(
                {
                    "prompt_mode": prompt_mode,
                    "modality": modality,
                    "patient_id": patient_id,
                    "num_small_target_slices": len(small),
                    "num_evaluated_small_target_slices": len(small),
                    "num_valid_hd95_small_target_slices": _finite_count(small, "hd95"),
                    "dice": _finite_mean(small, "dice"),
                    "hd95": _finite_mean(small, "hd95"),
                }
            )

    _write_csv(output_dir / "patient_dice_hd95.csv", all_patient_rows)
    _write_csv(
        output_dir / "patient_small_target_dice_hd95.csv",
        small_patient_rows,
    )

    macro_rows: list[dict[str, Any]] = []
    prompt_modes = sorted({str(row["prompt_mode"]) for row in all_patient_rows})
    for prompt_mode in prompt_modes:
        mode_all = [row for row in all_patient_rows if row["prompt_mode"] == prompt_mode]
        mode_small = [
            row for row in small_patient_rows if row["prompt_mode"] == prompt_mode
        ]
        modalities = sorted({str(row["modality"]) for row in mode_all})
        for modality in ["overall", *modalities]:
            scoped_all = (
                mode_all
                if modality == "overall"
                else [row for row in mode_all if row["modality"] == modality]
            )
            scoped_small = (
                mode_small
                if modality == "overall"
                else [row for row in mode_small if row["modality"] == modality]
            )
            for target_scope, scoped_rows in (
                ("all", scoped_all),
                ("small", scoped_small),
            ):
                macro_rows.append(
                    {
                        "prompt_mode": prompt_mode,
                        "modality": modality,
                        "target_scope": target_scope,
                        "num_patients_total": len(scoped_all),
                        "num_patients_evaluated": len(scoped_rows),
                        "num_valid_hd95_patients": _finite_count(scoped_rows, "hd95"),
                        "dice": _finite_mean(scoped_rows, "dice"),
                        "hd95": _finite_mean(scoped_rows, "hd95"),
                    }
                )
    _write_csv(output_dir / "patient_macro_dice_hd95.csv", macro_rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)

    # Resolve the final output directory so the added transformation is also
    # recorded alongside the original run configuration.
    preview_args = build_parser().parse_args(effective_argv)
    preview_config, _ = base.load_evaluation_config(
        Path(preview_args.config).expanduser().resolve()
    )
    preview_args = apply_config_defaults(preview_args, preview_config)

    return_code = base.main(effective_argv)
    output_dir = Path(preview_args.output_dir).expanduser().resolve()
    write_patient_dice_hd95_outputs(output_dir)
    run_config_path = output_dir / "run_config.json"
    if run_config_path.is_file():
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        run_config.update(
            {
                "box_expand_scale": preview_args.box_expand_scale,
                "box_expansion_note": (
                    "XYXY boxes are enlarged about their centers and clipped to "
                    "the image before the original Gaussian box-noise stage."
                ),
                "patient_metric_note": (
                    "Per-patient Dice/HD95 are means of finite 2D slice metrics "
                    "within each patient; they are not reconstructed 3D metrics."
                ),
                "patient_metric_outputs": {
                    "all_targets": "patient_dice_hd95.csv",
                    "small_targets": "patient_small_target_dice_hd95.csv",
                    "patient_macro": "patient_macro_dice_hd95.csv",
                },
            }
        )
        run_config_path.write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
