"""Area-ratio labels derived from independent binary mask PNG files."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image as PILImage


AREA_NAMES = ("small", "medium", "large")


def compute_area_ratio(mask_path: str | Path) -> float:
    """Return foreground pixels divided by all pixels, always finite in [0, 1]."""
    mask = np.asarray(PILImage.open(mask_path).convert("L")) > 0
    if mask.size == 0:
        raise ValueError(f"Mask has zero pixels: {mask_path}")
    ratio = float(np.count_nonzero(mask) / mask.size)
    if not np.isfinite(ratio):
        raise ValueError(f"Non-finite area ratio for mask: {mask_path}")
    return min(max(ratio, 0.0), 1.0)


def area_label_from_ratio(
    area_ratio: float, small_max: float, medium_max: float
) -> int:
    """Map a ratio to small=0, medium=1, large=2."""
    if not np.isfinite(area_ratio):
        raise ValueError(f"area_ratio must be finite, got {area_ratio}")
    if not 0 <= small_max < medium_max <= 1:
        raise ValueError(
            f"Expected 0 <= small_max < medium_max <= 1, got "
            f"{small_max}, {medium_max}"
        )
    if area_ratio < small_max:
        return 0
    if area_ratio < medium_max:
        return 1
    return 2


def load_area_thresholds(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        thresholds = json.load(handle)
    for field in ("small_max", "medium_max"):
        if field not in thresholds:
            raise ValueError(f"Area threshold file is missing {field!r}: {path}")
    small_max = float(thresholds["small_max"])
    medium_max = float(thresholds["medium_max"])
    area_label_from_ratio(0.0, small_max, medium_max)
    thresholds["small_max"] = small_max
    thresholds["medium_max"] = medium_max
    return thresholds


def _fallback_thresholds(
    ratios: Sequence[float], fallback_small: float, fallback_medium: float
) -> Tuple[float, float, str]:
    if ratios:
        center = float(np.median(np.asarray(ratios, dtype=np.float64)))
        margin = max(1e-6, min(0.05, max(center * 0.1, 1e-4)))
        low = max(0.0, center - margin)
        high = min(1.0, center + margin)
        if low < high:
            return low, high, "median_margin"
    if not 0 <= fallback_small < fallback_medium <= 1:
        raise ValueError(
            "Fallback area thresholds must satisfy "
            "0 <= small_max < medium_max <= 1"
        )
    return float(fallback_small), float(fallback_medium), "configured_fixed"


def compute_area_thresholds(
    ratios: Sequence[float],
    mode: Dict[str, Any],
    mr_patient_ids: Sequence[int],
    us_patient_ids: Sequence[int],
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Compute train-only quantile/fixed thresholds and class counts."""
    exclude_empty = bool(mode.get("exclude_empty_masks", True))
    values = [float(value) for value in ratios if np.isfinite(value)]
    if exclude_empty:
        values = [value for value in values if value > 0]
    if any(value < 0 or value > 1 for value in values):
        raise ValueError("All area ratios must be in [0, 1]")

    mode_type = str(mode.get("type", "quantile")).lower()
    q1 = float(mode.get("q1", 0.33))
    q2 = float(mode.get("q2", 0.67))
    fallback_used = False
    fallback_strategy = None

    if mode_type == "fixed":
        small_max = float(mode.get("small_max", 0.01))
        medium_max = float(mode.get("medium_max", 0.05))
        area_label_from_ratio(0.0, small_max, medium_max)
    elif mode_type == "quantile":
        if not 0 < q1 < q2 < 1:
            raise ValueError(f"Expected 0 < q1 < q2 < 1, got {q1}, {q2}")
        if len(values) >= 3:
            small_max, medium_max = np.quantile(values, [q1, q2]).tolist()
        else:
            small_max = medium_max = float("nan")
        if len(values) < 3 or not small_max < medium_max:
            warnings.warn(
                "Area threshold sample count is insufficient or q33 == q67; "
                "using a safe fallback strategy",
                RuntimeWarning,
                stacklevel=2,
            )
            small_max, medium_max, fallback_strategy = _fallback_thresholds(
                values,
                float(mode.get("fallback_small_max", 0.01)),
                float(mode.get("fallback_medium_max", 0.05)),
            )
            fallback_used = True
    else:
        raise ValueError(f"Unknown area threshold mode: {mode_type!r}")

    counts = {name: 0 for name in AREA_NAMES}
    for ratio in values:
        counts[AREA_NAMES[area_label_from_ratio(ratio, small_max, medium_max)]] += 1
    thresholds = {
        "mode": mode_type,
        "q1": q1,
        "q2": q2,
        "small_max": float(small_max),
        "medium_max": float(medium_max),
        "num_samples": len(values),
        "exclude_empty_masks": exclude_empty,
        "mr_patient_ids": [int(value) for value in mr_patient_ids],
        "us_patient_ids": [int(value) for value in us_patient_ids],
        "fallback_used": fallback_used,
        "fallback_strategy": fallback_strategy,
        "class_counts": counts,
    }
    return thresholds, counts


def save_area_thresholds(thresholds: Dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(thresholds, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)

