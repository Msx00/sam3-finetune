"""Automatic image-derived boundary pseudo-labels (not clinical annotations)."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image as PILImage
from torch.nn import functional as F


BOUNDARY_NAMES = ("clear", "fuzzy", "complex")


@dataclass(frozen=True)
class BoundaryScores:
    contrast: float
    complexity: float
    used_fallback: bool
    fallback_reason: str | None = None


def _validate_kernel_size(kernel_size: int) -> int:
    kernel_size = int(kernel_size)
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"boundary_band_width must be a positive odd number, got {kernel_size}")
    return kernel_size


def boundary_bands(mask: np.ndarray, kernel_size: int = 3) -> Tuple[np.ndarray, ...]:
    """Return inner, outer and union boundary bands for a binary mask."""
    kernel_size = _validate_kernel_size(kernel_size)
    binary = np.asarray(mask, dtype=bool)
    tensor = torch.from_numpy(binary.copy()).float()[None, None]
    pad = kernel_size // 2
    dilated = F.max_pool2d(
        F.pad(tensor, (pad, pad, pad, pad), value=0.0), kernel_size, stride=1
    ) > 0.5
    complement = 1.0 - tensor
    eroded = 1.0 - F.max_pool2d(
        F.pad(complement, (pad, pad, pad, pad), value=1.0),
        kernel_size,
        stride=1,
    )
    eroded = eroded > 0.5
    binary_tensor = tensor > 0.5
    inner = (binary_tensor & ~eroded).squeeze().numpy()
    outer = (dilated & ~binary_tensor).squeeze().numpy()
    return inner, outer, np.logical_or(inner, outer)


def _pixel_perimeter(mask: np.ndarray) -> float:
    padded = np.pad(np.asarray(mask, dtype=np.uint8), 1, mode="constant")
    horizontal = np.abs(np.diff(padded.astype(np.int16), axis=1)).sum()
    vertical = np.abs(np.diff(padded.astype(np.int16), axis=0)).sum()
    return float(horizontal + vertical)


def compute_boundary_scores(
    image_path: str | Path,
    mask_path: str | Path,
    boundary_band_width: int = 3,
    eps: float = 1e-6,
) -> BoundaryScores:
    """Compute contrast and shape complexity with finite fallback values."""
    image = np.asarray(PILImage.open(image_path).convert("L"), dtype=np.float32) / 255.0
    mask = np.asarray(PILImage.open(mask_path).convert("L")) > 0
    if image.shape != mask.shape:
        raise ValueError(
            f"Image/mask shape mismatch: image={image.shape}, mask={mask.shape}, "
            f"image_path={image_path}, mask_path={mask_path}"
        )
    area = int(mask.sum())
    if area == 0:
        return BoundaryScores(0.0, 1.0, True, "empty_mask")

    inner, outer, boundary = boundary_bands(mask, boundary_band_width)
    perimeter = _pixel_perimeter(mask)
    complexity = float(perimeter * perimeter / (4.0 * math.pi * area + eps))
    fallback_reasons: List[str] = []
    if not inner.any() or not outer.any() or not boundary.any():
        contrast = 0.0
        fallback_reasons.append("empty_boundary_band")
    else:
        contrast_raw = abs(float(image[inner].mean()) - float(image[outer].mean()))
        local_std = float(image[boundary].std())
        contrast = float(contrast_raw / (local_std + eps))
    if not np.isfinite(contrast):
        contrast = 0.0
        fallback_reasons.append("non_finite_contrast")
    if not np.isfinite(complexity) or complexity <= 0:
        complexity = 1.0
        fallback_reasons.append("non_finite_complexity")
    contrast = float(np.clip(contrast, 0.0, 1e6))
    complexity = float(np.clip(complexity, eps, 1e6))
    return BoundaryScores(
        contrast,
        complexity,
        bool(fallback_reasons),
        ",".join(fallback_reasons) if fallback_reasons else None,
    )


def boundary_label_from_scores(
    contrast: float, complexity: float, contrast_low: float, complexity_high: float
) -> int:
    """Apply fuzzy-first, then complex, then clear pseudo-label priority."""
    values = (contrast, complexity, contrast_low, complexity_high)
    if not all(np.isfinite(value) for value in values):
        raise ValueError(f"Boundary score/threshold is non-finite: {values}")
    if contrast < contrast_low:
        return 1
    if complexity > complexity_high:
        return 2
    return 0


def _modality_thresholds(
    valid_scores: Sequence[BoundaryScores],
    contrast_quantile: float,
    complexity_quantile: float,
) -> Tuple[float, float, bool]:
    if valid_scores:
        contrast = np.asarray([score.contrast for score in valid_scores], dtype=np.float64)
        complexity = np.asarray([score.complexity for score in valid_scores], dtype=np.float64)
        low = float(np.quantile(contrast, contrast_quantile))
        high = float(np.quantile(complexity, complexity_quantile))
        if np.isfinite(low) and np.isfinite(high):
            return low, high, False
    warnings.warn(
        "No valid boundary scores for a modality; using contrast_low=0 and "
        "complexity_high=1 fallback thresholds",
        RuntimeWarning,
        stacklevel=2,
    )
    return 0.0, 1.0, True


def compute_boundary_thresholds(
    scores_by_modality: Dict[str, Sequence[BoundaryScores]],
    contrast_quantile: float = 0.33,
    complexity_quantile: float = 0.67,
    boundary_band_width: int = 3,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, int]]]:
    """Compute separate MR/US TRAIN thresholds for automatic pseudo-labels."""
    _validate_kernel_size(boundary_band_width)
    if not 0 < contrast_quantile < 1 or not 0 < complexity_quantile < 1:
        raise ValueError("Boundary quantiles must lie strictly between 0 and 1")
    output: Dict[str, Any] = {}
    counts_by_modality: Dict[str, Dict[str, int]] = {}
    for modality in ("mr", "us"):
        all_scores = list(scores_by_modality.get(modality, []))
        valid = [score for score in all_scores if not score.used_fallback]
        contrast_low, complexity_high, threshold_fallback = _modality_thresholds(
            valid, contrast_quantile, complexity_quantile
        )
        counts = {name: 0 for name in BOUNDARY_NAMES}
        for score in all_scores:
            label = boundary_label_from_scores(
                score.contrast, score.complexity, contrast_low, complexity_high
            )
            counts[BOUNDARY_NAMES[label]] += 1
        fallback_reasons: Dict[str, int] = {}
        for score in all_scores:
            if score.used_fallback:
                reason = score.fallback_reason or "unknown"
                fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
        output[modality] = {
            "contrast_low": contrast_low,
            "complexity_high": complexity_high,
            "num_samples": len(all_scores),
            "num_valid_samples": len(valid),
            "fallback_count": len(all_scores) - len(valid),
            "fallback_reasons": fallback_reasons,
            "threshold_fallback_used": threshold_fallback,
            "class_counts": counts,
        }
        counts_by_modality[modality] = counts
    output.update(
        {
            "boundary_band_width": int(boundary_band_width),
            "contrast_quantile": float(contrast_quantile),
            "complexity_quantile": float(complexity_quantile),
            "label_rule": "fuzzy_first_then_complex_then_clear",
            "label_source": "automatic_image_pseudo_label_not_clinical_annotation",
        }
    )
    return output, counts_by_modality


def load_boundary_thresholds(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        thresholds = json.load(handle)
    for modality in ("mr", "us"):
        for key in ("contrast_low", "complexity_high"):
            if key not in thresholds.get(modality, {}):
                raise ValueError(f"Boundary threshold file lacks {modality}.{key}: {path}")
    return thresholds


def save_boundary_thresholds(thresholds: Dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(thresholds, handle, ensure_ascii=False, indent=2)
    temporary.replace(output)

