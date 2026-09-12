"""Patient-first filtering over the existing SAM3 COCO dataset."""

from __future__ import annotations

import random
import re
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image as PILImage
from torch.nn import functional as F
from torch.utils.data import Dataset

from .sample_index import (
    load_boxes_index,
    normalize_sample_key,
    resolve_sample_path,
    validate_final_coco,
)
from .area_labels import area_label_from_ratio, compute_area_ratio, load_area_thresholds
from .boundary_labels import (
    boundary_label_from_scores,
    compute_boundary_scores,
    load_boundary_thresholds,
)


_SLICE_PATTERN = re.compile(r"^slice_(\d+)\.png$", re.IGNORECASE)
_PROMPT_MODES = ("image_only", "text", "coarse_box", "accurate_box")

# Bumped whenever the on-disk label cache layout changes; see
# ``prepare_slice_labels.py`` for the writer.
LABEL_CACHE_VERSION = 1


def _normalize_prompt_probabilities(
    probabilities: Optional[Dict[str, Any]],
    *,
    default: Dict[str, float],
) -> Dict[str, float]:
    """Validate and normalize a prompt-mode probability mapping."""
    values = dict(default if probabilities is None else probabilities)
    unknown = sorted(set(values) - set(_PROMPT_MODES))
    if unknown:
        raise ValueError(
            f"Unknown prompt curriculum modes {unknown}; expected {_PROMPT_MODES}"
        )
    normalized = {mode: float(values.get(mode, 0.0)) for mode in _PROMPT_MODES}
    if any(value < 0.0 for value in normalized.values()):
        raise ValueError("Prompt curriculum probabilities must be non-negative")
    total = sum(normalized.values())
    if total <= 0.0:
        raise ValueError("At least one prompt curriculum probability must be positive")
    return {mode: value / total for mode, value in normalized.items()}


def add_box_noise_xyxy(
    boxes_xyxy: torch.Tensor,
    width: int,
    height: int,
    box_noise_std: float,
    box_noise_max: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Perturb XYXY box corners while keeping boxes valid and in the image.

    ``box_noise_std`` is relative to each box's width/height.  The resulting
    coordinate offsets can additionally be clipped to ``box_noise_max`` pixels.
    """
    boxes = torch.as_tensor(boxes_xyxy, dtype=torch.float32).reshape(-1, 4).clone()
    if boxes.numel() == 0 or box_noise_std == 0.0:
        return boxes
    if box_noise_std < 0.0:
        raise ValueError("box_noise_std must be >= 0")
    if box_noise_max is not None and box_noise_max <= 0.0:
        raise ValueError("box_noise_max must be positive when provided")
    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive")

    box_widths = (boxes[:, 2] - boxes[:, 0]).clamp_min(1.0)
    box_heights = (boxes[:, 3] - boxes[:, 1]).clamp_min(1.0)
    scale = torch.stack((box_widths, box_heights, box_widths, box_heights), dim=-1)
    noise = torch.randn(
        boxes.shape,
        dtype=boxes.dtype,
        device=boxes.device,
        generator=generator,
    ) * (float(box_noise_std) * scale)
    if box_noise_max is not None:
        noise.clamp_(-float(box_noise_max), float(box_noise_max))
    boxes.add_(noise)

    # Sorting handles rare corner crossings under large perturbations.  Keep at
    # least one pixel of width/height so downstream box encoders never see an
    # empty prompt.
    x1 = torch.minimum(boxes[:, 0], boxes[:, 2]).clamp(0.0, max(float(width) - 1.0, 0.0))
    y1 = torch.minimum(boxes[:, 1], boxes[:, 3]).clamp(0.0, max(float(height) - 1.0, 0.0))
    x2 = torch.maximum(boxes[:, 0], boxes[:, 2]).clamp(0.0, float(width))
    y2 = torch.maximum(boxes[:, 1], boxes[:, 3]).clamp(0.0, float(height))
    x2 = torch.maximum(x2, x1 + min(1.0, float(width))).clamp_max(float(width))
    y2 = torch.maximum(y2, y1 + min(1.0, float(height))).clamp_max(float(height))
    return torch.stack((x1, y1, x2, y2), dim=-1)


def _is_valid_xywh_bbox(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return False
    try:
        _, _, width, height = (float(item) for item in value)
    except (TypeError, ValueError):
        return False
    return width > 0 and height > 0


@dataclass(frozen=True)
class PatientSliceRecord:
    image_path: str
    relative_file_name: str
    patient_id: int
    slice_id: str
    slice_index: int
    modality: str
    split: str
    base_dataset_index: int
    image_id: int
    mask_path: Optional[str] = None
    box_prompt: Optional[List[List[float]]] = None
    text_prompt: str = "object"
    modality_label: int = 0
    area_ratio: float = 0.0
    area_label: int = -1
    boundary_contrast: float = 0.0
    boundary_complexity: float = 1.0
    boundary_label: int = -1
    boundary_fallback: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PatientDataset(Dataset):
    """Select numeric patient folders before exposing their SAM3 slices."""

    def __init__(
        self,
        base_dataset: Dataset,
        modality_root: str,
        split: str,
        modality: str,
        num_patients: Optional[int] = None,
        patient_ids: Optional[Sequence[int]] = None,
        sampling_mode: str = "random",
        seed: int = 42,
        mask_root: Optional[str] = None,
        boxes_json: Optional[str] = None,
        strict_dataset_check: bool = True,
        area_thresholds: Optional[str | Dict[str, Any]] = None,
        boundary_thresholds: Optional[str | Dict[str, Any]] = None,
        return_format: str = "sam3",
        max_slices_per_patient: Optional[int] = None,
        box_noise_std: float = 0.0,
        box_noise_max: Optional[float] = None,
        training: bool = False,
        prompt_curriculum: Optional[Dict[str, Any]] = None,
        prompt_seed: int = 42,
        label_cache_dir: Optional[str | Path] = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.modality_root = Path(modality_root)
        self.split = str(split)
        self.modality = str(modality).upper()
        if self.modality not in {"MR", "US"}:
            raise ValueError(f"modality must be MR or US, got {modality}")
        self.split_root = self.modality_root / self.split
        self.mask_root = Path(mask_root) if mask_root else None
        self.mask_split_root = (
            (
                self.mask_root
                if self.mask_root.name.lower() == self.split.lower()
                else self.mask_root / self.split
            )
            if self.mask_root is not None
            else None
        )
        self.strict_dataset_check = bool(strict_dataset_check)
        self.return_format = str(return_format).lower()
        self.max_slices_per_patient = (
            None if max_slices_per_patient is None else int(max_slices_per_patient)
        )
        self.box_noise_std = float(box_noise_std)
        self.box_noise_max = (
            None if box_noise_max is None else float(box_noise_max)
        )
        self.training = bool(training)
        self.current_epoch = 0
        self.prompt_seed = int(prompt_seed)
        self.prompt_curriculum = dict(prompt_curriculum or {})
        self.prompt_curriculum_enabled = bool(
            self.prompt_curriculum.get("enabled", False)
        )
        self.fixed_prompt_text = str(
            self.prompt_curriculum.get("fixed_text", "prostate")
        ).strip()
        if self.prompt_curriculum_enabled and not self.fixed_prompt_text:
            raise ValueError("prompt_curriculum.fixed_text must not be empty")
        self.prompt_start_probabilities = _normalize_prompt_probabilities(
            self.prompt_curriculum.get("start_probabilities"),
            default={
                "image_only": 0.25,
                "text": 0.15,
                "coarse_box": 0.40,
                "accurate_box": 0.20,
            },
        )
        self.prompt_end_probabilities = _normalize_prompt_probabilities(
            self.prompt_curriculum.get("end_probabilities"),
            default={
                "image_only": 0.65,
                "text": 0.15,
                "coarse_box": 0.20,
                "accurate_box": 0.0,
            },
        )
        self.prompt_decay_epochs = max(
            int(self.prompt_curriculum.get("decay_epochs", 20)), 1
        )
        self.prompt_evaluation_mode = str(
            self.prompt_curriculum.get("evaluation_mode", "image_only")
        ).lower()
        if self.prompt_evaluation_mode not in _PROMPT_MODES:
            raise ValueError(
                "prompt_curriculum.evaluation_mode must be one of "
                f"{_PROMPT_MODES}, got {self.prompt_evaluation_mode!r}"
            )
        self.coarse_box_expand_min = float(
            self.prompt_curriculum.get("coarse_box_expand_min", 0.15)
        )
        self.coarse_box_expand_max = float(
            self.prompt_curriculum.get("coarse_box_expand_max", 0.45)
        )
        self.coarse_box_jitter_std = float(
            self.prompt_curriculum.get("coarse_box_jitter_std", 0.10)
        )
        self.coarse_box_jitter_max = self.prompt_curriculum.get(
            "coarse_box_jitter_max", None
        )
        if self.coarse_box_jitter_max is not None:
            self.coarse_box_jitter_max = float(self.coarse_box_jitter_max)
        if not 0.0 <= self.coarse_box_expand_min <= self.coarse_box_expand_max:
            raise ValueError("coarse-box expansion must satisfy 0 <= min <= max")
        if self.coarse_box_jitter_std < 0.0:
            raise ValueError("coarse_box_jitter_std must be non-negative")
        if (
            self.coarse_box_jitter_max is not None
            and self.coarse_box_jitter_max <= 0.0
        ):
            raise ValueError("coarse_box_jitter_max must be positive when provided")
        if self.box_noise_std < 0.0:
            raise ValueError("box_noise_std must be >= 0")
        if self.box_noise_max is not None and self.box_noise_max <= 0.0:
            raise ValueError("box_noise_max must be positive when provided")
        if self.max_slices_per_patient is not None and self.max_slices_per_patient <= 0:
            raise ValueError("max_slices_per_patient must be positive when provided")
        if self.return_format not in {"sam3", "dict"}:
            raise ValueError("return_format must be 'sam3' or 'dict'")
        self.diagnostics: List[str] = []
        if isinstance(area_thresholds, (str, Path)):
            self.area_thresholds = load_area_thresholds(area_thresholds)
        else:
            self.area_thresholds = area_thresholds
        if isinstance(boundary_thresholds, (str, Path)):
            self.boundary_thresholds = load_boundary_thresholds(boundary_thresholds)
        else:
            self.boundary_thresholds = boundary_thresholds
        self.label_cache_dir = Path(label_cache_dir) if label_cache_dir else None
        self.label_cache_split_dir = (
            self.label_cache_dir / self.modality.lower() / self.split
            if self.label_cache_dir is not None
            else None
        )
        self.label_cache_band_width = (
            int(self.boundary_thresholds.get("boundary_band_width", 3))
            if isinstance(self.boundary_thresholds, dict)
            else 3
        )
        self.label_cache_stats: Dict[str, Any] = {
            "dir": str(self.label_cache_dir) if self.label_cache_dir else None,
            "reused": 0,
            "computed": 0,
        }
        coco_data = getattr(self.base_dataset, "coco_data", None)
        if isinstance(coco_data, dict):
            try:
                validate_final_coco(
                    coco_data,
                    independent_masks_available=self.mask_root is not None,
                )
            except (TypeError, ValueError) as error:
                if self.strict_dataset_check:
                    raise
                message = f"Final COCO validation warning: {error}"
                self.diagnostics.append(message)
                warnings.warn(message, RuntimeWarning, stacklevel=2)
        self.boxes_index = (
            load_boxes_index(
                boxes_json,
                strict=self.strict_dataset_check,
                diagnostics=self.diagnostics,
            )
            if boxes_json
            else {}
        )
        available = self._scan_patient_ids()
        self.patient_ids = self._select_patient_ids(
            available,
            num_patients=num_patients,
            patient_ids=patient_ids,
            sampling_mode=sampling_mode,
            seed=seed,
        )
        self.records = self._expand_and_match_slices()

    def set_epoch(self, epoch: int) -> None:
        """Set the deterministic prompt-curriculum epoch."""
        if int(epoch) < 0:
            raise ValueError("epoch must be non-negative")
        self.current_epoch = int(epoch)

    def _prompt_generator(self, index: int, stream: int = 0) -> torch.Generator:
        modality_offset = 0 if self.modality == "MR" else 10_000_019
        seed = (
            self.prompt_seed
            + modality_offset
            + self.current_epoch * 1_000_003
            + int(index) * 9_973
            + int(stream) * 104_729
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return generator

    def _prompt_probabilities(self) -> Dict[str, float]:
        progress = min(float(self.current_epoch) / self.prompt_decay_epochs, 1.0)
        values = {
            mode: self.prompt_start_probabilities[mode]
            + progress
            * (
                self.prompt_end_probabilities[mode]
                - self.prompt_start_probabilities[mode]
            )
            for mode in _PROMPT_MODES
        }
        return _normalize_prompt_probabilities(values, default=values)

    def _sample_prompt_mode(self, index: int) -> str:
        if not self.prompt_curriculum_enabled:
            return "legacy"
        if not self.training:
            return self.prompt_evaluation_mode
        probabilities = self._prompt_probabilities()
        draw = float(torch.rand((), generator=self._prompt_generator(index)).item())
        cumulative = 0.0
        for mode in _PROMPT_MODES:
            cumulative += probabilities[mode]
            if draw <= cumulative:
                return mode
        return _PROMPT_MODES[-1]

    def _scan_patient_ids(self) -> List[int]:
        if not self.split_root.is_dir():
            raise FileNotFoundError(f"Split directory not found: {self.split_root}")
        patient_ids = []
        for child in self.split_root.iterdir():
            if child.is_dir() and child.name.isdigit():
                patient_ids.append(int(child.name))
        patient_ids.sort()
        if not patient_ids:
            raise RuntimeError(f"No numeric patient folders found in {self.split_root}")
        if len(patient_ids) != len(set(patient_ids)):
            raise RuntimeError(f"Duplicate numeric patient IDs in {self.split_root}")
        return patient_ids

    @staticmethod
    def _select_patient_ids(
        available: Sequence[int],
        num_patients: Optional[int],
        patient_ids: Optional[Sequence[int]],
        sampling_mode: str,
        seed: int,
    ) -> List[int]:
        available_set = set(available)
        if patient_ids:
            selected = [int(patient_id) for patient_id in patient_ids]
            if len(selected) != len(set(selected)):
                raise ValueError(f"Explicit patient_ids contains duplicates: {selected}")
            missing = sorted(set(selected) - available_set)
            if missing:
                raise ValueError(f"Explicit patient IDs do not exist: {missing}")
            return sorted(selected)

        count = len(available) if num_patients is None else int(num_patients)
        if count < 0 or count > len(available):
            raise ValueError(
                f"Requested {count} patients, but only {len(available)} are available"
            )
        if count == len(available):
            return list(available)
        mode = sampling_mode.lower()
        if mode == "sequential":
            return list(available[:count])
        if mode == "random":
            return sorted(random.Random(int(seed)).sample(list(available), count))
        raise ValueError(
            "patient_sampling.mode must be 'random' or 'sequential', "
            f"got {sampling_mode!r}"
        )

    def _coco_index_by_file_name(self) -> Dict[str, tuple[int, int]]:
        images = getattr(self.base_dataset, "images", None)
        image_ids = getattr(self.base_dataset, "image_ids", None)
        if not isinstance(images, dict) or image_ids is None:
            raise TypeError("base_dataset must expose COCOSegmentDataset images/image_ids")
        output: Dict[str, tuple[int, int]] = {}
        for dataset_index, image_id in enumerate(image_ids):
            file_name = normalize_sample_key(images[image_id]["file_name"])
            if file_name in output:
                raise RuntimeError(f"Duplicate normalized COCO file_name: {file_name}")
            output[file_name] = (dataset_index, int(image_id))
        return output

    def _problem(self, message: str, error_type: type[Exception] = RuntimeError) -> bool:
        self.diagnostics.append(message)
        if self.strict_dataset_check:
            raise error_type(message)
        warnings.warn(message + "; sample will be skipped", RuntimeWarning, stacklevel=2)
        return False

    def _load_label_cache(self, patient_id: int) -> Dict[str, Dict[str, Any]]:
        """Return cached measurements for one patient, or an empty mapping.

        Shards are written by ``prepare_slice_labels.py``.  A missing, stale or
        unreadable shard is not an error: the caller falls back to computing the
        measurements for that patient.
        """
        if self.label_cache_split_dir is None or self.boundary_thresholds is None:
            return {}
        path = self.label_cache_split_dir / f"{patient_id}.json"
        if not path.is_file():
            return {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            warnings.warn(
                f"Ignoring unreadable label cache shard {path}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            return {}
        if not isinstance(payload, dict):
            return {}
        if int(payload.get("version", -1)) != LABEL_CACHE_VERSION:
            warnings.warn(
                f"Ignoring label cache shard with stale version {path}",
                RuntimeWarning,
                stacklevel=2,
            )
            return {}
        if int(payload.get("boundary_band_width", -1)) != self.label_cache_band_width:
            warnings.warn(
                f"Ignoring label cache shard with stale boundary_band_width {path}",
                RuntimeWarning,
                stacklevel=2,
            )
            return {}
        slices = payload.get("slices")
        return slices if isinstance(slices, dict) else {}

    def _text_prompt_for_image(self, image_id: int) -> str:
        annotations = getattr(self.base_dataset, "img_to_anns", {}).get(image_id, [])
        categories = getattr(self.base_dataset, "categories", {})
        if not annotations:
            return "object"
        return str(categories.get(annotations[0].get("category_id"), "object")).lower()

    def _expand_and_match_slices(self) -> List[PatientSliceRecord]:
        coco_index = self._coco_index_by_file_name()
        records: List[PatientSliceRecord] = []
        for patient_id in self.patient_ids:
            cached_slices = self._load_label_cache(patient_id)
            patient_dir = self.split_root / str(patient_id)
            slices = []
            for path in patient_dir.iterdir():
                match = _SLICE_PATTERN.match(path.name) if path.is_file() else None
                if match:
                    slices.append((int(match.group(1)), path))
            slices.sort(key=lambda item: item[0])
            if self.max_slices_per_patient is not None:
                slices = slices[: self.max_slices_per_patient]
            if not slices:
                raise RuntimeError(f"Patient {self.modality}/{patient_id} has no slice PNGs")
            for slice_index, image_path in slices:
                relative_name = f"{patient_id}/{image_path.name}"
                if relative_name not in coco_index:
                    self._problem(
                        f"Disk slice is missing from final COCO JSON: {relative_name}",
                        KeyError,
                    )
                    continue
                dataset_index, image_id = coco_index[relative_name]
                annotations = getattr(self.base_dataset, "img_to_anns", {}).get(
                    image_id, []
                )
                if not annotations:
                    self._problem(
                        f"Final COCO has no annotation for sample: {relative_name}",
                        ValueError,
                    )
                    continue
                malformed_bbox = next(
                    (
                        annotation.get("bbox")
                        for annotation in annotations
                        if not _is_valid_xywh_bbox(annotation.get("bbox"))
                    ),
                    None,
                )
                if malformed_bbox is not None:
                    self._problem(
                        f"Malformed COCO bbox for {relative_name}: {malformed_bbox!r}",
                        ValueError,
                    )
                    continue
                resolved_image = resolve_sample_path(
                    self.split_root,
                    relative_name,
                    self.split,
                    strict=self.strict_dataset_check,
                    diagnostics=self.diagnostics,
                )
                if resolved_image is None:
                    continue
                mask_path: Optional[Path] = None
                cached_entry = cached_slices.get(image_path.name)
                if self.mask_split_root is not None:
                    mask_path = resolve_sample_path(
                        self.mask_split_root,
                        relative_name,
                        self.split,
                        strict=self.strict_dataset_check,
                        diagnostics=self.diagnostics,
                    )
                    if mask_path is None:
                        continue
                    if cached_entry is not None and "error" in cached_entry:
                        raise RuntimeError(
                            f"Cached label extraction failed for {relative_name}: "
                            f"{cached_entry['error']}"
                        )
                    if not (
                        cached_entry is not None and cached_entry.get("mask_verified")
                    ):
                        try:
                            with PILImage.open(mask_path) as mask_image:
                                mask_image.verify()
                        except Exception as error:
                            self._problem(
                                f"Malformed mask PNG for {relative_name}: {error}",
                                ValueError,
                            )
                            continue
                boxes = self.boxes_index.get(relative_name) if self.boxes_index else None
                if self.boxes_index and boxes is None:
                    self._problem(
                        f"Sample is missing from boxes JSON: {relative_name}", KeyError
                    )
                    continue
                if boxes:
                    image_info = self.base_dataset.images[image_id]
                    width, height = int(image_info["width"]), int(image_info["height"])
                    if any(
                        box[0] < 0
                        or box[1] < 0
                        or box[2] > width
                        or box[3] > height
                        for box in boxes
                    ):
                        self._problem(
                            f"Box prompt is outside image bounds for {relative_name}: "
                            f"boxes={boxes}, size=({width}, {height})",
                            ValueError,
                        )
                        continue
                boundary_contrast = 0.0
                boundary_complexity = 1.0
                boundary_fallback = False
                if mask_path is not None and cached_entry is not None:
                    area_ratio = float(cached_entry["area_ratio"])
                    boundary_contrast = float(cached_entry["boundary_contrast"])
                    boundary_complexity = float(cached_entry["boundary_complexity"])
                    boundary_fallback = bool(
                        cached_entry.get("boundary_fallback", False)
                    )
                    self.label_cache_stats["reused"] += 1
                else:
                    area_ratio = (
                        compute_area_ratio(mask_path) if mask_path is not None else 0.0
                    )
                    self.label_cache_stats["computed"] += 1
                    if self.boundary_thresholds is not None and mask_path is not None:
                        boundary_scores = compute_boundary_scores(
                            resolved_image, mask_path, self.label_cache_band_width
                        )
                        boundary_contrast = boundary_scores.contrast
                        boundary_complexity = boundary_scores.complexity
                        boundary_fallback = boundary_scores.used_fallback
                area_label = -1
                if self.area_thresholds is not None:
                    area_label = area_label_from_ratio(
                        area_ratio,
                        float(self.area_thresholds["small_max"]),
                        float(self.area_thresholds["medium_max"]),
                    )
                boundary_label = -1
                if self.boundary_thresholds is not None and mask_path is not None:
                    modality_thresholds = self.boundary_thresholds[
                        self.modality.lower()
                    ]
                    boundary_label = boundary_label_from_scores(
                        boundary_contrast,
                        boundary_complexity,
                        float(modality_thresholds["contrast_low"]),
                        float(modality_thresholds["complexity_high"]),
                    )
                records.append(
                    PatientSliceRecord(
                        image_path=str(resolved_image),
                        relative_file_name=relative_name,
                        patient_id=patient_id,
                        slice_id=image_path.stem,
                        slice_index=slice_index,
                        modality=self.modality,
                        split=self.split,
                        base_dataset_index=dataset_index,
                        image_id=image_id,
                        mask_path=str(mask_path) if mask_path is not None else None,
                        box_prompt=boxes,
                        text_prompt=self._text_prompt_for_image(image_id),
                        modality_label=0 if self.modality == "MR" else 1,
                        area_ratio=area_ratio,
                        area_label=area_label,
                        boundary_contrast=boundary_contrast,
                        boundary_complexity=boundary_complexity,
                        boundary_label=boundary_label,
                        boundary_fallback=boundary_fallback,
                    )
                )
        return records

    @staticmethod
    def _normalized_cxcywh_boxes(
        boxes_xyxy: Sequence[Sequence[float]], width: int, height: int
    ) -> torch.Tensor:
        boxes = torch.as_tensor(boxes_xyxy, dtype=torch.float32)
        x1, y1, x2, y2 = boxes.unbind(dim=-1)
        return torch.stack(
            (
                (x1 + x2) / (2.0 * width),
                (y1 + y2) / (2.0 * height),
                (x2 - x1) / width,
                (y2 - y1) / height,
            ),
            dim=-1,
        ).clamp(0.0, 1.0)

    @property
    def num_slices(self) -> int:
        return len(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def _box_prompt_for_record(
        self,
        record: PatientSliceRecord,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        boxes = torch.tensor(record.box_prompt or [], dtype=torch.float32).reshape(-1, 4)
        if boxes.numel() == 0 or self.box_noise_std == 0.0:
            return boxes
        image_info = self.base_dataset.images[record.image_id]
        return add_box_noise_xyxy(
            boxes,
            width=int(image_info["width"]),
            height=int(image_info["height"]),
            box_noise_std=self.box_noise_std,
            box_noise_max=self.box_noise_max,
            generator=generator,
        )

    def _coarse_box_prompt_for_record(
        self,
        record: PatientSliceRecord,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Create a deliberately imprecise, expanded deployment-style box."""
        boxes = torch.tensor(record.box_prompt or [], dtype=torch.float32).reshape(-1, 4)
        if boxes.numel() == 0:
            return boxes
        image_info = self.base_dataset.images[record.image_id]
        width = float(image_info["width"])
        height = float(image_info["height"])
        box_width = (boxes[:, 2] - boxes[:, 0]).clamp_min(1.0)
        box_height = (boxes[:, 3] - boxes[:, 1]).clamp_min(1.0)
        expansion = torch.empty(len(boxes)).uniform_(
            self.coarse_box_expand_min,
            self.coarse_box_expand_max,
            generator=generator,
        )
        shift_x = torch.randn(len(boxes), generator=generator) * (
            self.coarse_box_jitter_std * box_width
        )
        shift_y = torch.randn(len(boxes), generator=generator) * (
            self.coarse_box_jitter_std * box_height
        )
        if self.coarse_box_jitter_max is not None:
            shift_x.clamp_(
                -self.coarse_box_jitter_max, self.coarse_box_jitter_max
            )
            shift_y.clamp_(
                -self.coarse_box_jitter_max, self.coarse_box_jitter_max
            )
        output = boxes.clone()
        output[:, 0] -= expansion * box_width
        output[:, 2] += expansion * box_width
        output[:, 1] -= expansion * box_height
        output[:, 3] += expansion * box_height
        output[:, (0, 2)] += shift_x[:, None]
        output[:, (1, 3)] += shift_y[:, None]
        output[:, 0].clamp_(0.0, max(width - 1.0, 0.0))
        output[:, 1].clamp_(0.0, max(height - 1.0, 0.0))
        output[:, 2].clamp_(1.0, width)
        output[:, 3].clamp_(1.0, height)
        output[:, 2] = torch.maximum(output[:, 2], output[:, 0] + 1.0).clamp_max(width)
        output[:, 3] = torch.maximum(output[:, 3], output[:, 1] + 1.0).clamp_max(height)
        return output

    def _resolve_prompt(
        self,
        index: int,
        record: PatientSliceRecord,
    ) -> tuple[str, str, torch.Tensor]:
        mode = self._sample_prompt_mode(index)
        generator = self._prompt_generator(index, stream=1)
        if mode == "legacy":
            return mode, record.text_prompt, self._box_prompt_for_record(
                record, generator=generator
            )
        if mode == "image_only":
            return mode, self.fixed_prompt_text, torch.empty((0, 4))
        if mode == "text":
            return mode, record.text_prompt, torch.empty((0, 4))
        if mode == "coarse_box":
            return mode, self.fixed_prompt_text, self._coarse_box_prompt_for_record(
                record, generator
            )
        if mode == "accurate_box":
            return mode, self.fixed_prompt_text, self._box_prompt_for_record(
                record, generator=generator
            )
        raise RuntimeError(f"Unhandled prompt mode: {mode}")

    def _load_sam3_datapoint(
        self,
        index: int,
        prompt_boxes_xyxy: Optional[torch.Tensor] = None,
        prompt_mode: Optional[str] = None,
        prompt_text: Optional[str] = None,
    ):
        record = self.records[index]
        datapoint = self.base_dataset[record.base_dataset_index]
        if record.mask_path is not None:
            mask_array = np.asarray(PILImage.open(record.mask_path).convert("L")).copy()
            mask = torch.from_numpy(mask_array > 0).float()[None, None]
            for image in datapoint.images:
                resized = F.interpolate(mask, size=image.size, mode="nearest").squeeze() > 0.5
                for obj in image.objects:
                    obj.segment = resized
        if prompt_boxes_xyxy is None:
            prompt_mode, prompt_text, prompt_boxes_xyxy = self._resolve_prompt(
                index, record
            )
        else:
            prompt_mode = prompt_mode or "explicit"
            prompt_text = prompt_text or record.text_prompt
        prompt_boxes_xyxy = torch.as_tensor(
            prompt_boxes_xyxy, dtype=torch.float32
        ).reshape(-1, 4)
        if prompt_boxes_xyxy.numel() > 0:
            image_info = self.base_dataset.images[record.image_id]
            prompt_boxes = self._normalized_cxcywh_boxes(
                prompt_boxes_xyxy,
                width=int(image_info["width"]),
                height=int(image_info["height"]),
            )
            for query in datapoint.find_queries:
                query.input_bbox = prompt_boxes
                query.input_bbox_label = torch.ones(
                    len(prompt_boxes), dtype=torch.long
                )
        for query in datapoint.find_queries:
            query.query_text = str(prompt_text or self.fixed_prompt_text)
            if prompt_boxes_xyxy.numel() == 0:
                query.input_bbox = None
                query.input_bbox_label = None
        # Datapoint is intentionally left structurally compatible with the
        # existing SAM3 collator. Metadata is also available via records[index].
        metadata = record.to_dict()
        metadata.update(
            {
                "prompt_mode": prompt_mode,
                "applied_prompt_text": str(prompt_text or self.fixed_prompt_text),
                "applied_box_prompt": prompt_boxes_xyxy.tolist(),
                "uses_external_bbox": bool(prompt_boxes_xyxy.numel()),
                "deployment_image_only": prompt_mode == "image_only",
            }
        )
        try:
            setattr(datapoint, "patient_metadata", metadata)
        except (AttributeError, TypeError):
            pass
        return datapoint

    def get_moe_sample(self, index: int) -> Dict[str, Any]:
        """Return the unified tensor/label dictionary used by staged MoE training."""
        record = self.records[index]
        prompt_mode, prompt_text, boxes = self._resolve_prompt(index, record)
        datapoint = self._load_sam3_datapoint(
            index,
            prompt_boxes_xyxy=boxes,
            prompt_mode=prompt_mode,
            prompt_text=prompt_text,
        )
        image = datapoint.images[0]
        segments = [obj.segment for obj in image.objects if obj.segment is not None]
        if segments:
            mask_gt = torch.stack([segment.bool() for segment in segments]).any(dim=0)
        else:
            mask_gt = torch.zeros(image.size, dtype=torch.bool)
        return {
            "image": image.data,
            "mask_gt": mask_gt,
            "box_prompt": boxes,
            "text_prompt": record.text_prompt,
            "applied_prompt_text": prompt_text,
            "prompt_mode": prompt_mode,
            "modality_label": record.modality_label,
            "patient_id": record.patient_id,
            "slice_id": record.slice_id,
            "area_ratio": record.area_ratio,
            "area_label": record.area_label,
            "boundary_contrast": record.boundary_contrast,
            "boundary_complexity": record.boundary_complexity,
            "boundary_label": record.boundary_label,
            "image_path": record.image_path,
            "mask_path": record.mask_path,
        }

    def __getitem__(self, index: int):
        if self.return_format == "dict":
            return self.get_moe_sample(index)
        return self._load_sam3_datapoint(index)


def save_selected_patients(
    datasets: Sequence[PatientDataset],
    output_path: str,
    sampling_mode: str,
    seed: int,
    resample_patients_each_epoch: bool = False,
) -> Dict[str, Any]:
    """Persist the exact patient-level split used by a training run."""
    selected = {"MR": [], "US": []}
    slice_counts = {"MR": 0, "US": 0}
    for dataset in datasets:
        selected[dataset.modality] = list(dataset.patient_ids)
        slice_counts[dataset.modality] += dataset.num_slices
    manifest = {
        "sampling_mode": str(sampling_mode).lower(),
        "seed": int(seed),
        "resample_patients_each_epoch": bool(resample_patients_each_epoch),
        "mr_patient_ids": selected["MR"],
        "us_patient_ids": selected["US"],
        "num_mr_patients": len(selected["MR"]),
        "num_us_patients": len(selected["US"]),
        "num_mr_slices": slice_counts["MR"],
        "num_us_slices": slice_counts["US"],
        "total_slices": slice_counts["MR"] + slice_counts["US"],
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    temp_path.replace(path)
    return manifest
