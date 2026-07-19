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
                area_ratio = compute_area_ratio(mask_path) if mask_path is not None else 0.0
                area_label = -1
                if self.area_thresholds is not None:
                    area_label = area_label_from_ratio(
                        area_ratio,
                        float(self.area_thresholds["small_max"]),
                        float(self.area_thresholds["medium_max"]),
                    )
                boundary_contrast = 0.0
                boundary_complexity = 1.0
                boundary_label = -1
                boundary_fallback = False
                if self.boundary_thresholds is not None and mask_path is not None:
                    band_width = int(
                        self.boundary_thresholds.get("boundary_band_width", 3)
                    )
                    boundary_scores = compute_boundary_scores(
                        resolved_image, mask_path, band_width
                    )
                    modality_thresholds = self.boundary_thresholds[
                        self.modality.lower()
                    ]
                    boundary_contrast = boundary_scores.contrast
                    boundary_complexity = boundary_scores.complexity
                    boundary_fallback = boundary_scores.used_fallback
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
        boxes = torch.tensor(boxes_xyxy, dtype=torch.float32)
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

    def _load_sam3_datapoint(self, index: int):
        record = self.records[index]
        datapoint = self.base_dataset[record.base_dataset_index]
        if record.mask_path is not None:
            mask_array = np.asarray(PILImage.open(record.mask_path).convert("L")).copy()
            mask = torch.from_numpy(mask_array > 0).float()[None, None]
            for image in datapoint.images:
                resized = F.interpolate(mask, size=image.size, mode="nearest").squeeze() > 0.5
                for obj in image.objects:
                    obj.segment = resized
        if record.box_prompt:
            image_info = self.base_dataset.images[record.image_id]
            prompt_boxes = self._normalized_cxcywh_boxes(
                record.box_prompt,
                width=int(image_info["width"]),
                height=int(image_info["height"]),
            )
            for query in datapoint.find_queries:
                query.input_bbox = prompt_boxes
                query.input_bbox_label = torch.ones(
                    len(prompt_boxes), dtype=torch.long
                )
        # Datapoint is intentionally left structurally compatible with the
        # existing SAM3 collator. Metadata is also available via records[index].
        try:
            setattr(datapoint, "patient_metadata", record.to_dict())
        except (AttributeError, TypeError):
            pass
        return datapoint

    def get_moe_sample(self, index: int) -> Dict[str, Any]:
        """Return the unified tensor/label dictionary used by staged MoE training."""
        record = self.records[index]
        datapoint = self._load_sam3_datapoint(index)
        image = datapoint.images[0]
        segments = [obj.segment for obj in image.objects if obj.segment is not None]
        if segments:
            mask_gt = torch.stack([segment.bool() for segment in segments]).any(dim=0)
        else:
            mask_gt = torch.zeros(image.size, dtype=torch.bool)
        boxes = torch.tensor(record.box_prompt or [], dtype=torch.float32).reshape(-1, 4)
        return {
            "image": image.data,
            "mask_gt": mask_gt,
            "box_prompt": boxes,
            "text_prompt": record.text_prompt,
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
