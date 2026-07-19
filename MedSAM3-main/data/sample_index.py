"""Canonical keys and strict path/index validation for medical SAM3 samples."""

from __future__ import annotations

import json
import warnings
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Sequence


_SPLIT_NAMES = {"train", "val", "valid", "validation", "test"}


def _portable_file_name(file_name: str) -> str:
    return str(file_name).strip().replace("\\", "/").lstrip("/")


def normalize_sample_key(file_name: str) -> str:
    """Return ``patient_id/slice_name`` for either supported JSON key form."""
    normalized = _portable_file_name(file_name)
    parts = PurePosixPath(normalized).parts
    if parts and parts[0].lower() in _SPLIT_NAMES:
        parts = parts[1:]
    if len(parts) != 2 or not parts[0].isdigit():
        raise ValueError(
            "Sample key must be patient_id/slice_name or "
            f"split/patient_id/slice_name, got {file_name!r}"
        )
    return PurePosixPath(*parts).as_posix()


def sample_key_aliases(file_name: str, split: str) -> List[str]:
    canonical = normalize_sample_key(file_name)
    return [canonical, f"{split}/{canonical}"]


def resolve_sample_path(
    root: str | Path,
    file_name: str,
    split: str,
    strict: bool = True,
    diagnostics: Optional[List[str]] = None,
) -> Optional[Path]:
    """Resolve a sample with the required three-attempt fallback order.

    ``root`` is normally the split directory, e.g. ``mr-2d/train``. Attempts:
    ``root/file_name``; ``root/strip_split(file_name)``;
    ``root.parent/file_name``.
    """
    root_path = Path(root)
    raw_name = _portable_file_name(file_name)
    canonical = normalize_sample_key(raw_name)
    candidates = [
        root_path / Path(raw_name),
        root_path / Path(canonical),
        root_path.parent / Path(raw_name),
    ]
    checked: List[Path] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in checked:
            continue
        checked.append(candidate)
        if candidate.is_file():
            return candidate
    message = (
        f"Missing sample for split={split!r}, file_name={file_name!r}; checked: "
        + ", ".join(str(path) for path in checked)
    )
    if diagnostics is not None:
        diagnostics.append(message)
    if strict:
        raise FileNotFoundError(message)
    warnings.warn(message + "; sample will be skipped", RuntimeWarning, stacklevel=2)
    return None


def load_boxes_index(
    boxes_json: str | Path,
    strict: bool = True,
    diagnostics: Optional[List[str]] = None,
) -> Dict[str, List[List[float]]]:
    """Load SAM3 XYXY boxes and collapse both supported key forms."""
    path = Path(boxes_json)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"boxes JSON must contain an object, got {type(payload)}")
    index: Dict[str, List[List[float]]] = {}
    for source_key, raw_boxes in payload.items():
        try:
            key = normalize_sample_key(source_key)
            if not isinstance(raw_boxes, list) or not raw_boxes:
                raise ValueError("box list is empty or not a list")
            boxes: List[List[float]] = []
            for raw_box in raw_boxes:
                if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
                    raise ValueError(f"invalid XYXY box: {raw_box!r}")
                box = [float(value) for value in raw_box]
                if box[2] <= box[0] or box[3] <= box[1]:
                    raise ValueError(f"non-positive XYXY box: {box!r}")
                boxes.append(box)
            if key in index and index[key] != boxes:
                raise ValueError(
                    f"conflicting aliases for {key!r}: {index[key]!r} vs {boxes!r}"
                )
            index[key] = boxes
        except (TypeError, ValueError) as error:
            message = f"Invalid boxes entry {source_key!r}: {error}"
            if diagnostics is not None:
                diagnostics.append(message)
            if strict:
                raise ValueError(message) from error
            warnings.warn(message + "; entry will be skipped", RuntimeWarning, stacklevel=2)
    return index


def validate_final_coco(
    coco_data: Dict[str, Any],
    independent_masks_available: bool,
) -> None:
    """Validate fields required by the native SAM3 trainer before indexing."""
    images = coco_data.get("images")
    annotations = coco_data.get("annotations")
    categories = coco_data.get("categories")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("Final COCO JSON requires list fields images and annotations")
    if not isinstance(categories, list) or not categories:
        raise ValueError("Final COCO JSON requires a non-empty categories field")
    image_ids = set()
    for image in images:
        for field in ("id", "file_name", "width", "height"):
            if field not in image:
                raise ValueError(f"COCO image is missing {field!r}: {image!r}")
        normalize_sample_key(image["file_name"])
        image_ids.add(int(image["id"]))
    for annotation in annotations:
        for field in ("id", "image_id", "bbox", "area", "category_id"):
            if field not in annotation:
                raise ValueError(f"COCO annotation is missing {field!r}: {annotation!r}")
        if int(annotation["image_id"]) not in image_ids:
            raise ValueError(
                f"annotation.image_id={annotation['image_id']} has no matching image"
            )
        bbox = annotation["bbox"]
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(f"Invalid COCO XYWH bbox: {bbox!r}")
        if not annotation.get("segmentation") and not independent_masks_available:
            raise ValueError(
                "annotation.segmentation is empty and no independent mask_root was provided"
            )

