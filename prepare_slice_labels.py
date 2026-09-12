#!/usr/bin/env python3
"""Cache the per-slice area/boundary measurements used by the MoE-SAM3 datasets.

``PatientDataset.__init__`` derives area and boundary pseudo-labels for every
selected slice.  Those measurements are deterministic functions of the image
and mask files, but the training code recomputes all of them on every launch,
and the cost is ~3 s per slice because each 500x500 ``max_pool2d`` call pays
the full synchronisation cost of PyTorch's default (very large) thread pool.

This script computes the measurements once, in parallel, single-threaded per
worker, and stores them under ``data-preprocess/labels``::

    <output-dir>/<modality>/<split>/<patient_id>.json

Only raw measurements are cached (``area_ratio``, ``boundary_contrast``,
``boundary_complexity``, ``boundary_fallback``).  ``area_label`` and
``boundary_label`` stay derived from ``datasetConfig/*.json`` at load time, so
changing the threshold files does not invalidate the cache.

Point the training config at the cache with::

    dataset:
      label_cache_dir: /path/to/sam3-finetune/data-preprocess/labels

Usage::

    python prepare_slice_labels.py                        # train+val, selected patients
    python prepare_slice_labels.py --scope all --workers 64
    python prepare_slice_labels.py --verify               # coverage report only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Every worker must stay single threaded: the measurements are tiny tensors and
# a large intra-op pool makes each call roughly an order of magnitude slower.
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR / "MedSAM3-main"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

from data.area_labels import compute_area_ratio  # noqa: E402
from data.boundary_labels import (  # noqa: E402
    compute_boundary_scores,
    load_boundary_thresholds,
)
from data.patient_dataset import (  # noqa: E402
    LABEL_CACHE_VERSION,
    PatientDataset,
    _SLICE_PATTERN,
)
from data.sample_index import resolve_sample_path  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "moe_sam3_train_from_scratch.yaml"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data-preprocess" / "labels"
MODALITIES = ("mr", "us")
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class PatientTask:
    """One patient shard to measure."""

    modality: str
    split: str
    patient_id: int
    image_split_root: str
    mask_split_root: Optional[str]
    boundary_band_width: int
    output_path: str
    overwrite: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute per-slice area/boundary measurements for MoE-SAM3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--modalities", nargs="+", choices=list(MODALITIES), default=list(MODALITIES)
    )
    parser.add_argument("--splits", nargs="+", choices=list(SPLITS), default=["train", "val"])
    parser.add_argument(
        "--scope",
        choices=("selected", "all"),
        default="selected",
        help="'selected' caches exactly the patients the config would use, "
        "'all' caches every patient folder on disk",
    )
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    parser.add_argument(
        "--overwrite", action="store_true", help="Recompute shards that already exist"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only process the first N shards"
    )
    parser.add_argument(
        "--verify", action="store_true", help="Only report cache coverage, measure nothing"
    )
    return parser.parse_args()


def _init_worker() -> None:
    torch.set_num_threads(1)


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config is not a mapping: {path}")
    dataset = config.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError(f"Config has no 'dataset' section: {path}")
    return dataset


def _boundary_band_width(dataset: Dict[str, Any]) -> int:
    path = dataset.get("boundary_threshold_file")
    if not path:
        return 3
    thresholds = load_boundary_thresholds(path)
    return int(thresholds.get("boundary_band_width", 3))


def _split_root(root: str | Path, split: str) -> Path:
    return Path(root) / split


def _modality_roots(dataset: Dict[str, Any], modality: str) -> Tuple[Path, Optional[Path]]:
    image_root = dataset.get(f"{modality}_root") or dataset.get(f"{modality}_image_root")
    if not image_root:
        raise ValueError(f"Config has no {modality}_root")
    mask_root = dataset.get(f"{modality}_mask_root")
    return Path(image_root), (Path(mask_root) if mask_root else None)


def _list_slices(image_split_root: Path, patient_id: int) -> List[str]:
    patient_dir = image_split_root / str(patient_id)
    if not patient_dir.is_dir():
        raise FileNotFoundError(f"Patient image directory not found: {patient_dir}")
    indexed: List[Tuple[int, str]] = []
    for path in patient_dir.iterdir():
        if not path.is_file():
            continue
        match = _SLICE_PATTERN.match(path.name)
        if match:
            indexed.append((int(match.group(1)), path.name))
    indexed.sort()
    return [name for _, name in indexed]


def _dataset_config_patients(
    dataset: Dict[str, Any], modality: str, split: str, available: Sequence[int]
) -> List[int]:
    train_split = str(dataset.get("train_split", "train"))
    val_split = str(dataset.get("val_split", "val"))
    if split == train_split:
        sampling = dataset.get("patient_sampling") or {}
        explicit = (dataset.get("patient_ids") or {}).get(modality) or []
        num_patients = dataset.get(f"num_{modality}_patients")
        default_mode = "random"
    elif split == val_split:
        sampling = dataset.get("val_patient_sampling") or {}
        explicit = (dataset.get("val_patient_ids") or {}).get(modality) or []
        num_patients = dataset.get(f"num_{modality}_val_patients")
        default_mode = "sequential"
    else:
        return list(available)
    return PatientDataset._select_patient_ids(
        available,
        num_patients=num_patients,
        patient_ids=explicit,
        sampling_mode=sampling.get("mode", default_mode),
        seed=int(sampling.get("seed", 42)),
    )


def _measure_slice(
    image_path: Path, mask_path: Optional[Path], band_width: int
) -> Dict[str, Any]:
    """Reproduce ``_expand_and_match_slices`` measurements for one slice."""
    entry: Dict[str, Any] = {
        "mask_verified": False,
        "area_ratio": 0.0,
        "boundary_contrast": 0.0,
        "boundary_complexity": 1.0,
        "boundary_fallback": False,
        "boundary_fallback_reason": None,
    }
    if mask_path is None:
        return entry
    with PILImage.open(mask_path) as mask_image:
        mask_image.verify()
    entry["mask_verified"] = True
    entry["area_ratio"] = float(compute_area_ratio(mask_path))
    scores = compute_boundary_scores(image_path, mask_path, band_width)
    entry["boundary_contrast"] = float(scores.contrast)
    entry["boundary_complexity"] = float(scores.complexity)
    entry["boundary_fallback"] = bool(scores.used_fallback)
    entry["boundary_fallback_reason"] = scores.fallback_reason
    return entry


def _measure_patient(task: PatientTask) -> Tuple[Dict[str, Any], List[str]]:
    image_split_root = Path(task.image_split_root)
    mask_split_root = Path(task.mask_split_root) if task.mask_split_root else None
    entries: Dict[str, Any] = {}
    skipped: List[str] = []
    for name in _list_slices(image_split_root, task.patient_id):
        relative_name = f"{task.patient_id}/{name}"
        resolved_image = resolve_sample_path(
            image_split_root, relative_name, task.split, strict=False
        )
        mask_path = None
        if mask_split_root is not None:
            mask_path = resolve_sample_path(
                mask_split_root, relative_name, task.split, strict=False
            )
        if resolved_image is None or (mask_split_root is not None and mask_path is None):
            skipped.append(relative_name)
            continue
        try:
            entries[name] = _measure_slice(
                resolved_image, mask_path, task.boundary_band_width
            )
        except Exception as error:  # noqa: BLE001 - recorded for the loader to replay
            entries[name] = {"error": f"{type(error).__name__}: {error}"}
    return entries, skipped


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / (path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    os.replace(temp_path, path)


def _process_patient(task: PatientTask) -> Dict[str, Any]:
    output_path = Path(task.output_path)
    result: Dict[str, Any] = {
        "modality": task.modality,
        "split": task.split,
        "patient_id": task.patient_id,
    }
    if output_path.is_file() and not task.overwrite:
        result.update(status="skip", num_slices=None)
        return result
    try:
        entries, skipped = _measure_patient(task)
    except Exception as error:  # noqa: BLE001 - surfaced in the run summary
        result.update(status="error", error=f"{type(error).__name__}: {error}")
        return result
    _write_json_atomic(
        output_path,
        {
            "version": LABEL_CACHE_VERSION,
            "modality": task.modality,
            "split": task.split,
            "patient_id": task.patient_id,
            "boundary_band_width": task.boundary_band_width,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "num_slices": len(entries),
            "skipped_slices": skipped,
            "slices": entries,
        },
    )
    result.update(status="done", num_slices=len(entries), skipped=len(skipped))
    return result


def _read_shard(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _build_tasks(args: argparse.Namespace, dataset: Dict[str, Any]) -> List[PatientTask]:
    band_width = _boundary_band_width(dataset)
    tasks: List[PatientTask] = []
    for modality in args.modalities:
        image_root, mask_root = _modality_roots(dataset, modality)
        for split in args.splits:
            image_split_root = _split_root(image_root, split)
            if not image_split_root.is_dir():
                print(f"[skip] {modality}/{split}: no such directory {image_split_root}")
                continue
            available = sorted(
                int(path.name)
                for path in image_split_root.iterdir()
                if path.is_dir() and path.name.isdigit()
            )
            if not available:
                print(f"[skip] {modality}/{split}: no numeric patient folders")
                continue
            if args.scope == "all":
                patients = available
            else:
                patients = _dataset_config_patients(dataset, modality, split, available)
            mask_split_root = (
                _split_root(mask_root, split) if mask_root is not None else None
            )
            if mask_split_root is not None and not mask_split_root.is_dir():
                raise FileNotFoundError(f"Mask split directory not found: {mask_split_root}")
            print(
                f"[plan] {modality}/{split}: {len(patients)}/{len(available)} patients "
                f"({args.scope}), band_width={band_width}"
            )
            for patient_id in patients:
                tasks.append(
                    PatientTask(
                        modality=modality,
                        split=split,
                        patient_id=patient_id,
                        image_split_root=str(image_split_root),
                        mask_split_root=(
                            str(mask_split_root) if mask_split_root is not None else None
                        ),
                        boundary_band_width=band_width,
                        output_path=str(
                            args.output_dir / modality / split / f"{patient_id}.json"
                        ),
                        overwrite=args.overwrite,
                    )
                )
    if not tasks:
        raise RuntimeError("No patients to process")
    return tasks


def _verify(args: argparse.Namespace, tasks: Sequence[PatientTask]) -> int:
    missing: List[str] = []
    stale: List[str] = []
    incomplete: List[str] = []
    covered = 0
    expected = 0
    for task in tasks:
        path = Path(task.output_path)
        names = _list_slices(Path(task.image_split_root), task.patient_id)
        expected += len(names)
        payload = _read_shard(path) if path.is_file() else None
        if payload is None:
            missing.append(str(path))
            continue
        if int(payload.get("version", -1)) != LABEL_CACHE_VERSION:
            stale.append(f"{path} (version={payload.get('version')})")
            continue
        if int(payload.get("boundary_band_width", -1)) != task.boundary_band_width:
            stale.append(
                f"{path} (band_width={payload.get('boundary_band_width')} "
                f"!= {task.boundary_band_width})"
            )
            continue
        slices = payload.get("slices") or {}
        absent = [name for name in names if name not in slices]
        if absent:
            incomplete.append(f"{path} (missing {len(absent)}/{len(names)} slices)")
            continue
        covered += len(names)
    print(f"\nshards expected : {len(tasks)}")
    print(f"slices expected : {expected}")
    print(f"slices cached   : {covered}")
    for label, items in (
        ("missing shards", missing),
        ("stale shards", stale),
        ("incomplete shards", incomplete),
    ):
        if items:
            print(f"{label}: {len(items)}")
            for item in items[:10]:
                print(f"  - {item}")
    return 0 if covered == expected and not (missing or stale or incomplete) else 1


def _run(args: argparse.Namespace, tasks: Sequence[PatientTask]) -> int:
    pending = sum(1 for task in tasks if args.overwrite or not Path(task.output_path).is_file())
    print(
        f"\n{len(tasks)} shards ({pending} to compute), "
        f"{args.workers} workers, output -> {args.output_dir}"
    )
    if pending == 0:
        print("Nothing to do; use --overwrite to recompute.")
        return 0

    started = time.time()
    done = skipped = failed = slices = 0
    errors: List[str] = []
    try:
        with ProcessPoolExecutor(
            max_workers=args.workers, initializer=_init_worker
        ) as executor:
            futures = [executor.submit(_process_patient, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                status = result["status"]
                if status == "done":
                    done += 1
                    slices += int(result.get("num_slices") or 0)
                elif status == "skip":
                    skipped += 1
                else:
                    failed += 1
                    errors.append(
                        f"{result['modality']}/{result['split']}/{result['patient_id']}: "
                        f"{result.get('error')}"
                    )
                completed = done + skipped + failed
                if completed % 25 == 0 or completed == len(tasks):
                    elapsed = time.time() - started
                    rate = completed / elapsed if elapsed > 0 else 0.0
                    remaining = (pending - done - failed) / rate if rate > 0 else 0.0
                    print(
                        f"  {completed}/{len(tasks)} shards | {slices} slices | "
                        f"{elapsed:.0f}s elapsed | eta {remaining:.0f}s",
                        flush=True,
                    )
    except KeyboardInterrupt:
        print("\nInterrupted; completed shards are already on disk and will be skipped next run.")
        return 130

    elapsed = time.time() - started
    print(
        f"\ndone: {done} computed, {skipped} already cached, {failed} failed, "
        f"{slices} slices measured in {elapsed:.1f}s"
    )
    for message in errors[:20]:
        print(f"  ! {message}")
    if failed:
        print("Re-run the script to retry the failed shards.")
        return 1
    print(
        "\nNext: set dataset.label_cache_dir in the training YAML to\n"
        f"  {args.output_dir}"
    )
    return 0


def main() -> int:
    args = parse_args()
    dataset = _load_config(args.config)
    tasks = _build_tasks(args, dataset)
    if args.verify:
        return _verify(args, tasks)
    return _run(args, tasks)


if __name__ == "__main__":
    raise SystemExit(main())
