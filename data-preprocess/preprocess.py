#!/usr/bin/env python3
"""One-shot preprocessing for the MoE-SAM3 data that ``train.sh`` loads.

``PatientDataset`` derives modality / area / boundary labels while it is being
constructed, and the area + boundary measurements are the expensive part: they
cost roughly 0.4 s per slice with a small PyTorch thread pool (about 3 s with
the default pool), i.e. hours for the ~23k selected train/val slices.  The
trainer recomputes them on every launch unless ``dataset.label_cache_dir``
already holds the measurements written by ``prepare_slice_labels.py``.

This script produces, ahead of training, every artifact the trainer consumes:

* the ``datasetConfig/*.json`` train/val patient manifests,
* the ``*_sam3.json`` + ``*_boxes_xyxy.json`` pair per modality/split,
* the per-slice area/boundary cache under ``data-preprocess/labels``,
* (only with ``--force``) the train-split area/boundary threshold files,
* a verification pass that fails loudly if anything is missing.

Typical use::

    ./run_preprocess.sh                 # every stage, default config
    ./run_preprocess.sh verify          # coverage report only
    ./run_preprocess.sh labels --workers 64

See ``README.md`` in this directory for the full workflow.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import yaml
except ImportError as error:  # pragma: no cover - environment guidance
    raise SystemExit(
        "PyYAML is required. Activate the training environment first, e.g. "
        "`CONDA_ENV=sam3 ./run_preprocess.sh`, or `conda activate sam3`."
    ) from error


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = REPO_ROOT / "MedSAM3-main"

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "moe_sam3_train_from_scratch.yaml"
DEFAULT_LABEL_CACHE_DIR = SCRIPT_DIR / "labels"

TRANS_FORMAT = REPO_ROOT / "trans_format.py"
PREPARE_SLICE_LABELS = REPO_ROOT / "prepare_slice_labels.py"
COMPUTE_AREA_THRESHOLDS = PROJECT_ROOT / "tools" / "compute_area_thresholds.py"
COMPUTE_BOUNDARY_THRESHOLDS = PROJECT_ROOT / "tools" / "compute_boundary_thresholds.py"
PATIENT_DATASET = PROJECT_ROOT / "data" / "patient_dataset.py"

MODALITIES = ("mr", "us")
SPLITS = ("train", "val", "test")
MANIFEST_SPLITS = ("train", "val")
STAGES = ("patients", "coco", "labels", "thresholds", "verify", "all")

# Same slice naming contract as ``PatientDataset._SLICE_PATTERN``.
_SLICE_PATTERN = re.compile(r"^slice_(\d+)\.png$", re.IGNORECASE)


class PreprocessError(RuntimeError):
    """Raised when a stage cannot produce a required artifact."""


@dataclass(frozen=True)
class Paths:
    """Resolved artifact locations for one config."""

    config: Path
    label_cache_dir: Path
    selected_manifest: Path
    selected_val_manifest: Path
    area_thresholds: Path
    boundary_thresholds: Path


def log(message: str) -> None:
    print(f"[preprocess] {message}", flush=True)


def _load_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_atomic(path: Path, payload: Any, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=indent)
    temporary.replace(path)


def _run(command: Sequence[Any], *, dry_run: bool = False) -> bool:
    printable = " ".join(str(part) for part in command)
    log(f"$ {printable}")
    if dry_run:
        return True
    started = time.time()
    result = subprocess.run([str(part) for part in command], check=False)
    elapsed = time.time() - started
    if result.returncode != 0:
        log(f"FAILED (exit={result.returncode}, {elapsed:.1f}s)")
        return False
    log(f"done in {elapsed:.1f}s")
    return True


# ---------------------------------------------------------------------------
# Config handling
# ---------------------------------------------------------------------------


def load_dataset_config(config_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise PreprocessError(f"Config is not a mapping: {config_path}")
    dataset = config.get("dataset")
    if not isinstance(dataset, dict) or not dataset:
        raise PreprocessError(f"Config has no non-empty 'dataset' section: {config_path}")
    return config, dataset


def resolve_paths(
    dataset: Dict[str, Any],
    config_path: Path,
    *,
    label_cache_dir: Optional[Path] = None,
) -> Paths:
    """Resolve every artifact path from the dataset section of the config."""
    manifest = dataset.get("selected_patients_json")
    val_manifest = dataset.get("selected_val_patients_json")
    area = dataset.get("area_threshold_file")
    boundary = dataset.get("boundary_threshold_file")
    if not manifest or not val_manifest:
        raise PreprocessError(
            "dataset.selected_patients_json and dataset.selected_val_patients_json "
            "must point at the manifests the trainer loads"
        )
    if not area or not boundary:
        raise PreprocessError(
            "dataset.area_threshold_file and dataset.boundary_threshold_file are required"
        )
    cache_root = label_cache_dir or dataset.get("label_cache_dir") or DEFAULT_LABEL_CACHE_DIR
    return Paths(
        config=Path(config_path),
        label_cache_dir=Path(cache_root),
        selected_manifest=Path(manifest),
        selected_val_manifest=Path(val_manifest),
        area_thresholds=Path(area),
        boundary_thresholds=Path(boundary),
    )


def _modality_root(dataset: Dict[str, Any], modality: str) -> Optional[Path]:
    value = dataset.get(f"{modality}_root") or dataset.get(f"{modality}_image_root")
    return Path(value) if value else None


def _modality_mask_root(dataset: Dict[str, Any], modality: str) -> Optional[Path]:
    value = dataset.get(f"{modality}_mask_root")
    return Path(value) if value else None


def _split_name(dataset: Dict[str, Any], split: str) -> str:
    if split == "train":
        return str(dataset.get("train_split", "train"))
    if split == "val":
        return str(dataset.get("val_split", "val"))
    return "test"


def _configured_path(dataset: Dict[str, Any], *keys: str, split: str) -> Optional[Path]:
    for key in keys:
        value = dataset.get(key)
        if value:
            return Path(str(value).format(split=split))
    return None


def initial_coco_json_path(dataset: Dict[str, Any], modality: str, split: str) -> Path:
    """The COCO JSON produced by the upstream 3D -> 2D export step."""
    root = _modality_root(dataset, modality)
    if root is None:
        raise PreprocessError(f"dataset.{modality}_root is required")
    return root / f"{split}.json"


def coco_json_path(dataset: Dict[str, Any], modality: str, split: str) -> Path:
    configured = _configured_path(
        dataset, f"{modality}_{split}_coco_json", f"{modality}_coco_json", split=split
    )
    if configured is not None:
        return configured
    root = _modality_root(dataset, modality)
    if root is None:
        raise PreprocessError(f"dataset.{modality}_root is required")
    return root / f"{split}_sam3.json"


def boxes_json_path(dataset: Dict[str, Any], modality: str, split: str) -> Path:
    configured = _configured_path(
        dataset, f"{modality}_{split}_boxes_json", f"{modality}_boxes_json", split=split
    )
    if configured is not None:
        return configured
    root = _modality_root(dataset, modality)
    if root is None:
        raise PreprocessError(f"dataset.{modality}_root is required")
    return root / f"{split}_boxes_xyxy.json"


# ---------------------------------------------------------------------------
# Patient selection (mirrors PatientDataset._select_patient_ids)
# ---------------------------------------------------------------------------


def scan_patient_ids(split_root: Path) -> List[int]:
    if not Path(split_root).is_dir():
        return []
    return sorted(
        int(child.name)
        for child in Path(split_root).iterdir()
        if child.is_dir() and child.name.isdigit()
    )


def list_slice_names(patient_dir: Path) -> List[str]:
    """Return ``slice_XXXX.png`` names sorted by their numeric index."""
    indexed: List[Tuple[int, str]] = []
    for path in Path(patient_dir).iterdir():
        if not path.is_file():
            continue
        match = _SLICE_PATTERN.match(path.name)
        if match:
            indexed.append((int(match.group(1)), path.name))
    indexed.sort()
    return [name for _, name in indexed]


def select_patient_ids(
    available: Sequence[int],
    num_patients: Optional[int],
    patient_ids: Optional[Sequence[int]],
    sampling_mode: str,
    seed: int,
) -> List[int]:
    """Select patient folders exactly like ``PatientDataset._select_patient_ids``."""
    available = sorted(int(value) for value in available)
    available_set = set(available)
    if patient_ids:
        selected = [int(value) for value in patient_ids]
        if len(selected) != len(set(selected)):
            raise PreprocessError(f"Explicit patient_ids contains duplicates: {selected}")
        missing = sorted(set(selected) - available_set)
        if missing:
            raise PreprocessError(f"Explicit patient IDs do not exist: {missing}")
        return sorted(selected)
    count = len(available) if num_patients is None else int(num_patients)
    if count < 0 or count > len(available):
        raise PreprocessError(
            f"Requested {count} patients, but only {len(available)} are available"
        )
    if count == len(available):
        return list(available)
    mode = str(sampling_mode).lower()
    if mode == "sequential":
        return list(available[:count])
    if mode == "random":
        return sorted(Random(int(seed)).sample(list(available), count))
    raise PreprocessError(
        "patient sampling mode must be 'random' or 'sequential', " f"got {sampling_mode!r}"
    )


def manifest_plan(dataset: Dict[str, Any], split: str) -> Dict[str, str]:
    """Describe the sampling knobs the trainer uses for one manifest split."""
    if split == "train":
        return {
            "num_key": "num_{modality}_patients",
            "ids_key": "patient_ids",
            "sampling_key": "patient_sampling",
            "default_mode": "random",
        }
    return {
        "num_key": "num_{modality}_val_patients",
        "ids_key": "val_patient_ids",
        "sampling_key": "val_patient_sampling",
        "default_mode": "sequential",
    }


def build_manifest(dataset: Dict[str, Any], split: str) -> Dict[str, Any]:
    """Recompute the patient manifest ``save_selected_patients`` would write."""
    plan = manifest_plan(dataset, split)
    split_name = _split_name(dataset, split)
    sampling = dataset.get(plan["sampling_key"]) or {}
    sampling_mode = str(sampling.get("mode", plan["default_mode"]))
    seed = int(sampling.get("seed", 42))
    explicit_all = dataset.get(plan["ids_key"]) or {}
    patient_ids: Dict[str, List[int]] = {}
    slice_counts: Dict[str, int] = {}
    for modality in MODALITIES:
        root = _modality_root(dataset, modality)
        if root is None:
            raise PreprocessError(f"dataset.{modality}_root is required")
        split_root = root / split_name
        available = scan_patient_ids(split_root)
        if not available:
            raise PreprocessError(
                f"No numeric patient folders under {split_root}; "
                f"cannot build the {split} manifest"
            )
        selected = select_patient_ids(
            available,
            dataset.get(plan["num_key"].format(modality=modality)),
            explicit_all.get(modality),
            sampling_mode,
            seed,
        )
        patient_ids[modality] = selected
        slice_counts[modality] = sum(
            len(list_slice_names(split_root / str(patient_id))) for patient_id in selected
        )
    return {
        "sampling_mode": sampling_mode.lower(),
        "seed": seed,
        "resample_patients_each_epoch": False,
        "mr_patient_ids": patient_ids["mr"],
        "us_patient_ids": patient_ids["us"],
        "num_mr_patients": len(patient_ids["mr"]),
        "num_us_patients": len(patient_ids["us"]),
        "num_mr_slices": slice_counts["mr"],
        "num_us_slices": slice_counts["us"],
        "total_slices": slice_counts["mr"] + slice_counts["us"],
    }


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def stage_patients(paths: Paths, dataset: Dict[str, Any], args: argparse.Namespace) -> bool:
    for split, manifest_path in (
        ("train", paths.selected_manifest),
        ("val", paths.selected_val_manifest),
    ):
        manifest = build_manifest(dataset, split)
        log(
            f"{split} patients: MR={manifest['num_mr_patients']} "
            f"({manifest['num_mr_slices']} slices), US={manifest['num_us_patients']} "
            f"({manifest['num_us_slices']} slices) -> {manifest_path}"
        )
        if not args.dry_run:
            _write_json_atomic(manifest_path, manifest)
    return True


def stage_coco(paths: Paths, dataset: Dict[str, Any], args: argparse.Namespace) -> bool:
    ok = True
    for modality in MODALITIES:
        mask_root = _modality_mask_root(dataset, modality)
        for split in args.splits:
            source = initial_coco_json_path(dataset, modality, split)
            if not source.is_file():
                log(f"[skip] {modality}/{split}: no initial COCO at {source}")
                continue
            target = coco_json_path(dataset, modality, split)
            boxes = boxes_json_path(dataset, modality, split)
            if not args.force and target.is_file() and boxes.is_file():
                if (
                    target.stat().st_mtime >= source.stat().st_mtime
                    and boxes.stat().st_mtime >= source.stat().st_mtime
                ):
                    log(f"[skip] {modality}/{split}: {target.name} is up to date")
                    continue
            command: List[Any] = [
                sys.executable,
                TRANS_FORMAT,
                "--input-json",
                source,
                "--output-json",
                target,
                "--boxes-json",
                boxes,
            ]
            if mask_root is not None:
                command += ["--mask-root", mask_root]
            if args.category_name:
                command += ["--category-name", args.category_name]
            ok = _run(command, dry_run=args.dry_run) and ok
    return ok


def stage_labels(paths: Paths, dataset: Dict[str, Any], args: argparse.Namespace) -> bool:
    command: List[Any] = [
        sys.executable,
        PREPARE_SLICE_LABELS,
        "--config",
        paths.config,
        "--output-dir",
        paths.label_cache_dir,
        "--modalities",
        *MODALITIES,
        "--splits",
        *args.splits,
        "--scope",
        "selected",
        "--workers",
        str(args.workers),
    ]
    if args.force:
        command.append("--overwrite")
    if args.limit is not None:
        command += ["--limit", str(args.limit)]
    return _run(command, dry_run=args.dry_run)


def stage_thresholds(paths: Paths, dataset: Dict[str, Any], args: argparse.Namespace) -> bool:
    area_ready = paths.area_thresholds.is_file()
    boundary_ready = paths.boundary_thresholds.is_file()
    if area_ready and boundary_ready and not args.force:
        log(
            "[skip] train-split thresholds already exist "
            f"({paths.area_thresholds.name}, {paths.boundary_thresholds.name}); "
            "use --force to recompute them from the train masks"
        )
        return True
    if not paths.selected_manifest.is_file():
        raise PreprocessError(
            "The train patient manifest is required to compute thresholds; "
            f"run the 'patients' stage first ({paths.selected_manifest})"
        )
    ok = _run(
        [
            sys.executable,
            COMPUTE_AREA_THRESHOLDS,
            "--config",
            paths.config,
            "--selected-patients",
            paths.selected_manifest,
            "--output",
            paths.area_thresholds,
        ],
        dry_run=args.dry_run,
    )
    ok = (
        _run(
            [
                sys.executable,
                COMPUTE_BOUNDARY_THRESHOLDS,
                "--config",
                paths.config,
                "--selected-patients",
                paths.selected_manifest,
                "--output",
                paths.boundary_thresholds,
            ],
            dry_run=args.dry_run,
        )
        and ok
    )
    return ok


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def label_cache_version() -> int:
    """Read ``LABEL_CACHE_VERSION`` from the dataset code that validates it."""
    text = PATIENT_DATASET.read_text(encoding="utf-8")
    match = re.search(r"^LABEL_CACHE_VERSION\s*=\s*(\d+)", text, re.MULTILINE)
    if not match:
        raise PreprocessError(f"Cannot read LABEL_CACHE_VERSION from {PATIENT_DATASET}")
    return int(match.group(1))


def expected_band_width(paths: Paths) -> int:
    try:
        payload = _load_json(paths.boundary_thresholds)
    except (OSError, json.JSONDecodeError) as error:
        raise PreprocessError(
            f"Cannot read {paths.boundary_thresholds}: {type(error).__name__}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise PreprocessError(
            f"Boundary threshold file must be an object: {paths.boundary_thresholds}"
        )
    return int(payload.get("boundary_band_width", 3))


def _check_coco(dataset: Dict[str, Any], split: str, problems: List[str]) -> int:
    images = 0
    for modality in MODALITIES:
        if not initial_coco_json_path(dataset, modality, split).is_file():
            log(f"[skip] {modality}/{split}: no initial COCO on disk")
            continue
        coco_path = coco_json_path(dataset, modality, split)
        boxes_path = boxes_json_path(dataset, modality, split)
        try:
            coco = _load_json(coco_path)
        except (OSError, json.JSONDecodeError) as error:
            problems.append(f"{modality}/{split}: cannot read {coco_path} ({error})")
            continue
        for field in ("images", "annotations", "categories"):
            if not isinstance(coco.get(field), list) or not coco[field]:
                problems.append(
                    f"{modality}/{split}: {coco_path} has no non-empty '{field}'"
                )
        try:
            boxes = _load_json(boxes_path)
        except (OSError, json.JSONDecodeError) as error:
            problems.append(f"{modality}/{split}: cannot read {boxes_path} ({error})")
            continue
        if not isinstance(boxes, dict) or not boxes:
            problems.append(f"{modality}/{split}: {boxes_path} is empty")
        images += len(coco.get("images") or [])
    return images


def _check_thresholds(paths: Paths, problems: List[str]) -> None:
    for path, keys in (
        (paths.area_thresholds, ("small_max", "medium_max")),
        (paths.boundary_thresholds, ("mr", "us", "boundary_band_width")),
    ):
        try:
            payload = _load_json(path)
        except (OSError, json.JSONDecodeError) as error:
            problems.append(f"cannot read threshold file {path} ({error})")
            continue
        missing = [key for key in keys if key not in payload]
        if missing:
            problems.append(f"threshold file {path} is missing {missing}")


def _check_manifests(
    paths: Paths, dataset: Dict[str, Any], problems: List[str]
) -> Dict[str, Dict[str, Any]]:
    manifests: Dict[str, Dict[str, Any]] = {}
    for split, manifest_path in (
        ("train", paths.selected_manifest),
        ("val", paths.selected_val_manifest),
    ):
        try:
            stored = _load_json(manifest_path)
        except (OSError, json.JSONDecodeError) as error:
            problems.append(f"{split}: cannot read manifest {manifest_path} ({error})")
            continue
        try:
            expected = build_manifest(dataset, split)
        except PreprocessError as error:
            problems.append(f"{split}: cannot recompute manifest ({error})")
            continue
        for key in ("mr_patient_ids", "us_patient_ids"):
            if sorted(stored.get(key) or []) != sorted(expected[key]):
                problems.append(
                    f"{split}: manifest patient ids differ from the config selection "
                    f"({key}); re-run the 'patients' stage"
                )
        if int(stored.get("total_slices", -1)) != int(expected["total_slices"]):
            problems.append(
                f"{split}: manifest records {stored.get('total_slices')} slices but the "
                f"config selection now yields {expected['total_slices']}"
            )
        manifests[split] = expected
    return manifests


def _check_label_cache(
    paths: Paths,
    dataset: Dict[str, Any],
    manifests: Dict[str, Dict[str, Any]],
    problems: List[str],
) -> Tuple[int, int]:
    version = label_cache_version()
    band_width = expected_band_width(paths)
    cached_slices = 0
    expected_slices = 0
    for split, manifest in manifests.items():
        split_name = _split_name(dataset, split)
        for modality in MODALITIES:
            cache_dir = paths.label_cache_dir / modality / split_name
            mask_root = _modality_mask_root(dataset, modality)
            mask_split = mask_root / split_name if mask_root else None
            image_root = _modality_root(dataset, modality)
            image_split = image_root / split_name if image_root else None
            if mask_split is not None and not mask_split.is_dir():
                problems.append(
                    f"{modality}/{split_name}: mask directory missing: {mask_split}"
                )
                continue
            for patient_id in manifest[f"{modality}_patient_ids"]:
                names = (
                    list_slice_names(image_split / str(patient_id))
                    if image_split is not None
                    else []
                )
                expected_slices += len(names)
                shard_path = cache_dir / f"{patient_id}.json"
                try:
                    shard = _load_json(shard_path)
                except (OSError, json.JSONDecodeError) as error:
                    problems.append(f"missing/corrupt label shard {shard_path} ({error})")
                    continue
                if int(shard.get("version", -1)) != version:
                    problems.append(
                        f"stale label shard {shard_path} "
                        f"(version={shard.get('version')} != {version})"
                    )
                    continue
                if int(shard.get("boundary_band_width", -1)) != band_width:
                    problems.append(
                        f"stale label shard {shard_path} "
                        f"(band_width={shard.get('boundary_band_width')} != {band_width})"
                    )
                    continue
                slices = shard.get("slices")
                if not isinstance(slices, dict):
                    problems.append(f"label shard {shard_path} has no 'slices' mapping")
                    continue
                absent = [name for name in names if name not in slices]
                if absent:
                    problems.append(
                        f"label shard {shard_path} misses {len(absent)}/{len(names)} slices"
                    )
                    continue
                cached_slices += len(names)
    return cached_slices, expected_slices


def verify(
    paths: Paths,
    dataset: Dict[str, Any],
    *,
    splits: Sequence[str] = SPLITS,
    report_limit: int = 20,
) -> List[str]:
    """Return the list of problems that would break ``train.sh``."""
    problems: List[str] = []
    images = 0
    for split in splits:
        images += _check_coco(dataset, split, problems)
    _check_thresholds(paths, problems)
    manifests = _check_manifests(paths, dataset, problems)
    cached = expected = 0
    if manifests:
        cached, expected = _check_label_cache(paths, dataset, manifests, problems)
    log(f"COCO images indexed        : {images}")
    log(f"label cache dir            : {paths.label_cache_dir}")
    log(f"cached train/val slices    : {cached}/{expected}")
    if problems:
        log(f"problems found             : {len(problems)}")
        for item in problems[:report_limit]:
            log(f"  - {item}")
        if len(problems) > report_limit:
            log(f"  ... and {len(problems) - report_limit} more")
    else:
        log("all required artifacts are present and complete")
    return problems


def stage_verify(paths: Paths, dataset: Dict[str, Any], args: argparse.Namespace) -> bool:
    return not verify(paths, dataset, splits=tuple(args.splits) + ("test",))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the train/val artifacts train.sh loads, so training does not "
            "preprocess the dataset on every launch"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "stage",
        nargs="?",
        choices=STAGES,
        default="all",
        help="stage to run; 'all' runs patients -> coco -> labels -> thresholds -> verify",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=["train", "val"],
        help="splits handled by the coco/labels stages",
    )
    parser.add_argument(
        "--label-cache-dir",
        type=Path,
        default=None,
        help="override dataset.label_cache_dir",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, os.cpu_count() or 8),
        help="parallel workers for the per-slice measurement stage",
    )
    parser.add_argument(
        "--category-name",
        default=None,
        help="foreground category name for trans_format.py; default keeps the input name",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only measure the first N label shards (smoke runs)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute artifacts that already exist (including thresholds)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the planned commands without running them",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")
    if not (PROJECT_ROOT / "data").is_dir():
        raise SystemExit(f"MedSAM3-main project not found under {REPO_ROOT}")
    _, dataset = load_dataset_config(config_path)
    paths = resolve_paths(dataset, config_path, label_cache_dir=args.label_cache_dir)
    log(f"config      : {paths.config}")
    log(f"label cache : {paths.label_cache_dir}")

    stages = {
        "patients": stage_patients,
        "coco": stage_coco,
        "labels": stage_labels,
        "thresholds": stage_thresholds,
        "verify": stage_verify,
    }
    order = (
        ["patients", "coco", "labels", "thresholds", "verify"]
        if args.stage == "all"
        else [args.stage]
    )
    started = time.time()
    for name in order:
        log(f"=== stage: {name} ===")
        try:
            ok = stages[name](paths, dataset, args)
        except PreprocessError as error:
            log(f"stage '{name}' failed: {error}")
            return 1
        if not ok:
            log(f"stage '{name}' failed")
            return 1
    log(f"finished in {time.time() - started:.1f}s")
    if args.stage == "all":
        log(
            "next: train as usual (bash train.sh); the trainer reports "
            "'area+boundary labels: N/N from cache'"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
