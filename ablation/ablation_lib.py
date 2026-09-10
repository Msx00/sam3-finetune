#!/usr/bin/env python3
"""Shared utilities for reproducible HMoE-SAM3 ablation experiments.

The training code consumes a fully resolved YAML file.  This module adds a
small, explicit inheritance layer on top of it and protects every dataset and
checkpoint path while producing one independent configuration per method and
seed.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import random
import re
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import yaml


SCHEMA_VERSION = 1
PATH_KEY_TOKENS = ("root", "checkpoint", "json", "file", "path", "dir", "resume")
PROTECTED_SECTIONS = ("dataset", "model", "svanet")
MUTABLE_OUTPUT_PATHS = {
    "training.save_dir", "output.output_dir", "wandb.wandb_dir",
}


class StudyError(ValueError):
    """Raised when a study definition violates the experiment contract."""


@dataclass(frozen=True)
class RunSpec:
    experiment_id: str
    seed: int
    config_path: Path
    output_dir: Path
    description: str
    comparison: str

    @property
    def run_id(self) -> str:
        return f"{self.experiment_id}__seed_{self.seed}"


def load_yaml(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise StudyError(f"YAML root must be a mapping: {path}")
    return value


def dump_yaml(value: Mapping[str, Any], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            dict(value), handle, allow_unicode=True, sort_keys=False, width=100,
        )


def deep_merge(
    base: Mapping[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a recursive mapping merge; lists/scalars are replaced atomically."""
    result: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def flatten_mapping(
    value: Mapping[str, Any], prefix: str = ""
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            flattened.update(flatten_mapping(item, name))
        else:
            flattened[name] = item
    return flattened


def _is_path_key(parts: Sequence[str]) -> bool:
    if not parts or ".".join(parts) in MUTABLE_OUTPUT_PATHS:
        return False
    key = parts[-1].lower()
    looks_like_path = any(token in key for token in PATH_KEY_TOKENS)
    return looks_like_path and (
        parts[0] in PROTECTED_SECTIONS
        or parts[0] == "training"
        or "checkpoint" in key
        or "resume" in key
    )


def protected_paths(config: Mapping[str, Any]) -> dict[str, Any]:
    """Collect data/model dependency paths that an ablation may never mutate."""
    protected: dict[str, Any] = {}

    def visit(value: Any, parts: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(item, parts + (str(key),))
            return
        if _is_path_key(parts):
            protected[".".join(parts)] = copy.deepcopy(value)

    visit(config, ())
    return protected


def assert_protected_paths_unchanged(
    base: Mapping[str, Any], candidate: Mapping[str, Any]
) -> None:
    before = protected_paths(base)
    after = protected_paths(candidate)
    errors = []
    for key in sorted(set(before) | set(after)):
        if before.get(key) != after.get(key):
            errors.append(f"{key}: {before.get(key)!r} -> {after.get(key)!r}")
    if errors:
        joined = "\n  ".join(errors)
        raise StudyError(
            "Ablation overrides changed protected data/checkpoint paths:\n  " + joined
        )


def validate_method_config(config: Mapping[str, Any], label: str = "config") -> None:
    """Validate cross-field assumptions needed for interpretable ablations."""
    moe = dict(config.get("moe") or {})
    if moe.get("confidence_routing", False):
        low = float(moe.get("confidence_low_threshold", 0.0))
        high = float(moe.get("confidence_high_threshold", 1.0))
        if not 0.0 <= low <= high <= 1.0:
            raise StudyError(
                f"{label}: require 0 <= confidence_low_threshold <= "
                "confidence_high_threshold <= 1"
            )
        if int(moe.get("confidence_top_k", 1)) < 1:
            raise StudyError(f"{label}: confidence_top_k must be positive")
    regularizer = str(moe.get("router_regularizer", "none"))
    if regularizer not in {"none", "uniform", "batch_prior"}:
        raise StudyError(f"{label}: unsupported router_regularizer={regularizer!r}")
    routing_source = str(moe.get("routing_feature_source", "backbone_fpn"))
    if routing_source not in {"backbone_fpn", "decoder_memory"}:
        raise StudyError(
            f"{label}: routing_feature_source must be 'backbone_fpn' or "
            "'decoder_memory'"
        )
    if int(moe.get("rank", 1)) < 1 or float(moe.get("alpha", 1.0)) <= 0.0:
        raise StudyError(f"{label}: MoE rank and alpha must be positive")
    layers = moe.get("decoder_layers", config.get("model", {}).get("moe_decoder_layers", []))
    if layers and any(int(layer) < 1 for layer in layers):
        raise StudyError(f"{label}: decoder layer ids are 1-based positive integers")
    source_layer = moe.get(
        "router_source_layer", config.get("router", {}).get("router_source_layer")
    )
    if layers and source_layer is not None and not 1 <= int(source_layer) < min(map(int, layers)):
        raise StudyError(
            f"{label}: router_source_layer must precede every routed decoder layer"
        )
    initial_scale = float(moe.get("residual_scale_init", 0.0))
    maximum_scale = float(moe.get("residual_scale_max", 1.0))
    if maximum_scale <= 0.0 or not 0.0 <= initial_scale <= maximum_scale:
        raise StudyError(
            f"{label}: residual_scale_init must be within [0, residual_scale_max]"
        )

    curriculum = config.get("dataset", {}).get("prompt_curriculum", {})
    if isinstance(curriculum, Mapping) and curriculum.get("enabled", False):
        expected_modes = {"image_only", "text", "coarse_box", "accurate_box"}
        for schedule_name in ("start_probabilities", "end_probabilities"):
            probabilities = curriculum.get(schedule_name)
            if not isinstance(probabilities, Mapping):
                raise StudyError(f"{label}: missing prompt {schedule_name}")
            if set(probabilities) != expected_modes:
                raise StudyError(
                    f"{label}: {schedule_name} modes must be {sorted(expected_modes)}"
                )
            values = [float(value) for value in probabilities.values()]
            if any(value < 0.0 for value in values) or not math.isclose(
                sum(values), 1.0, abs_tol=1e-8
            ):
                raise StudyError(f"{label}: {schedule_name} must be nonnegative and sum to 1")
        if curriculum.get("evaluation_mode") != "image_only":
            raise StudyError(
                f"{label}: formal ablations require evaluation_mode=image_only"
            )

    teacher = config.get("router", {}).get("teacher_forcing", {})
    if isinstance(teacher, Mapping) and teacher.get("enabled", False):
        start = float(teacher.get("start_ratio", 0.0))
        end = float(teacher.get("end_ratio", 0.0))
        if not 0.0 <= start <= 1.0 or not 0.0 <= end <= 1.0:
            raise StudyError(f"{label}: teacher-forcing ratios must be within [0, 1]")

    svanet = config.get("svanet", {})
    if isinstance(svanet, Mapping) and svanet.get("enable", False):
        for key in (
            "fusion_weight", "locator_threshold", "locator_min_confidence",
            "sam3_min_confidence", "min_area_confidence", "max_component_fraction",
        ):
            if key in svanet and not 0.0 <= float(svanet[key]) <= 1.0:
                raise StudyError(f"{label}: svanet.{key} must be within [0, 1]")


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_identity(path: Path) -> dict[str, Any]:
    """Return a best-effort code revision marker without requiring Git."""
    try:
        revision = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=True, capture_output=True, text=True,
        ).stdout
        return {"git_commit": revision, "git_dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}


def resolve_path(path: str | Path, relative_to: Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else (relative_to / path).resolve()


def validate_study(study: Mapping[str, Any], study_path: Path) -> None:
    if int(study.get("schema_version", -1)) != SCHEMA_VERSION:
        raise StudyError(
            f"Unsupported schema_version={study.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    for key in ("base_config", "reference", "experiments", "suites", "seeds"):
        if key not in study:
            raise StudyError(f"Missing required study key: {key}")
    base_path = resolve_path(str(study["base_config"]), study_path.parent)
    if not base_path.is_file():
        raise StudyError(f"Base config does not exist: {base_path}")
    reference = study["reference"]
    if not isinstance(reference, Mapping) or not reference.get("id"):
        raise StudyError("reference must be a mapping with a non-empty id")
    experiments = study["experiments"]
    if not isinstance(experiments, Mapping):
        raise StudyError("experiments must be a mapping")
    all_ids = {str(reference["id"]), *map(str, experiments)}
    if len(all_ids) != len(experiments) + 1:
        raise StudyError("Reference id must not also appear under experiments")
    for suite_name, members in study["suites"].items():
        if not isinstance(members, list) or not members:
            raise StudyError(f"Suite {suite_name!r} must be a non-empty list")
        unknown = set(map(str, members)) - all_ids - {"*"}
        if unknown:
            raise StudyError(
                f"Suite {suite_name!r} contains unknown experiments: {sorted(unknown)}"
            )
    seeds = study["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise StudyError("seeds must be a non-empty list")
    if len({int(seed) for seed in seeds}) != len(seeds):
        raise StudyError("seeds must be unique integers")


def load_study(path: Path | str) -> tuple[dict[str, Any], Path]:
    study_path = Path(path).expanduser().resolve()
    study = load_yaml(study_path)
    validate_study(study, study_path)
    return study, study_path


def experiment_table(study: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    reference = copy.deepcopy(dict(study["reference"]))
    table = {str(reference["id"]): reference}
    reference_overrides = dict(reference.get("overrides") or {})
    for experiment_id, raw in study["experiments"].items():
        entry = copy.deepcopy(dict(raw or {}))
        inherit = bool(entry.pop("inherits_reference", True))
        overrides = dict(entry.get("overrides") or {})
        entry["overrides"] = (
            deep_merge(reference_overrides, overrides) if inherit else overrides
        )
        entry["id"] = str(experiment_id)
        table[str(experiment_id)] = entry
    return table


def select_experiment_ids(
    study: Mapping[str, Any], suite: str, include: str | None = None
) -> list[str]:
    if suite not in study["suites"]:
        raise StudyError(
            f"Unknown suite {suite!r}; choices={sorted(study['suites'])}"
        )
    table = experiment_table(study)
    members = list(map(str, study["suites"][suite]))
    selected = list(table) if "*" in members else members
    if include:
        pattern = re.compile(include)
        selected = [name for name in selected if pattern.search(name)]
    if not selected:
        raise StudyError("Experiment selection is empty")
    return selected


def _default_output_root(base: Mapping[str, Any], study: Mapping[str, Any]) -> Path:
    output = str(base.get("output", {}).get("output_dir", "outputs/reference"))
    output_path = Path(output)
    name = str(study.get("output_name", "ablation_hmoe_v2"))
    return output_path.parent / name


def _configure_seed(config: MutableMapping[str, Any], seed: int) -> None:
    training = config.setdefault("training", {})
    training["seed"] = int(seed)
    training["data_order_seed"] = int(seed)
    training.setdefault("deterministic", True)
    training.setdefault("cudnn_benchmark", False)
    prompt = config.setdefault("dataset", {}).get("prompt_curriculum")
    if isinstance(prompt, MutableMapping):
        prompt["seed"] = int(seed)


def _configure_output(
    config: MutableMapping[str, Any], output_dir: Path, experiment_id: str, seed: int
) -> None:
    output_text = str(output_dir)
    config.setdefault("training", {})["save_dir"] = output_text
    config.setdefault("output", {})["output_dir"] = output_text
    wandb = config.setdefault("wandb", {})
    wandb["wandb_name"] = f"{experiment_id}-s{seed}"
    wandb["wandb_group"] = "hmoe-v2-ablation"
    wandb["wandb_dir"] = str(output_dir / "wandb")
    wandb["wandb_run_id"] = f"abl-{experiment_id}-s{seed}"
    tags = list(wandb.get("wandb_tags") or [])
    wandb["wandb_tags"] = list(dict.fromkeys(tags + ["ablation", experiment_id, f"seed-{seed}"]))


def generate_run_specs(
    study: Mapping[str, Any],
    study_path: Path,
    suite: str,
    seeds: Sequence[int] | None = None,
    include: str | None = None,
    generated_root: Path | None = None,
    output_root: Path | None = None,
    write: bool = True,
) -> list[RunSpec]:
    base_path = resolve_path(str(study["base_config"]), study_path.parent)
    base = load_yaml(base_path)
    table = experiment_table(study)
    selected = select_experiment_ids(study, suite, include)
    selected_seeds = [int(seed) for seed in (seeds or study["seeds"])]
    if len(set(selected_seeds)) != len(selected_seeds):
        raise StudyError("Requested seeds must be unique")
    generated_root = generated_root or (study_path.parent / "generated")
    output_root = output_root or _default_output_root(base, study)
    specs: list[RunSpec] = []
    manifest_runs = []
    for experiment_id in selected:
        experiment = table[experiment_id]
        resolved_method = deep_merge(base, experiment.get("overrides") or {})
        assert_protected_paths_unchanged(base, resolved_method)
        validate_method_config(resolved_method, label=experiment_id)
        for seed in selected_seeds:
            config = copy.deepcopy(resolved_method)
            _configure_seed(config, seed)
            # Include the suite to prevent a smoke/routing run from overwriting
            # the same experiment id in a formal primary study.
            output_dir = output_root / suite / experiment_id / f"seed_{seed}"
            _configure_output(config, output_dir, experiment_id, seed)
            assert_protected_paths_unchanged(base, config)
            config_path = generated_root / suite / experiment_id / f"seed_{seed}.yaml"
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "study": study_path.name,
                "suite": suite,
                "experiment_id": experiment_id,
                "seed": seed,
                "reference_id": str(study["reference"]["id"]),
                "description": str(experiment.get("description", "")),
                "comparison": str(experiment.get("comparison", "")),
                "base_config": str(base_path),
                "base_config_sha256": stable_hash(base),
                "method_overrides_sha256": stable_hash(experiment.get("overrides") or {}),
                "output_dir": str(output_dir),
            }
            config["ablation"] = metadata
            if write:
                dump_yaml(config, config_path)
            spec = RunSpec(
                experiment_id=experiment_id,
                seed=seed,
                config_path=config_path,
                output_dir=output_dir,
                description=metadata["description"],
                comparison=metadata["comparison"],
            )
            specs.append(spec)
            manifest_runs.append({
                **metadata,
                "config_path": str(config_path),
                "resolved_config_sha256": stable_hash(config),
                "method_overrides": copy.deepcopy(experiment.get("overrides") or {}),
            })
    if write:
        manifest_path = generated_root / suite / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "suite": suite,
            "reference_id": str(study["reference"]["id"]),
            **git_identity(base_path.parent.parent),
            "protected_paths": protected_paths(base),
            "runs": manifest_runs,
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return specs


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    records = []
    if not path.is_file():
        return records
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StudyError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        if isinstance(value, dict):
            records.append(value)
    return records


def _load_router_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise StudyError(f"Expected a JSON list: {path}")
    return [item for item in value if isinstance(item, dict)]


def _add_ratio(
    metrics: MutableMapping[str, Any],
    numerator_key: str,
    denominator_key: str,
    output_key: str,
) -> None:
    """Add a zero-safe ratio only when both source counters were logged."""
    if numerator_key not in metrics or denominator_key not in metrics:
        return
    numerator = metrics[numerator_key]
    denominator = metrics[denominator_key]
    if not isinstance(numerator, (int, float)) or not isinstance(
        denominator, (int, float)
    ):
        return
    metrics[output_key] = (
        float(numerator) / float(denominator) if float(denominator) else 0.0
    )


def add_derived_epoch_metrics(metrics: MutableMapping[str, Any]) -> None:
    """Derive size-normalized prompt and safe-refinement diagnostics.

    ``router_statistics.json`` stores raw SvANet counters, while W&B computes
    some ratios separately.  Deriving the same ratios here makes offline
    ablation summaries independent of W&B and comparable across validation
    sets of different sizes.
    """
    for split in ("train", "validation"):
        svanet = f"{split}.svanet"
        small = f"{svanet}.small_count"
        trigger = f"{svanet}.trigger_count"
        _add_ratio(metrics, trigger, small, f"{svanet}.trigger_ratio")
        for name in (
            "empty_mask", "unreliable_mask", "low_area_confidence_skip",
            "no_reliable_roi_skip",
        ):
            _add_ratio(
                metrics, f"{svanet}.{name}_count", small,
                f"{svanet}.{name}_ratio",
            )
        for name in ("box_fallback", "locator_fallback", "full_image_fallback"):
            _add_ratio(
                metrics, f"{svanet}.{name}_count", trigger,
                f"{svanet}.{name}_ratio",
            )
        low_skip = f"{svanet}.low_area_confidence_skip_count"
        roi_skip = f"{svanet}.no_reliable_roi_skip_count"
        if low_skip in metrics and roi_skip in metrics:
            metrics[f"{svanet}.total_skip_count"] = (
                float(metrics[low_skip]) + float(metrics[roi_skip])
            )
            _add_ratio(
                metrics, f"{svanet}.total_skip_count", small,
                f"{svanet}.total_skip_ratio",
            )

        prompt_prefix = f"{split}.prompts."
        prompt_counts = {
            key: float(value)
            for key, value in metrics.items()
            if key.startswith(prompt_prefix)
            and key.count(".") == 2
            and isinstance(value, (int, float))
        }
        prompt_total = sum(prompt_counts.values())
        for key, value in prompt_counts.items():
            metrics[f"{key}_ratio"] = value / prompt_total if prompt_total else 0.0


def extract_run_metrics(output_dir: Path, epoch: str = "best") -> dict[str, Any] | None:
    """Read one training run and return flattened metrics for one selected epoch."""
    val_records = read_json_lines(output_dir / "val_stats.json")
    router_records = _load_router_records(output_dir / "router_statistics.json")
    available_epochs = {
        int(item["epoch"]): item for item in router_records if "epoch" in item
    }
    if not val_records and not available_epochs:
        return None
    if epoch == "best":
        if val_records:
            selected_val = min(val_records, key=lambda item: float(item["val_loss"]))
            selected_epoch = int(selected_val["epoch"])
        else:
            selected_epoch = max(available_epochs)
            selected_val = {}
    elif epoch == "last":
        epochs = [int(item["epoch"]) for item in val_records] + list(available_epochs)
        selected_epoch = max(epochs)
        selected_val = next(
            (item for item in reversed(val_records) if int(item["epoch"]) == selected_epoch),
            {},
        )
    else:
        try:
            selected_epoch = int(epoch)
        except ValueError as exc:
            raise StudyError("epoch must be 'best', 'last', or an integer") from exc
        selected_val = next(
            (item for item in val_records if int(item["epoch"]) == selected_epoch), {}
        )
    router = available_epochs.get(selected_epoch, {})
    combined: dict[str, Any] = {"epoch": selected_epoch}
    if selected_val:
        combined["train_loss"] = selected_val.get("train_loss")
        combined["val_loss"] = selected_val.get("val_loss")
    combined.update(flatten_mapping(router))
    add_derived_epoch_metrics(combined)
    return combined


def numeric_metrics(record: Mapping[str, Any]) -> dict[str, float]:
    result = {}
    for key, value in record.items():
        if key in {"epoch", "seed"} or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            result[key] = float(value)
    return result


def mean_sd(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    samples: int = 10000,
    seed: int = 20260910,
) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = random.Random(seed)
    n = len(values)
    estimates = sorted(
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(samples)
    )
    alpha = (1.0 - confidence) / 2.0
    low_index = max(0, min(samples - 1, int(math.floor(alpha * samples))))
    high_index = max(0, min(samples - 1, int(math.ceil((1.0 - alpha) * samples)) - 1))
    return estimates[low_index], estimates[high_index]


def paired_sign_flip_pvalue(
    differences: Sequence[float], seed: int = 20260910, max_draws: int = 100000
) -> float:
    """Exact (small n) or Monte-Carlo paired randomization p-value."""
    values = [float(value) for value in differences]
    if not values:
        return math.nan
    observed = abs(statistics.fmean(values))
    n = len(values)
    total = 1 << n if n <= 20 else max_draws
    extreme = 0
    rng = random.Random(seed)
    for draw in range(total):
        if n <= 20:
            signed = [value if draw & (1 << index) else -value for index, value in enumerate(values)]
        else:
            signed = [value if rng.random() < 0.5 else -value for value in values]
        if abs(statistics.fmean(signed)) >= observed - 1e-15:
            extreme += 1
    return extreme / total if n <= 20 else (extreme + 1) / (total + 1)


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    finite = sorted(
        ((name, value) for name, value in pvalues.items() if math.isfinite(value)),
        key=lambda item: item[1],
    )
    adjusted: dict[str, float] = {name: math.nan for name in pvalues}
    running = 0.0
    count = len(finite)
    for rank, (name, value) in enumerate(finite):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[name] = running
    return adjusted


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
