#!/usr/bin/env python3
"""Aggregate seed repetitions and perform paired ablation comparisons."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from ablation_lib import (
    StudyError,
    bootstrap_ci,
    extract_run_metrics,
    holm_adjust,
    mean_sd,
    numeric_metrics,
    paired_sign_flip_pvalue,
    write_csv,
)


HERE = Path(__file__).resolve().parent

DEFAULT_METRICS = [
    # Deployment segmentation quality.
    "validation.segmentation.patient_macro_dice",
    "validation.segmentation.slice_dice",
    "validation.segmentation.patient_macro_iou",
    "validation.segmentation.MR_dice",
    "validation.segmentation.US_dice",
    "validation.segmentation.small_dice",
    "validation.segmentation.medium_dice",
    "validation.segmentation.large_dice",
    "validation.segmentation.clear_dice",
    "validation.segmentation.fuzzy_dice",
    "validation.segmentation.complex_dice",
    # Router correctness and calibration.
    "validation.router.modality_accuracy",
    "validation.router.area_accuracy",
    "validation.router.boundary_accuracy",
    "validation.router.modality_entropy",
    "validation.router.area_entropy",
    "validation.router.boundary_entropy",
    "validation.router.area_routing_confidence",
    "validation.router.boundary_routing_confidence",
    "validation.router.area_shared_ratio",
    "validation.router.area_topk_ratio",
    "validation.router.area_top1_ratio",
    "validation.router.boundary_shared_ratio",
    "validation.router.boundary_topk_ratio",
    "validation.router.boundary_top1_ratio",
    "validation.router.actual_teacher_forcing_ratio",
    # Safe-refinement execution/fallback accounting. Ratios are derived from
    # the raw counters in router_statistics.json, without requiring W&B.
    "validation.svanet.trigger_ratio",
    "validation.svanet.total_skip_ratio",
    "validation.svanet.low_area_confidence_skip_ratio",
    "validation.svanet.no_reliable_roi_skip_ratio",
    "validation.svanet.unreliable_mask_ratio",
    "validation.svanet.locator_fallback_ratio",
    "validation.svanet.box_fallback_ratio",
    "validation.svanet.full_image_fallback_ratio",
    # Protocol audit and optimization diagnostics.
    "validation.prompts.image_only_ratio",
    "validation.loss.locator_loss",
    "val_loss",
]


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise StudyError(f"Manifest does not exist; run generate first: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("runs"), list):
        raise StudyError(f"Invalid manifest: {path}")
    return value


def collect_records(
    manifest: Mapping[str, Any], epoch: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for run in manifest["runs"]:
        output_dir = Path(str(run["output_dir"]))
        metrics = extract_run_metrics(output_dir, epoch=epoch)
        identity = {
            "experiment_id": str(run["experiment_id"]),
            "seed": int(run["seed"]),
            "output_dir": str(output_dir),
            "config_path": str(run["config_path"]),
        }
        if metrics is None:
            missing.append(identity)
            continue
        records.append({**identity, **metrics})
    return records, missing


def build_long_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        for metric, value in numeric_metrics(record).items():
            rows.append({
                "experiment_id": record["experiment_id"],
                "seed": record["seed"],
                "epoch": record.get("epoch"),
                "metric": metric,
                "value": value,
                "output_dir": record["output_dir"],
            })
    return rows


def aggregate_rows(
    long_rows: Sequence[Mapping[str, Any]], confidence: float, bootstrap_samples: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in long_rows:
        grouped[(str(row["experiment_id"]), str(row["metric"]))].append(
            float(row["value"])
        )
    result = []
    for (experiment_id, metric), values in sorted(grouped.items()):
        mean, sd = mean_sd(values)
        low, high = bootstrap_ci(
            values, confidence=confidence, samples=bootstrap_samples
        )
        result.append({
            "experiment_id": experiment_id,
            "metric": metric,
            "n": len(values),
            "mean": mean,
            "sd": sd,
            "ci_low": low,
            "ci_high": high,
        })
    return result


def comparison_rows(
    long_rows: Sequence[Mapping[str, Any]],
    reference_id: str,
    metrics: Sequence[str],
    confidence: float,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int, str], float] = {}
    experiments = set()
    for row in long_rows:
        experiment = str(row["experiment_id"])
        metric = str(row["metric"])
        experiments.add(experiment)
        by_key[(experiment, int(row["seed"]), metric)] = float(row["value"])
    comparisons: list[dict[str, Any]] = []
    raw_pvalues: dict[tuple[str, str], float] = {}
    for metric in metrics:
        reference_seeds = {
            seed: value
            for (experiment, seed, name), value in by_key.items()
            if experiment == reference_id and name == metric
        }
        for experiment in sorted(experiments - {reference_id}):
            candidate_seeds = {
                seed: value
                for (name, seed, metric_name), value in by_key.items()
                if name == experiment and metric_name == metric
            }
            common = sorted(set(reference_seeds) & set(candidate_seeds))
            if not common:
                continue
            differences = [
                candidate_seeds[seed] - reference_seeds[seed] for seed in common
            ]
            mean, sd = mean_sd(differences)
            low, high = bootstrap_ci(
                differences, confidence=confidence, samples=bootstrap_samples
            )
            pvalue = paired_sign_flip_pvalue(differences)
            raw_pvalues[(experiment, metric)] = pvalue
            comparisons.append({
                "experiment_id": experiment,
                "reference_id": reference_id,
                "metric": metric,
                "paired_seeds": ",".join(map(str, common)),
                "n_pairs": len(common),
                "mean_delta_candidate_minus_reference": mean,
                "sd_delta": sd,
                "ci_low": low,
                "ci_high": high,
                "randomization_p": pvalue,
                "holm_p": math.nan,
            })
    # Control family-wise error independently for each reported metric.
    for metric in metrics:
        family = {
            experiment: value
            for (experiment, name), value in raw_pvalues.items()
            if name == metric
        }
        adjusted = holm_adjust(family)
        for row in comparisons:
            if row["metric"] == metric:
                row["holm_p"] = adjusted.get(str(row["experiment_id"]), math.nan)
    return comparisons


def format_number(value: Any, digits: int = 4) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def write_markdown_report(
    path: Path,
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    missing: Sequence[Mapping[str, Any]],
    aggregates: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    primary_metric: str,
    epoch: str,
) -> None:
    reference_id = str(manifest["reference_id"])
    selected_aggregates = {
        str(row["experiment_id"]): row
        for row in aggregates if row["metric"] == primary_metric
    }
    selected_comparisons = {
        str(row["experiment_id"]): row
        for row in comparisons if row["metric"] == primary_metric
    }
    lines = [
        "# HMoE-SAM3 消融实验统计报告",
        "",
        f"- Suite: `{manifest.get('suite', '')}`",
        f"- Reference: `{reference_id}`",
        f"- Epoch selection: `{epoch}`",
        f"- Primary metric: `{primary_metric}`",
        f"- Completed runs: {len(records)}",
        f"- Missing runs: {len(missing)}",
        "",
        "## Primary result",
        "",
        "| Experiment | n | Mean ± SD | 95% bootstrap CI | Δ vs reference | Holm p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for experiment, aggregate in sorted(selected_aggregates.items()):
        comparison = selected_comparisons.get(experiment, {})
        delta = "reference" if experiment == reference_id else format_number(
            comparison.get("mean_delta_candidate_minus_reference")
        )
        holm = "-" if experiment == reference_id else format_number(
            comparison.get("holm_p")
        )
        lines.append(
            f"| {experiment} | {aggregate['n']} | "
            f"{format_number(aggregate['mean'])} ± {format_number(aggregate['sd'])} | "
            f"[{format_number(aggregate['ci_low'])}, {format_number(aggregate['ci_high'])}] | "
            f"{delta} | {holm} |"
        )
    lines.extend([
        "",
        "## Completeness",
        "",
    ])
    if missing:
        lines.append("以下运行缺少 `val_stats.json` 和/或 `router_statistics.json`：")
        lines.append("")
        for item in missing:
            lines.append(f"- `{item['experiment_id']}` seed={item['seed']}: `{item['output_dir']}`")
    else:
        lines.append("全部 manifest 运行均有可读取结果。")
    lines.extend([
        "",
        "## Statistical note",
        "",
        "差值为 candidate − full_v2，并且仅使用两方法共有的相同 seed。置信区间为种子级配对 bootstrap；p 值为双侧配对 sign-flip randomization test，并在每个 metric 内采用 Holm 校正。三 seed 仅适合筛选，最终论文建议至少五 seed，并同时报告病例级 bootstrap 结果。",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize repeated-seed HMoE-SAM3 ablations"
    )
    parser.add_argument("--generated-root", type=Path, default=HERE / "generated")
    parser.add_argument("--suite", default="primary")
    parser.add_argument("--epoch", default="best", help="best, last, or integer")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument(
        "--primary-metric",
        default="validation.segmentation.patient_macro_dice",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--report-dir", type=Path, default=HERE / "reports")
    parser.add_argument(
        "--allow-incomplete", action="store_true",
        help="Return success even if one or more manifest runs have no results",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not (0.0 < args.confidence < 1.0):
        raise SystemExit("--confidence must be between 0 and 1")
    if args.bootstrap_samples < 100:
        raise SystemExit("--bootstrap-samples must be at least 100")
    manifest_path = args.generated_root.expanduser().resolve() / args.suite / "manifest.json"
    try:
        manifest = load_manifest(manifest_path)
        records, missing = collect_records(manifest, args.epoch)
        if not records:
            raise StudyError("No completed result records were found")
        long_rows = build_long_rows(records)
        aggregates = aggregate_rows(
            long_rows, confidence=args.confidence,
            bootstrap_samples=args.bootstrap_samples,
        )
        comparisons = comparison_rows(
            long_rows,
            reference_id=str(manifest["reference_id"]),
            metrics=args.metrics,
            confidence=args.confidence,
            bootstrap_samples=args.bootstrap_samples,
        )
    except (StudyError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    report_dir = args.report_dir.expanduser().resolve() / args.suite
    write_csv(
        report_dir / "per_seed_metrics.csv", long_rows,
        ["experiment_id", "seed", "epoch", "metric", "value", "output_dir"],
    )
    write_csv(
        report_dir / "aggregate_metrics.csv", aggregates,
        ["experiment_id", "metric", "n", "mean", "sd", "ci_low", "ci_high"],
    )
    write_csv(
        report_dir / "paired_comparisons.csv", comparisons,
        [
            "experiment_id", "reference_id", "metric", "paired_seeds", "n_pairs",
            "mean_delta_candidate_minus_reference", "sd_delta", "ci_low", "ci_high",
            "randomization_p", "holm_p",
        ],
    )
    (report_dir / "missing_runs.json").write_text(
        json.dumps(missing, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown_report(
        report_dir / "report.md",
        manifest=manifest,
        records=records,
        missing=missing,
        aggregates=aggregates,
        comparisons=comparisons,
        primary_metric=args.primary_metric,
        epoch=args.epoch,
    )
    print(f"Collected {len(records)} runs; missing {len(missing)}")
    print(f"Reports: {report_dir}")
    if missing and not args.allow_incomplete:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
