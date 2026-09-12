#!/usr/bin/env python3
"""Fast, GPU-free tests for the ablation orchestration layer."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ablation_lib import (
    StudyError,
    assert_protected_paths_unchanged,
    bootstrap_ci,
    deep_merge,
    extract_run_metrics,
    generate_run_specs,
    load_study,
    load_yaml,
    paired_sign_flip_pvalue,
    protected_paths,
    validate_method_config,
)
from summarize_ablation import DEFAULT_METRICS, build_long_rows, comparison_rows


HERE = Path(__file__).resolve().parent


class ConfigTests(unittest.TestCase):
    def test_deep_merge_replaces_list_and_preserves_siblings(self):
        base = {"a": {"x": 1, "items": [1, 2]}, "b": 2}
        result = deep_merge(base, {"a": {"items": [3], "y": 4}})
        self.assertEqual(result, {"a": {"x": 1, "items": [3], "y": 4}, "b": 2})
        self.assertEqual(base["a"]["items"], [1, 2])

    def test_protected_path_guard(self):
        base = {
            "dataset": {"mr_root": "/data/mr", "prompt_curriculum": {"enabled": True}},
            "model": {"sam3_checkpoint": "/weights/sam3.pt"},
        }
        safe = deep_merge(base, {"dataset": {"prompt_curriculum": {"enabled": False}}})
        assert_protected_paths_unchanged(base, safe)
        unsafe = deep_merge(base, {"dataset": {"mr_root": "/different"}})
        with self.assertRaises(StudyError):
            assert_protected_paths_unchanged(base, unsafe)

    def test_real_study_resolves_and_preserves_paths(self):
        study, path = load_study(HERE / "study.yaml")
        specs = generate_run_specs(
            study, path, suite="smoke", seeds=[42], write=False,
        )
        self.assertEqual({spec.experiment_id for spec in specs}, {"full_v2", "parallel_router"})
        self.assertTrue(protected_paths({"dataset": {"mr_root": "/x"}}))

    def test_all_suite_cell_count_and_routing_source_diagnostic(self):
        study, path = load_study(HERE / "study.yaml")
        specs = generate_run_specs(study, path, suite="all", write=False)
        self.assertEqual(len(specs), 93)  # 31 methods x 3 fixed screening seeds
        diagnostic = study["experiments"]["decoder_memory_routing_source"]
        self.assertEqual(
            diagnostic["overrides"]["moe"]["routing_feature_source"],
            "decoder_memory",
        )

    def test_generated_seed_reaches_all_runtime_seed_fields(self):
        study, path = load_study(HERE / "study.yaml")
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            specs = generate_run_specs(
                study, path, suite="smoke", seeds=[101],
                generated_root=temporary / "generated",
                output_root=temporary / "outputs",
                write=True,
            )
            config = load_yaml(specs[0].config_path)
            self.assertEqual(config["training"]["seed"], 101)
            self.assertEqual(config["training"]["data_order_seed"], 101)
            self.assertEqual(config["dataset"]["prompt_curriculum"]["seed"], 101)

    def test_prompt_probabilities_must_sum_to_one(self):
        config = {
            "dataset": {
                "prompt_curriculum": {
                    "enabled": True,
                    "evaluation_mode": "image_only",
                    "start_probabilities": {
                        "image_only": 0.8, "text": 0.1,
                        "coarse_box": 0.1, "accurate_box": 0.1,
                    },
                    "end_probabilities": {
                        "image_only": 1.0, "text": 0.0,
                        "coarse_box": 0.0, "accurate_box": 0.0,
                    },
                }
            }
        }
        with self.assertRaises(StudyError):
            validate_method_config(config)


class ResultTests(unittest.TestCase):
    def test_extract_best_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "val_stats.json").write_text(
                '\n'.join([
                    json.dumps({"epoch": 1, "train_loss": 2.0, "val_loss": 1.0}),
                    json.dumps({"epoch": 2, "train_loss": 1.5, "val_loss": 0.7}),
                ]) + '\n',
                encoding="utf-8",
            )
            (root / "router_statistics.json").write_text(
                json.dumps([
                    {"epoch": 1, "validation": {"segmentation": {"patient_macro_dice": 0.6}}},
                    {
                        "epoch": 2,
                        "stage": 5,
                        "train": {"loss": {"total_loss": 0.9}},
                        "validation": {
                            "loss": {"total_loss": 0.7, "locator_loss": 0.1},
                            "router": {
                                "area_routing_confidence": 0.72,
                                "boundary_routing_confidence": 0.68,
                                "area_shared_ratio": 0.10,
                                "area_topk_ratio": 0.30,
                                "area_top1_ratio": 0.60,
                                "boundary_shared_ratio": 0.20,
                                "boundary_topk_ratio": 0.25,
                                "boundary_top1_ratio": 0.55,
                            },
                            "svanet": {
                                "small_count": 20,
                                "trigger_count": 15,
                                "empty_mask_count": 4,
                                "unreliable_mask_count": 5,
                                "low_area_confidence_skip_count": 2,
                                "no_reliable_roi_skip_count": 3,
                                "training_cap_skip_count": 1,
                                "locator_fallback_count": 4,
                                "box_fallback_count": 1,
                                "full_image_fallback_count": 0,
                            },
                            "prompts": {"image_only": 20},
                            "segmentation": {"patient_macro_dice": 0.8},
                        },
                    },
                ]),
                encoding="utf-8",
            )
            metrics = extract_run_metrics(root, epoch="best")
            self.assertIsNotNone(metrics)
            assert metrics is not None
            self.assertEqual(metrics["epoch"], 2)
            self.assertEqual(metrics["validation.segmentation.patient_macro_dice"], 0.8)
            self.assertEqual(metrics["validation.svanet.trigger_ratio"], 0.75)
            self.assertEqual(metrics["validation.svanet.total_skip_ratio"], 0.30)
            self.assertEqual(metrics["validation.svanet.training_cap_skip_ratio"], 0.05)
            self.assertEqual(metrics["validation.svanet.locator_fallback_ratio"], 4 / 15)
            self.assertEqual(metrics["validation.prompts.image_only_ratio"], 1.0)

    def test_default_metrics_cover_v2_route_and_refinement_diagnostics(self):
        required = {
            "validation.router.area_routing_confidence",
            "validation.router.boundary_routing_confidence",
            "validation.router.area_shared_ratio",
            "validation.router.area_topk_ratio",
            "validation.router.area_top1_ratio",
            "validation.router.boundary_shared_ratio",
            "validation.router.boundary_topk_ratio",
            "validation.router.boundary_top1_ratio",
            "validation.svanet.trigger_ratio",
            "validation.svanet.total_skip_ratio",
            "validation.svanet.low_area_confidence_skip_ratio",
            "validation.svanet.no_reliable_roi_skip_ratio",
        }
        self.assertTrue(required.issubset(DEFAULT_METRICS))

    def test_best_falls_back_to_last_router_epoch_without_validation_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "router_statistics.json").write_text(
                json.dumps([
                    {"epoch": 1, "train": {"loss": {"total_loss": 2.0}}},
                    {"epoch": 3, "train": {"loss": {"total_loss": 1.0}}},
                ]),
                encoding="utf-8",
            )
            metrics = extract_run_metrics(root, epoch="best")
            self.assertIsNotNone(metrics)
            assert metrics is not None
            self.assertEqual(metrics["epoch"], 3)

    def test_statistics_are_deterministic(self):
        first = bootstrap_ci([0.1, 0.2, 0.3], samples=1000)
        second = bootstrap_ci([0.1, 0.2, 0.3], samples=1000)
        self.assertEqual(first, second)
        self.assertGreaterEqual(paired_sign_flip_pvalue([0.1, 0.1, 0.1]), 0.0)

    def test_summary_uses_same_seed_pairs(self):
        records = [
            {"experiment_id": "full_v2", "seed": 1, "epoch": 2, "metric": 0.8,
             "output_dir": "/tmp/a", "config_path": "/tmp/a.yaml"},
            {"experiment_id": "full_v2", "seed": 2, "epoch": 2, "metric": 0.7,
             "output_dir": "/tmp/b", "config_path": "/tmp/b.yaml"},
            {"experiment_id": "ablated", "seed": 1, "epoch": 2, "metric": 0.7,
             "output_dir": "/tmp/c", "config_path": "/tmp/c.yaml"},
        ]
        rows = build_long_rows(records)
        self.assertNotIn("seed", {row["metric"] for row in rows})
        comparisons = comparison_rows(
            rows, "full_v2", ["metric"], confidence=0.95, bootstrap_samples=100,
        )
        self.assertEqual(comparisons[0]["paired_seeds"], "1")
        self.assertAlmostEqual(
            comparisons[0]["mean_delta_candidate_minus_reference"], -0.1,
        )

    def test_summary_cli_reads_native_json_array_and_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated = root / "generated" / "primary"
            output = root / "output"
            generated.mkdir(parents=True)
            output.mkdir()
            (generated / "manifest.json").write_text(
                json.dumps({
                    "suite": "primary",
                    "reference_id": "full_v2",
                    "runs": [{
                        "experiment_id": "full_v2",
                        "seed": 42,
                        "output_dir": str(output),
                        "config_path": str(root / "seed_42.yaml"),
                    }],
                }),
                encoding="utf-8",
            )
            (output / "val_stats.json").write_text(
                json.dumps({"epoch": 1, "train_loss": 1.0, "val_loss": 0.8}) + "\n",
                encoding="utf-8",
            )
            (output / "router_statistics.json").write_text(
                json.dumps([{
                    "epoch": 1,
                    "stage": 5,
                    "validation": {
                        "router": {
                            "area_routing_confidence": 0.7,
                            "area_shared_ratio": 0.1,
                        },
                        "svanet": {
                            "small_count": 10,
                            "trigger_count": 8,
                            "low_area_confidence_skip_count": 1,
                            "no_reliable_roi_skip_count": 1,
                        },
                        "segmentation": {"patient_macro_dice": 0.85},
                    },
                }]),
                encoding="utf-8",
            )
            report_root = root / "reports"
            completed = subprocess.run(
                [
                    sys.executable, "-B", str(HERE / "summarize_ablation.py"),
                    "--generated-root", str(root / "generated"),
                    "--suite", "primary",
                    "--report-dir", str(report_root),
                    "--bootstrap-samples", "100",
                ],
                check=False, capture_output=True, text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            metrics_csv = (report_root / "primary" / "per_seed_metrics.csv").read_text(
                encoding="utf-8-sig"
            )
            self.assertIn("validation.router.area_shared_ratio", metrics_csv)
            self.assertIn("validation.svanet.trigger_ratio", metrics_csv)
            self.assertIn("validation.svanet.total_skip_ratio", metrics_csv)


if __name__ == "__main__":
    unittest.main()
