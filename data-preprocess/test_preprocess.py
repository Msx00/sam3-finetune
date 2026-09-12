#!/usr/bin/env python3
"""Unit tests for the ``data-preprocess`` pipeline.

Run from this directory with the training environment (needs neither torch nor
a GPU)::

    python -m unittest -v test_preprocess.py

The tests build a miniature dataset in a temporary directory, so they never
touch the real ``/mnt/afs`` data.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import preprocess as pp  # noqa: E402


def _touch(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, (dict, list)):
        path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        path.write_text(str(payload), encoding="utf-8")


def _write_shard(path: Path, slices, version: int, band_width: int, modality="mr", split="train"):
    _touch(
        path,
        {
            "version": version,
            "modality": modality,
            "split": split,
            "patient_id": int(path.stem),
            "boundary_band_width": band_width,
            "slices": {name: {"area_ratio": 0.02} for name in slices},
        },
    )


class SelectionTests(unittest.TestCase):
    def test_random_selection_is_sorted_and_reproducible(self):
        available = list(range(1, 51))
        first = pp.select_patient_ids(available, 10, None, "random", 42)
        second = pp.select_patient_ids(available, 10, None, "random", 42)
        self.assertEqual(first, second)
        self.assertEqual(first, sorted(first))
        self.assertEqual(len(first), 10)
        self.assertNotEqual(first, pp.select_patient_ids(available, 10, None, "random", 7))

    def test_sequential_and_explicit_selection(self):
        available = list(range(1, 11))
        self.assertEqual(pp.select_patient_ids(available, 3, None, "sequential", 42), [1, 2, 3])
        self.assertEqual(pp.select_patient_ids(available, None, None, "random", 1), available)
        self.assertEqual(pp.select_patient_ids(available, None, [5, 2], "random", 1), [2, 5])

    def test_missing_explicit_patient_is_rejected(self):
        with self.assertRaises(pp.PreprocessError):
            pp.select_patient_ids([1, 2, 3], None, [9], "random", 1)

    def test_too_many_patients_is_rejected(self):
        with self.assertRaises(pp.PreprocessError):
            pp.select_patient_ids([1, 2, 3], 4, None, "random", 1)

    def test_matches_patient_dataset_selection(self):
        """The pipeline must pick exactly the patients the trainer would pick."""
        try:
            sys.path.insert(0, str(pp.PROJECT_ROOT))
            from data.patient_dataset import PatientDataset  # type: ignore
        except Exception as error:  # pragma: no cover - torch missing
            self.skipTest(f"PatientDataset unavailable: {error}")
        available = list(range(1, 60))
        for mode, seed, count in (("random", 42, 7), ("random", 7, 13), ("sequential", 42, 5)):
            with self.subTest(mode=mode, seed=seed, count=count):
                expected = PatientDataset._select_patient_ids(
                    available, count, None, mode, seed
                )
                self.assertEqual(
                    pp.select_patient_ids(available, count, None, mode, seed), expected
                )


class ManifestTests(unittest.TestCase):
    def test_manifest_keys_match_trainer_writer(self):
        source = (pp.PROJECT_ROOT / "data" / "patient_dataset.py").read_text(encoding="utf-8")
        block = source.split("def save_selected_patients", 1)[1].split("def ", 1)[0]
        trainer_keys = set(re.findall(r'"([a-z_]+)":', block))
        required = {
            "mr_patient_ids",
            "us_patient_ids",
            "num_mr_patients",
            "num_us_patients",
            "num_mr_slices",
            "num_us_slices",
            "total_slices",
            "sampling_mode",
            "seed",
            "resample_patients_each_epoch",
        }
        self.assertTrue(required.issubset(trainer_keys), trainer_keys)


class PipelineTests(unittest.TestCase):
    """End-to-end check of manifest/stage/verify handling on a tiny dataset."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.plan = {
            "mr": {"train": [1, 2, 3], "val": [1], "test": [1]},
            "us": {"train": [1, 2], "val": [1], "test": [1]},
        }
        self.slices = {"mr": ["slice_0000.png", "slice_0001.png"], "us": ["slice_0000.png"]}
        for modality, splits in self.plan.items():
            for split, patients in splits.items():
                for patient_id in patients:
                    for name in self.slices[modality]:
                        _touch(self.root / f"{modality}-2d" / split / str(patient_id) / name, "png")
                        _touch(
                            self.root / f"{modality}-mask-2d" / split / str(patient_id) / name,
                            "png",
                        )
                _touch(self.root / f"{modality}-2d" / f"{split}.json", {"images": []})
        self.dataset = self._dataset_config()
        self.paths = pp.resolve_paths(self.dataset, self.root / "config.yaml")

    def tearDown(self):
        self._tmp.cleanup()

    def _dataset_config(self):
        dataset = {
            "train_split": "train",
            "val_split": "val",
            "num_mr_patients": 2,
            "num_us_patients": 2,
            "patient_sampling": {"mode": "sequential", "seed": 42},
            "num_mr_val_patients": 1,
            "num_us_val_patients": 1,
            "val_patient_sampling": {"mode": "sequential", "seed": 42},
            "strict_dataset_check": True,
            "label_cache_dir": str(self.root / "labels"),
            "selected_patients_json": str(self.root / "selected_patients.json"),
            "selected_val_patients_json": str(self.root / "selected_val_patients.json"),
            "area_threshold_file": str(self.root / "area_thresholds.json"),
            "boundary_threshold_file": str(self.root / "boundary_thresholds.json"),
        }
        for modality in ("mr", "us"):
            dataset[f"{modality}_root"] = str(self.root / f"{modality}-2d")
            dataset[f"{modality}_mask_root"] = str(self.root / f"{modality}-mask-2d")
        return dataset

    def _args(self, **overrides):
        args = pp.parse_args(["all"])
        args.splits = ["train", "val", "test"]
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def _write_ready_artifacts(self):
        args = self._args()
        self.assertTrue(pp.stage_patients(self.paths, self.dataset, args))
        for modality, splits in self.plan.items():
            for split in splits:
                coco_path = pp.coco_json_path(self.dataset, modality, split)
                boxes_path = pp.boxes_json_path(self.dataset, modality, split)
                _touch(
                    coco_path,
                    {
                        "images": [
                            {"id": 0, "file_name": f"{modality}/{split}", "width": 8, "height": 8}
                        ],
                        "annotations": [{"id": 0}],
                        "categories": [{"id": 1, "name": "prostate"}],
                    },
                )
                _touch(boxes_path, {"1/slice_0000.png": [[1.0, 1.0, 3.0, 3.0]]})
        _touch(self.paths.area_thresholds, {"small_max": 0.01, "medium_max": 0.05})
        _touch(self.paths.boundary_thresholds, {"mr": {}, "us": {}, "boundary_band_width": 3})
        version = pp.label_cache_version()
        for split, manifest_path in (
            ("train", self.paths.selected_manifest),
            ("val", self.paths.selected_val_manifest),
        ):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for modality in ("mr", "us"):
                for patient_id in manifest[f"{modality}_patient_ids"]:
                    _write_shard(
                        self.paths.label_cache_dir / modality / split / f"{patient_id}.json",
                        self.slices[modality],
                        version,
                        3,
                        modality,
                        split,
                    )

    def test_build_manifest_counts_selected_slices(self):
        manifest = pp.build_manifest(self.dataset, "train")
        self.assertEqual(manifest["mr_patient_ids"], [1, 2])
        self.assertEqual(manifest["us_patient_ids"], [1, 2])
        self.assertEqual(manifest["num_mr_slices"], 4)
        self.assertEqual(manifest["total_slices"], 6)
        val = pp.build_manifest(self.dataset, "val")
        self.assertEqual(val["mr_patient_ids"], [1])
        self.assertEqual(val["total_slices"], 3)

    def test_verify_passes_on_complete_artifacts(self):
        self._write_ready_artifacts()
        problems = pp.verify(self.paths, self.dataset, splits=("train", "val", "test"))
        self.assertEqual(problems, [])

    def test_verify_flags_missing_label_shard(self):
        self._write_ready_artifacts()
        victim = self.paths.label_cache_dir / "mr" / "train" / "2.json"
        victim.unlink()
        problems = pp.verify(self.paths, self.dataset, splits=("train", "val", "test"))
        self.assertTrue(any("label shard" in item for item in problems), problems)

    def test_verify_flags_incomplete_shard_and_stale_band_width(self):
        self._write_ready_artifacts()
        shard = self.paths.label_cache_dir / "us" / "train" / "1.json"
        payload = json.loads(shard.read_text(encoding="utf-8"))
        payload["slices"] = {"slice_0000.png": {}}
        payload["boundary_band_width"] = 5
        _touch(shard, payload)
        problems = pp.verify(self.paths, self.dataset, splits=("train", "val", "test"))
        self.assertTrue(any("stale label shard" in item for item in problems), problems)

    def test_verify_flags_drifted_manifest(self):
        self._write_ready_artifacts()
        manifest = json.loads(self.paths.selected_manifest.read_text(encoding="utf-8"))
        manifest["mr_patient_ids"] = [3]
        _touch(self.paths.selected_manifest, manifest)
        problems = pp.verify(self.paths, self.dataset, splits=("train", "val", "test"))
        self.assertTrue(any("manifest patient ids differ" in item for item in problems), problems)

    def test_verify_flags_missing_sam3_json(self):
        self._write_ready_artifacts()
        pp.coco_json_path(self.dataset, "mr", "train").unlink()
        problems = pp.verify(self.paths, self.dataset, splits=("train", "val", "test"))
        self.assertTrue(any("cannot read" in item for item in problems), problems)

    def test_coco_paths_honour_configured_keys(self):
        self.dataset["mr_train_coco_json"] = str(self.root / "custom" / "train_sam3.json")
        self.dataset["mr_train_boxes_json"] = str(self.root / "custom" / "train_boxes.json")
        self.assertEqual(
            pp.coco_json_path(self.dataset, "mr", "train"),
            self.root / "custom" / "train_sam3.json",
        )
        self.assertEqual(
            pp.boxes_json_path(self.dataset, "mr", "train"),
            self.root / "custom" / "train_boxes.json",
        )
        self.assertEqual(
            pp.coco_json_path(self.dataset, "us", "train"),
            self.root / "us-2d" / "train_sam3.json",
        )

    def test_stage_coco_skips_up_to_date_targets(self):
        self._write_ready_artifacts()
        args = self._args()
        with mock.patch.object(pp, "_run") as run:
            self.assertTrue(pp.stage_coco(self.paths, self.dataset, args))
        run.assert_not_called()

    def test_stage_labels_builds_expected_command(self):
        args = self._args(workers=7, force=True, limit=3)
        with mock.patch.object(pp, "_run", return_value=True) as run:
            self.assertTrue(pp.stage_labels(self.paths, self.dataset, args))
        command = [str(part) for part in run.call_args[0][0]]
        self.assertIn("prepare_slice_labels.py", command[1])
        self.assertIn("--overwrite", command)
        self.assertIn("--scope", command)
        self.assertEqual(command[command.index("--workers") + 1], "7")
        self.assertEqual(command[command.index("--limit") + 1], "3")

    def test_thresholds_are_not_recomputed_without_force(self):
        self._write_ready_artifacts()
        args = self._args(force=False)
        with mock.patch.object(pp, "_run") as run:
            self.assertTrue(pp.stage_thresholds(self.paths, self.dataset, args))
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
