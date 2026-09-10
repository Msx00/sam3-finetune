"""GPU-free deployment-safety tests for ``infer_moe_sam3.py``."""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import pytest
import torch
from PIL import Image as PILImage


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_inference_module():
    """Load the CLI with real collator code but without optional SAM3 packages.

    The production environment owns dependencies such as iopath/decord.  These
    unit tests exercise only input construction, so lightweight data classes
    let the actual collator run without importing model building or CUDA code.
    """

    @dataclass
    class InferenceMetadata:
        coco_image_id: int
        original_image_id: int
        original_category_id: int
        original_size: tuple[int, int]
        object_id: int
        frame_index: int
        is_conditioning_only: Optional[bool] = False

    @dataclass
    class FindQueryLoaded:
        query_text: str
        image_id: int
        object_ids_output: List[int]
        is_exhaustive: bool
        query_processing_order: int = 0
        input_bbox: Optional[torch.Tensor] = None
        input_bbox_label: Optional[torch.Tensor] = None
        input_points: Optional[torch.Tensor] = None
        semantic_target: Optional[torch.Tensor] = None
        is_pixel_exhaustive: Optional[bool] = None
        inference_metadata: Optional[InferenceMetadata] = None

    @dataclass
    class Image:
        data: torch.Tensor
        objects: list
        size: tuple[int, int]
        blurring_mask: Optional[dict] = None

    @dataclass
    class Datapoint:
        find_queries: List[FindQueryLoaded]
        images: List[Image]
        raw_images: Optional[list] = None

    fake_image_dataset = types.ModuleType("sam3.train.data.sam3_image_dataset")
    fake_image_dataset.InferenceMetadata = InferenceMetadata
    fake_image_dataset.FindQueryLoaded = FindQueryLoaded
    fake_image_dataset.Image = Image
    fake_image_dataset.Datapoint = Datapoint

    fake_trainer = types.ModuleType("train_sam3_lora_native")
    fake_trainer.SAM3TrainerNative = type("SAM3TrainerNative", (), {})

    package_paths = {
        "sam3": PROJECT_ROOT / "sam3",
        "sam3.train": PROJECT_ROOT / "sam3" / "train",
        "sam3.train.data": PROJECT_ROOT / "sam3" / "train" / "data",
    }
    replacements = {
        name: types.ModuleType(name) for name in package_paths
    }
    for name, path in package_paths.items():
        replacements[name].__path__ = [str(path)]
    replacements[fake_image_dataset.__name__] = fake_image_dataset
    replacements[fake_trainer.__name__] = fake_trainer

    previous = {name: sys.modules.get(name) for name in replacements}
    modules_before = set(sys.modules)
    inserted_project_path = str(PROJECT_ROOT) not in sys.path
    try:
        if inserted_project_path:
            sys.path.insert(0, str(PROJECT_ROOT))
        sys.modules.update(replacements)
        spec = importlib.util.spec_from_file_location(
            "_infer_moe_sam3_deployment_test", PROJECT_ROOT / "infer_moe_sam3.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        if inserted_project_path:
            sys.path.remove(str(PROJECT_ROOT))
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module
        for name in set(sys.modules) - modules_before:
            if name.startswith("sam3.") or name == "sam3":
                sys.modules.pop(name, None)
    return module


infer = _load_inference_module()


def _write_image(path: Path, size=(7, 5), color=(10, 80, 240)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.new("RGB", size, color).save(path)


def test_default_cli_uses_direct_stage5_config():
    args = infer.parse_args(["--checkpoint", "weights.pt"])

    assert args.config == "configs/moe_sam3_stage5_direct.yaml"
    assert args.task_text == "prostate"
    assert args.image is None
    assert args.image_dir is None


def test_raw_slice_datapoint_has_no_gt_or_modality_metadata(tmp_path):
    image_path = tmp_path / "slice.png"
    _write_image(image_path, size=(7, 5))
    dataset = infer.ImageOnlySliceDataset(
        [image_path], task_text="target organ", resolution=12
    )

    datapoint = dataset[0]
    query = datapoint.find_queries[0]
    assert datapoint.images[0].data.shape == (3, 12, 12)
    assert datapoint.images[0].data.dtype == torch.float32
    assert datapoint.images[0].objects == []
    assert datapoint.raw_images is None
    assert query.query_text == "target organ"
    assert query.object_ids_output == []
    assert query.input_bbox is None
    assert query.input_bbox_label is None
    assert query.semantic_target is None
    assert query.inference_metadata.original_size == (5, 7)
    assert query.inference_metadata.original_category_id == -1

    forbidden = {
        "mask_path",
        "box_prompt",
        "area_ratio",
        "area_label",
        "boundary_label",
        "modality",
        "modality_label",
    }
    assert forbidden.isdisjoint(datapoint.patient_metadata)
    assert datapoint.patient_metadata["uses_external_bbox"] is False

    batch = infer._image_only_collate([datapoint], "target organ")
    infer.assert_image_only_batch(
        batch["input"], "target organ", batch["_patient_metadata"]
    )
    assert batch["input"].img_batch.shape == (1, 3, 12, 12)
    assert batch["input"].find_targets[0].segments is None


def test_dataset_split_wrapper_strips_prompt_objects_and_targets():
    metadata = infer.InferenceMetadata(
        coco_image_id=91,
        original_image_id=91,
        original_category_id=7,
        original_size=(30, 20),
        object_id=0,
        frame_index=0,
    )
    annotated = infer.Datapoint(
        find_queries=[
            infer.FindQueryLoaded(
                query_text="annotation class",
                image_id=0,
                object_ids_output=[0],
                is_exhaustive=True,
                input_bbox=torch.tensor([[0.5, 0.5, 0.2, 0.3]]),
                input_bbox_label=torch.ones(1, dtype=torch.long),
                semantic_target=torch.ones(8, 8),
                inference_metadata=metadata,
            )
        ],
        images=[
            infer.SAM3Image(
                data=torch.zeros(3, 8, 8),
                objects=[SimpleNamespace(bbox=torch.ones(4), segment=torch.ones(8, 8))],
                size=(8, 8),
            )
        ],
    )
    annotated.patient_metadata = {
        "image_path": "/kept/for/output.png",
        "mask_path": "/kept/for/evaluation.png",
        "modality": "MR",
        "area_label": 2,
        "uses_external_bbox": True,
    }

    clean = infer.DeploymentImageOnlyDataset([annotated], "fixed target")[0]

    assert clean.images[0].objects == []
    assert clean.raw_images is None
    assert len(clean.find_queries) == 1
    assert clean.find_queries[0].query_text == "fixed target"
    assert clean.find_queries[0].object_ids_output == []
    assert clean.find_queries[0].input_bbox is None
    assert clean.find_queries[0].semantic_target is None
    assert clean.find_queries[0].inference_metadata.original_category_id == -1
    # Evaluation-only provenance remains outside the model input.
    assert clean.patient_metadata["mask_path"] == "/kept/for/evaluation.png"
    assert clean.patient_metadata["uses_external_bbox"] is False

    batch = infer._image_only_collate([clean], "fixed target")
    infer.assert_image_only_batch(
        batch["input"], "fixed target", batch["_patient_metadata"]
    )
    assert int(batch["input"].find_targets[0].num_boxes.sum()) == 0


def test_image_only_assertions_fail_closed_on_box_or_gt_metadata(tmp_path):
    image_path = tmp_path / "slice.png"
    _write_image(image_path)
    datapoint = infer.ImageOnlySliceDataset([image_path], "organ", resolution=8)[0]
    batch = infer._image_only_collate([datapoint], "organ")
    find_input = batch["input"].find_inputs[0]
    find_input.input_boxes = torch.ones(1, 1, 4)
    find_input.input_boxes_mask = torch.zeros(1, 1, dtype=torch.bool)

    with pytest.raises(RuntimeError, match="bounding-box"):
        infer.assert_image_only_batch(batch["input"], "organ")

    clean_batch = infer._image_only_collate([datapoint], "organ")
    with pytest.raises(RuntimeError, match="Metadata declares"):
        infer.assert_image_only_batch(
            clean_batch["input"], "organ", [{"uses_external_bbox": True}]
        )


def test_directory_discovery_is_recursive_sorted_and_bounded(tmp_path):
    _write_image(tmp_path / "b.PNG")
    _write_image(tmp_path / "nested" / "a.jpg")
    (tmp_path / "ignore.txt").write_text("not an image", encoding="utf-8")

    paths = infer.discover_image_paths(image_dir=str(tmp_path))
    assert paths == sorted([tmp_path / "b.PNG", tmp_path / "nested" / "a.jpg"])
    assert infer.discover_image_paths(image_dir=str(tmp_path), maximum=1) == paths[:1]
    with pytest.raises(ValueError, match="positive"):
        infer.discover_image_paths(image_dir=str(tmp_path), maximum=0)


def test_runtime_disables_resume_and_forces_zero_interactive_steps(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "training:\n  stage: 5\nmoe: {}\nrouter: {}\nsvanet: {}\n",
        encoding="utf-8",
    )
    calls = {}

    class FakeStageManager:
        def load_checkpoint(self, checkpoint, allowed_stages):
            calls["checkpoint"] = (checkpoint, allowed_stages)

        def set_module_modes(self, training):
            calls["training"] = training

    class FakeTrainer:
        def __init__(self, config, **kwargs):
            calls["trainer"] = (config, kwargs)
            self.model = SimpleNamespace(num_interactive_steps_val=9)
            self._unwrapped_model = self.model
            self.stage_manager = FakeStageManager()

    monkeypatch.setattr(infer, "SAM3TrainerNative", FakeTrainer)
    monkeypatch.setattr(infer, "resolve_moe_runtime_config", lambda config: {"ok": True})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    trainer, _, stage = infer.build_runtime(
        str(config_path), "checkpoint.pt", stage=None, device=0
    )

    assert stage == 5
    assert calls["trainer"][1]["load_training_resume"] is False
    assert calls["training"] is False
    assert calls["checkpoint"] == ("checkpoint.pt", {5})
    assert trainer.model.num_interactive_steps_val == 0
    infer.assert_zero_interactive_steps(trainer)
    trainer.model.num_interactive_steps_val = 1
    with pytest.raises(RuntimeError, match="num_interactive_steps_val"):
        infer.assert_zero_interactive_steps(trainer)
