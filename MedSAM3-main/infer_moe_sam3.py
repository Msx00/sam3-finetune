#!/usr/bin/env python3
"""Inference entry for Hierarchical LoRA-MoE-SAM3 with optional SvANet."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import yaml
from PIL import Image as PILImage, ImageDraw
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from models.inference_utils import (
    move_to_device, normalized_xyxy_prompts, route_predictions,
    select_best_mask_logits,
)
from models.runtime_config import resolve_moe_runtime_config
from models.svanet_roi_adapter import largest_component_bbox
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image as SAM3Image,
    InferenceMetadata,
)
from train_sam3_lora_native import SAM3TrainerNative


_MODEL_RESOLUTION = 1008
_IMAGE_SUFFIXES = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Infer Hierarchical LoRA-MoE-SAM3")
    parser.add_argument("--config", default="configs/moe_sam3_stage5_direct.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage", type=int, choices=range(1, 6), default=None)
    parser.add_argument("--split", default=None)
    image_source = parser.add_mutually_exclusive_group()
    image_source.add_argument(
        "--image",
        help="Run a single image slice without loading dataset annotations.",
    )
    image_source.add_argument(
        "--image-dir",
        help="Run all supported image slices below a directory, recursively.",
    )
    parser.add_argument(
        "--task-text",
        default="prostate",
        help="One fixed text query used for every slice (default: prostate).",
    )
    parser.add_argument("--output-dir", default="outputs/moe_inference")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args(argv)


def _validate_task_text(task_text: str) -> str:
    task_text = str(task_text).strip()
    if not task_text:
        raise ValueError("--task-text must contain a non-empty fixed query")
    return task_text


def _neutral_inference_metadata(
    sample_index: int,
    original_size: tuple[int, int],
) -> InferenceMetadata:
    """Return structural SAM3 metadata without labels from an annotation file."""
    sample_index = int(sample_index)
    return InferenceMetadata(
        coco_image_id=sample_index,
        original_image_id=sample_index,
        original_category_id=-1,
        original_size=(int(original_size[0]), int(original_size[1])),
        object_id=-1,
        frame_index=0,
        is_conditioning_only=False,
    )


def _image_only_datapoint(
    image: SAM3Image,
    task_text: str,
    sample_index: int,
    original_size: tuple[int, int],
) -> Datapoint:
    """Construct the only model input accepted by this deployment CLI.

    The returned sample has one fixed text query and deliberately contains no
    output object IDs, geometry prompt, mask, semantic target, or raw image.
    """
    task_text = _validate_task_text(task_text)
    clean_image = SAM3Image(data=image.data, objects=[], size=image.size)
    query = FindQueryLoaded(
        query_text=task_text,
        image_id=0,
        object_ids_output=[],
        is_exhaustive=False,
        query_processing_order=0,
        input_bbox=None,
        input_bbox_label=None,
        input_points=None,
        semantic_target=None,
        inference_metadata=_neutral_inference_metadata(
            sample_index, original_size
        ),
    )
    return Datapoint(find_queries=[query], images=[clean_image], raw_images=None)


class ImageOnlySliceDataset(Dataset):
    """PIL-backed slice dataset which never opens annotation or prompt files."""

    def __init__(
        self,
        image_paths: Sequence[str | Path],
        task_text: str,
        resolution: int = _MODEL_RESOLUTION,
    ) -> None:
        self.image_paths = [Path(path) for path in image_paths]
        self.task_text = _validate_task_text(task_text)
        self.resolution = int(resolution)
        if not self.image_paths:
            raise ValueError("At least one input image is required")
        if self.resolution <= 0:
            raise ValueError("resolution must be positive")
        for path in self.image_paths:
            if not path.is_file():
                raise FileNotFoundError(f"Input image not found: {path}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Datapoint:
        path = self.image_paths[index]
        with PILImage.open(path) as source:
            image = source.convert("RGB")
            original_width, original_height = image.size
            resized = image.resize(
                (self.resolution, self.resolution), PILImage.Resampling.BILINEAR
            )
            array = np.asarray(resized, dtype=np.float32).copy() / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        tensor = (tensor - 0.5) / 0.5
        datapoint = _image_only_datapoint(
            SAM3Image(
                data=tensor,
                objects=[],
                size=(self.resolution, self.resolution),
            ),
            task_text=self.task_text,
            sample_index=index,
            original_size=(original_height, original_width),
        )
        # Only operational provenance is retained.  In particular, this does
        # not contain a mask, GT box, area class, or modality label.
        datapoint.patient_metadata = {
            "image_path": str(path),
            "sample_id": f"{index:06d}_{path.stem}",
            "slice_id": path.stem,
            "original_size": [original_height, original_width],
            "task_text": self.task_text,
            "prompt_mode": "image_only",
            "applied_prompt_text": self.task_text,
            "applied_box_prompt": [],
            "uses_external_bbox": False,
            "deployment_image_only": True,
        }
        return datapoint


class DeploymentImageOnlyDataset(Dataset):
    """Remove every annotation-derived model input from a dataset split.

    Dataset metadata remains out-of-band so legacy evaluation output (including
    optional GT-mask export) stays compatible.  It is never attached to the
    SAM3 ``BatchedDatapoint`` or used as a routing target.
    """

    def __init__(self, dataset: Dataset, task_text: str) -> None:
        self.dataset = dataset
        self.task_text = _validate_task_text(task_text)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Datapoint:
        annotated = self.dataset[index]
        if not isinstance(annotated, Datapoint):
            raise TypeError(
                "Dataset-split inference requires dataset.return_format='sam3', "
                f"got {type(annotated).__name__}"
            )
        if len(annotated.images) != 1:
            raise ValueError(
                "Image inference requires exactly one image per Datapoint, "
                f"got {len(annotated.images)}"
            )
        if not annotated.find_queries:
            raise ValueError("SAM3 Datapoint must contain at least one find query")

        source_query = annotated.find_queries[0]
        original_size = annotated.images[0].size
        if source_query.inference_metadata is not None:
            original_size = tuple(source_query.inference_metadata.original_size)
        clean = _image_only_datapoint(
            annotated.images[0],
            task_text=self.task_text,
            sample_index=index,
            original_size=original_size,
        )
        metadata = dict(getattr(annotated, "patient_metadata", None) or {})
        metadata.update(
            {
                "task_text": self.task_text,
                "prompt_mode": "image_only",
                "applied_prompt_text": self.task_text,
                "applied_box_prompt": [],
                "uses_external_bbox": False,
                "deployment_image_only": True,
            }
        )
        clean.patient_metadata = metadata
        return clean


def discover_image_paths(
    image: Optional[str] = None,
    image_dir: Optional[str] = None,
    maximum: Optional[int] = None,
) -> list[Path]:
    """Resolve deterministic raw-image inputs without consulting a dataset."""
    if image and image_dir:
        raise ValueError("--image and --image-dir are mutually exclusive")
    if maximum is not None and int(maximum) <= 0:
        raise ValueError("--max-samples must be positive when provided")
    if image:
        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(f"Input image not found: {path}")
        if path.suffix.lower() not in _IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported input image extension: {path.suffix}")
        paths = [path]
    elif image_dir:
        root = Path(image_dir)
        if not root.is_dir():
            raise NotADirectoryError(f"Input image directory not found: {root}")
        paths = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        )
        if not paths:
            raise FileNotFoundError(f"No supported image slices found below: {root}")
    else:
        raise ValueError("Either --image or --image-dir is required")
    if maximum is not None:
        paths = paths[: int(maximum)]
    return paths


def _image_only_collate(batch: Sequence[Datapoint], task_text: str):
    metadata = [getattr(item, "patient_metadata", None) for item in batch]
    output = collate_fn_api(list(batch), dict_key="input", with_seg_masks=False)
    output["_patient_metadata"] = metadata
    output["_task_text"] = _validate_task_text(task_text)
    assert_image_only_batch(output["input"], output["_task_text"], metadata)
    return output


def assert_image_only_batch(
    input_batch: Any,
    task_text: str,
    metadata: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
) -> None:
    """Fail closed if a GT/geometry signal reaches deployment inference."""
    task_text = _validate_task_text(task_text)
    if len(input_batch.find_inputs) != 1 or len(input_batch.find_targets) != 1:
        raise RuntimeError("Deployment inference requires exactly one SAM3 find stage")
    if input_batch.find_text_batch != [task_text]:
        raise RuntimeError(
            "Deployment inference must use one fixed task text; got "
            f"{input_batch.find_text_batch!r}"
        )
    find_input = input_batch.find_inputs[0]
    if find_input.input_boxes.numel() or find_input.input_boxes_mask.numel():
        raise RuntimeError(
            "External bounding-box prompts are forbidden in image-only inference"
        )
    if find_input.input_points is not None and find_input.input_points.numel():
        raise RuntimeError("Point prompts are forbidden in image-only inference")
    target = input_batch.find_targets[0]
    if target.num_boxes.numel() and int(target.num_boxes.sum().item()) != 0:
        raise RuntimeError("GT target boxes leaked into deployment inference")
    if target.boxes.numel() or target.object_ids.numel():
        raise RuntimeError("GT objects leaked into deployment inference")
    if target.segments is not None or target.is_valid_segment is not None:
        raise RuntimeError("GT masks must not be collated for deployment inference")
    for sample_metadata in metadata or ():
        if sample_metadata and sample_metadata.get("uses_external_bbox"):
            raise RuntimeError("Metadata declares an external bbox in image-only inference")


def _deployment_model(trainer: SAM3TrainerNative):
    model = getattr(trainer, "_unwrapped_model", None)
    return model if model is not None else trainer.model


def force_zero_interactive_steps(trainer: SAM3TrainerNative) -> None:
    """Disable validation-time GT feedback and verify that it stayed disabled."""
    model = _deployment_model(trainer)
    if not hasattr(model, "num_interactive_steps_val"):
        raise AttributeError("SAM3 model has no num_interactive_steps_val safety control")
    model.num_interactive_steps_val = 0
    assert_zero_interactive_steps(trainer)


def assert_zero_interactive_steps(trainer: SAM3TrainerNative) -> None:
    steps = int(getattr(_deployment_model(trainer), "num_interactive_steps_val", -1))
    if steps != 0:
        raise RuntimeError(
            "Image-only inference requires num_interactive_steps_val == 0; "
            f"got {steps}"
        )


def build_runtime(config_path: str, checkpoint: str, stage: int | None, device: int):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    selected_stage = int(stage or config.get("training", {}).get("stage", 5))
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    moe_config = resolve_moe_runtime_config(config)
    router_config = config.get("router") or {}
    trainer = SAM3TrainerNative(
        config_path,
        moe_config=moe_config,
        patient_dataset_config=config.get("dataset"),
        router_config=router_config,
        training_stage=selected_stage,
        svanet_config=config.get("svanet"),
        load_stage_dependencies=False,
        load_training_resume=False,
    )
    trainer.stage_manager.load_checkpoint(
        checkpoint, allowed_stages={selected_stage}
    )
    trainer.stage_manager.set_module_modes(training=False)
    force_zero_interactive_steps(trainer)
    return trainer, config, selected_stage


def build_loader(
    trainer: SAM3TrainerNative,
    config: Dict[str, Any],
    split: str,
    maximum,
    task_text: str = "prostate",
):
    """Build the backward-compatible annotated-split loader, sanitized for deployment."""
    del config  # Dataset configuration is already owned by the trainer.
    if maximum is not None and int(maximum) <= 0:
        raise ValueError("--max-samples must be positive when provided")
    dataset = trainer._build_patient_dataset(split, training=False)
    dataset = DeploymentImageOnlyDataset(dataset, task_text)
    if maximum is not None:
        dataset = Subset(dataset, range(min(int(maximum), len(dataset))))

    def collate(batch):
        return _image_only_collate(batch, task_text)

    return DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate
    )


def build_image_loader(
    image_paths: Sequence[str | Path],
    task_text: str,
) -> DataLoader:
    """Build a loader which has no dependency on COCO/PatientDataset files."""
    dataset = ImageOnlySliceDataset(image_paths, task_text)

    def collate(batch):
        return _image_only_collate(batch, task_text)

    return DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate
    )


def run_batch(trainer: SAM3TrainerNative, batch: Dict[str, Any]):
    assert_zero_interactive_steps(trainer)
    assert_image_only_batch(
        batch["input"],
        batch.get("_task_text", "prostate"),
        batch.get("_patient_metadata"),
    )
    input_batch = move_to_device(batch["input"], trainer.device)
    # Pure deployment path: routing receives image features only, never GT
    # modality/area/boundary metadata.
    trainer.moe_controller.set_routing_targets(None)
    with torch.no_grad():
        outputs = trainer.model(input_batch)
        final_output = outputs.output[-1][-1]
        base_logits, query_indices = select_best_mask_logits(final_output)
        routes = trainer.moe_controller.current_routes
        if routes is None:
            raise RuntimeError("MoE routes are unavailable after SAM3 forward")
        find_input = input_batch.find_inputs[-1]
        image_indices = find_input.img_ids.long()
        images = input_batch.img_batch[image_indices]
        adapter_output = None
        final_logits = base_logits
        if trainer.svanet_adapter is not None:
            area_logits = routes.get("area_selected_logits")
            if area_logits is None:
                area_logits = routes["area_logits"]
            adapter_output = trainer.svanet_adapter(
                images=images,
                sam3_logits=base_logits,
                area_logits=area_logits,
                box_prompts=normalized_xyxy_prompts(find_input),
                locator_logits=routes.get("image_locator_logits"),
            )
            final_logits = adapter_output["final_logits"]
    metadata = batch.get("_patient_metadata") or []
    if len(metadata) != len(final_logits):
        raise RuntimeError(
            "One prediction per input slice is required, but received "
            f"{len(final_logits)} predictions for {len(metadata)} metadata records"
        )
    return {
        "input_batch": input_batch,
        "model_output": final_output,
        "base_logits": base_logits,
        "final_logits": final_logits,
        "query_indices": query_indices,
        "routes": routes,
        "adapter_output": adapter_output,
        "metadata": metadata,
    }


def _mask_image(logits: torch.Tensor, size: tuple[int, int]) -> PILImage.Image:
    resized = F.interpolate(
        logits[None, None].float(), size=(size[1], size[0]),
        mode="bilinear", align_corners=False,
    )[0, 0]
    array = (resized.sigmoid().cpu().numpy() >= 0.5).astype(np.uint8) * 255
    return PILImage.fromarray(array, mode="L")


def _overlay_box(image: PILImage.Image, box, color="red") -> PILImage.Image:
    output = image.convert("RGB").copy()
    if box is not None:
        ImageDraw.Draw(output).rectangle(tuple(box), outline=color, width=2)
    return output


def _safe_output_name(value: Any) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return value or "slice"


def save_batch(result: Dict[str, Any], output_root: Path):
    records = []
    adapter = result["adapter_output"]
    trigger_mask = (
        adapter["trigger_mask"].detach().cpu().tolist()
        if adapter is not None else [False] * len(result["metadata"])
    )
    trigger_position = 0
    for index, metadata in enumerate(result["metadata"]):
        metadata = dict(metadata or {})
        route = route_predictions(result["routes"], index)
        if all(key in metadata for key in ("modality", "patient_id", "slice_id")):
            # Preserve the established annotated-dataset directory convention.
            name = (
                f"{metadata['modality']}_{metadata['patient_id']}_"
                f"{metadata['slice_id']}"
            )
        else:
            name = (
                metadata.get("sample_id")
                or metadata.get("slice_id")
                or f"slice_{index:06d}"
            )
        name = _safe_output_name(name)
        sample_dir = output_root / name
        sample_dir.mkdir(parents=True, exist_ok=True)
        image_path = metadata.get("image_path")
        if not image_path:
            raise KeyError(
                "Inference metadata must contain image_path for output saving"
            )
        with PILImage.open(image_path) as source:
            original = source.convert("RGB")
        original.save(sample_dir / "original_image.png")
        if metadata.get("mask_path"):
            PILImage.open(metadata["mask_path"]).convert("L").save(sample_dir / "gt_mask.png")
        base_mask = _mask_image(result["base_logits"][index], original.size)
        final_mask = _mask_image(result["final_logits"][index], original.size)
        base_mask.save(sample_dir / "sam3_mask.png")
        final_mask.save(sample_dir / "final_mask.png")
        sam_box = largest_component_bbox(
            result["base_logits"][index].sigmoid() >= 0.5
        )
        if sam_box is not None:
            scale_x = original.width / result["base_logits"].shape[-1]
            scale_y = original.height / result["base_logits"].shape[-2]
            sam_box_original = (
                sam_box[0] * scale_x, sam_box[1] * scale_y,
                sam_box[2] * scale_x, sam_box[3] * scale_y,
            )
        else:
            sam_box_original = None
        _overlay_box(original, sam_box_original).save(sample_dir / "sam3_bbox.png")

        roi_box = None
        if adapter is not None and trigger_mask[index]:
            roi_box = adapter["roi_boxes"][trigger_position]
            x1, y1, x2, y2 = roi_box
            scale_x = original.width / result["final_logits"].shape[-1]
            scale_y = original.height / result["final_logits"].shape[-2]
            original_roi = (
                int(x1 * scale_x), int(y1 * scale_y),
                int(x2 * scale_x), int(y2 * scale_y),
            )
            _overlay_box(original, original_roi, "yellow").save(sample_dir / "expanded_roi.png")
            original.crop(original_roi).save(sample_dir / "roi_image.png")
            if metadata.get("mask_path"):
                PILImage.open(metadata["mask_path"]).convert("L").crop(original_roi).save(
                    sample_dir / "roi_gt.png"
                )
            roi_logits = adapter["svanet_roi_logits"][trigger_position]
            _mask_image(roi_logits, (original_roi[2] - original_roi[0], original_roi[3] - original_roi[1])).save(
                sample_dir / "svanet_roi_pred.png"
            )
            trigger_position += 1
        record: Dict[str, Any] = {
            "image_path": image_path,
            "slice_id": metadata.get("slice_id", Path(image_path).stem),
            "task_text": metadata.get("task_text", metadata.get("applied_prompt_text")),
            "prompt_mode": "image_only",
            "uses_external_bbox": False,
            **route,
            "selected_query": int(result["query_indices"][index].item()),
            "svanet_triggered": bool(trigger_mask[index]),
            "roi_box_model_coordinates": roi_box,
            "sam3_mask": str(sample_dir / "sam3_mask.png"),
            "final_mask": str(sample_dir / "final_mask.png"),
        }
        if "patient_id" in metadata:
            record["patient_id"] = metadata["patient_id"]
        if "modality" in metadata:
            record["dataset_modality"] = metadata["modality"]
        (sample_dir / "prediction.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        records.append(record)
    return records


def main():
    args = parse_args()
    task_text = _validate_task_text(args.task_text)
    if (args.image or args.image_dir) and args.split is not None:
        raise ValueError("--split cannot be combined with --image or --image-dir")
    trainer, config, stage = build_runtime(
        args.config, args.checkpoint, args.stage, args.device
    )
    if args.image or args.image_dir:
        paths = discover_image_paths(
            image=args.image,
            image_dir=args.image_dir,
            maximum=args.max_samples,
        )
        loader = build_image_loader(paths, task_text)
        input_mode = "raw_image"
        split = None
    else:
        split = args.split or config.get("dataset", {}).get("test_split", "test")
        loader = build_loader(
            trainer, config, split, args.max_samples, task_text=task_text
        )
        input_mode = "dataset_split"
    output_root = Path(args.output_dir)
    all_records = []
    for batch in loader:
        all_records.extend(save_batch(run_batch(trainer, batch), output_root))
    summary = {
        "stage": stage,
        "checkpoint": args.checkpoint,
        "input_mode": input_mode,
        "split": split,
        "task_text": task_text,
        "uses_external_bbox": False,
        "num_interactive_steps_val": 0,
        "predictions": all_records,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "predictions.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(all_records)} predictions to {output_root.resolve()}")


if __name__ == "__main__":
    main()
