#!/usr/bin/env python3
"""Inference entry for Hierarchical LoRA-MoE-SAM3 with optional SvANet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from PIL import Image as PILImage, ImageDraw
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from models.inference_utils import (
    move_to_device, normalized_xyxy_prompts, route_predictions,
    select_best_mask_logits,
)
from models.svanet_roi_adapter import largest_component_bbox
from sam3.train.data.collator import collate_fn_api
from train_sam3_lora_native import SAM3TrainerNative


def parse_args():
    parser = argparse.ArgumentParser(description="Infer Hierarchical LoRA-MoE-SAM3")
    parser.add_argument("--config", default="configs/moe_sam3.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage", type=int, choices=range(1, 6), default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--output-dir", default="outputs/moe_inference")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def build_runtime(config_path: str, checkpoint: str, stage: int | None, device: int):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    selected_stage = int(stage or config.get("training", {}).get("stage", 5))
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    moe_config = dict(config.get("moe") or {})
    router_config = config.get("router") or {}
    if "routing_type" in router_config:
        moe_config["routing_mode"] = str(router_config["routing_type"]).lower()
    trainer = SAM3TrainerNative(
        config_path,
        moe_config=moe_config,
        patient_dataset_config=config.get("dataset"),
        router_config=router_config,
        training_stage=selected_stage,
        svanet_config=config.get("svanet"),
        load_stage_dependencies=False,
    )
    trainer.stage_manager.load_checkpoint(
        checkpoint, allowed_stages={selected_stage}
    )
    trainer.stage_manager.set_module_modes(training=False)
    return trainer, config, selected_stage


def build_loader(trainer: SAM3TrainerNative, config: Dict[str, Any], split: str, maximum):
    dataset = trainer._build_patient_dataset(split, training=False)
    if maximum is not None:
        dataset = Subset(dataset, range(min(int(maximum), len(dataset))))

    def collate(batch):
        metadata = [getattr(item, "patient_metadata", None) for item in batch]
        output = collate_fn_api(batch, dict_key="input", with_seg_masks=True)
        output["_patient_metadata"] = metadata
        return output

    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)


def run_batch(trainer: SAM3TrainerNative, batch: Dict[str, Any]):
    input_batch = move_to_device(batch["input"], trainer.device)
    trainer._set_moe_routing_targets(
        batch.get("_patient_metadata"), input_batch, epoch=0, training=False
    )
    with torch.no_grad():
        outputs = trainer.model(input_batch)
        final_output = outputs.output[-1][-1]
        base_logits, query_indices = select_best_mask_logits(final_output)
        routes = trainer.moe_controller.current_routes
        find_input = input_batch.find_inputs[-1]
        image_indices = find_input.img_ids.long()
        images = input_batch.img_batch[image_indices]
        adapter_output = None
        final_logits = base_logits
        if trainer.svanet_adapter is not None:
            adapter_output = trainer.svanet_adapter(
                images=images,
                sam3_logits=base_logits,
                area_logits=routes["area_logits"],
                box_prompts=normalized_xyxy_prompts(find_input),
            )
            final_logits = adapter_output["final_logits"]
    return {
        "input_batch": input_batch,
        "model_output": final_output,
        "base_logits": base_logits,
        "final_logits": final_logits,
        "query_indices": query_indices,
        "routes": routes,
        "adapter_output": adapter_output,
        "metadata": batch.get("_patient_metadata") or [],
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


def save_batch(result: Dict[str, Any], output_root: Path):
    records = []
    adapter = result["adapter_output"]
    trigger_mask = (
        adapter["trigger_mask"].detach().cpu().tolist()
        if adapter is not None else [False] * len(result["metadata"])
    )
    trigger_position = 0
    for index, metadata in enumerate(result["metadata"]):
        route = route_predictions(result["routes"], index)
        name = f"{metadata['modality']}_{metadata['patient_id']}_{metadata['slice_id']}"
        sample_dir = output_root / name
        sample_dir.mkdir(parents=True, exist_ok=True)
        original = PILImage.open(metadata["image_path"]).convert("RGB")
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
        record = {
            "image_path": metadata["image_path"],
            "patient_id": metadata["patient_id"],
            "slice_id": metadata["slice_id"],
            **route,
            "selected_query": int(result["query_indices"][index].item()),
            "svanet_triggered": bool(trigger_mask[index]),
            "roi_box_model_coordinates": roi_box,
            "sam3_mask": str(sample_dir / "sam3_mask.png"),
            "final_mask": str(sample_dir / "final_mask.png"),
        }
        (sample_dir / "prediction.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        records.append(record)
    return records


def main():
    args = parse_args()
    trainer, config, stage = build_runtime(
        args.config, args.checkpoint, args.stage, args.device
    )
    split = args.split or config.get("dataset", {}).get("test_split", "test")
    loader = build_loader(trainer, config, split, args.max_samples)
    output_root = Path(args.output_dir)
    all_records = []
    for batch in loader:
        all_records.extend(save_batch(run_batch(trainer, batch), output_root))
    summary = {"stage": stage, "checkpoint": args.checkpoint, "predictions": all_records}
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "predictions.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(all_records)} predictions to {output_root.resolve()}")


if __name__ == "__main__":
    main()
