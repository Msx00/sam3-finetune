
#!/usr/bin/env python3
"""
SAM3 LoRA Training Script

Validation Strategy (Following SAM3):
  - During training: Only compute validation LOSS (fast, no metrics)
  - After training: Run validate_sam3_lora.py for full metrics (mAP, cgF1) with NMS

This approach significantly speeds up training by avoiding expensive metric computation
during each epoch, while still monitoring overfitting via validation loss.

Multi-GPU Training:
  Single GPU:
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml

  Multi-GPU (DDP):
    torchrun --nproc_per_node=2 train_sam3_lora_native.py --config configs/full_lora_config.yaml --multi-gpu

  Multi-GPU with specific GPUs:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_sam3_lora_native.py --config configs/full_lora_config.yaml --multi-gpu
"""

import os
import argparse
import random
import yaml
import json
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
from pathlib import Path
import numpy as np
from PIL import Image as PILImage
import contextlib

# Distributed training imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# SAM3 Imports
from sam3.model_builder import build_sam3_image_model
from sam3.model.model_misc import SAM3Output
from sam3.train.loss.loss_fns import IABCEMdetr, Boxes, Masks, CORE_LOSS_KEY
from sam3.train.loss.sam3_loss import Sam3LossWrapper
from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import Datapoint, Image, Object, FindQueryLoaded, InferenceMetadata
from sam3.model.box_ops import box_xywh_to_xyxy
from lora_layers import LoRAConfig, apply_lora_to_model, save_lora_weights, count_parameters

from torchvision.transforms import v2
import pycocotools.mask as mask_utils  # Required for RLE mask decoding in COCO dataset
from sam3.train.masks_ops import rle_encode  # For encoding masks to RLE format

# Note: Evaluation modules (mAP, cgF1, NMS) are in validate_sam3_lora.py
# Training only computes validation loss, following SAM3's approach


# ============================================================================
# Distributed Training Utilities
# ============================================================================

def setup_distributed():
    """Initialize distributed training environment."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    return local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size():
    """Get the number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """Get the rank of current process."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def print_rank0(*args, **kwargs):
    """Print only on rank 0."""
    if is_main_process():
        print(*args, **kwargs)


class ResumableRandomBatchSampler:
    """Deterministic shuffled batches that can start at an epoch batch offset."""

    def __init__(self, dataset_size, batch_size, seed):
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.start_batch = 0
        if self.dataset_size < 0:
            raise ValueError("dataset_size must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    @property
    def total_batches(self):
        return (self.dataset_size + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch, start_batch=0):
        start_batch = int(start_batch)
        if start_batch < 0 or start_batch > self.total_batches:
            raise ValueError(
                f"start_batch={start_batch} is outside [0, {self.total_batches}]"
            )
        self.epoch = int(epoch)
        self.start_batch = start_batch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(
            self.dataset_size, generator=generator
        ).tolist()
        for offset in range(
            self.start_batch * self.batch_size,
            self.dataset_size,
            self.batch_size,
        ):
            yield indices[offset:offset + self.batch_size]

    def __len__(self):
        return self.total_batches - self.start_batch


class COCOSegmentDataset(Dataset):
    """Dataset class for COCO format segmentation data"""
    def __init__(self, data_dir, split="train", annotation_file=None):
        """
        Args:
            data_dir: Root directory containing train/valid/test folders
            split: One of 'train', 'valid', 'test'
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.split_dir = self.data_dir / split

        # Load COCO annotations
        ann_file = Path(annotation_file) if annotation_file else (
            self.split_dir / "_annotations.coco.json"
        )
        if annotation_file and not ann_file.is_absolute():
            ann_file = self.data_dir / ann_file
        if not ann_file.exists():
            raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")

        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)

        # Build index: image_id -> image info
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.image_ids = sorted(list(self.images.keys()))

        # Build index: image_id -> list of annotations
        self.img_to_anns = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        # Load categories
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        print(f"Loaded COCO dataset: {split} split")
        print(f"  Images: {len(self.image_ids)}")
        print(f"  Annotations: {len(self.coco_data['annotations'])}")
        print(f"  Categories: {self.categories}")

        self.resolution = 1008
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]

        # Load image
        from data.sample_index import resolve_sample_path

        img_path = resolve_sample_path(
            self.split_dir,
            img_info['file_name'],
            self.split,
            strict=True,
        )
        pil_image = PILImage.open(img_path).convert("RGB")
        orig_w, orig_h = pil_image.size

        # Resize image
        pil_image = pil_image.resize((self.resolution, self.resolution), PILImage.BILINEAR)

        # Transform to tensor
        image_tensor = self.transform(pil_image)

        # Get annotations for this image
        annotations = self.img_to_anns.get(img_id, [])

        objects = []
        object_class_names = []

        # Scale factors
        scale_w = self.resolution / orig_w
        scale_h = self.resolution / orig_h

        for i, ann in enumerate(annotations):
            # Get bbox - format is [x, y, width, height] in COCO format
            bbox_coco = ann.get("bbox", None)
            if bbox_coco is None:
                continue

            # Get class name from category_id
            category_id = ann.get("category_id", 0)
            class_name = self.categories.get(category_id, "object")
            object_class_names.append(class_name)

            # Convert from COCO [x, y, w, h] to normalized [cx, cy, w, h] (CxCyWH)
            # SAM3 internally expects boxes in CxCyWH format normalized to [0, 1]
            x, y, w, h = bbox_coco
            cx = x + w / 2.0
            cy = y + h / 2.0

            # Scale to resolution and normalize to [0, 1]
            box_tensor = torch.tensor([
                cx * scale_w / self.resolution,
                cy * scale_h / self.resolution,
                w * scale_w / self.resolution,
                h * scale_h / self.resolution,
            ], dtype=torch.float32)

            # Handle segmentation mask (polygon or RLE format)
            segment = None
            segmentation = ann.get("segmentation", None)

            if segmentation:
                try:
                    # Check if it's RLE format (dict) or polygon format (list)
                    if isinstance(segmentation, dict):
                        # RLE format: {"counts": "...", "size": [h, w]}
                        mask_np = mask_utils.decode(segmentation)
                    elif isinstance(segmentation, list):
                        # Polygon format: [[x1, y1, x2, y2, ...], ...]
                        # Convert polygon to RLE, then decode
                        rles = mask_utils.frPyObjects(segmentation, orig_h, orig_w)
                        rle = mask_utils.merge(rles)
                        mask_np = mask_utils.decode(rle)
                    else:
                        print(f"Warning: Unknown segmentation format: {type(segmentation)}")
                        segment = None
                        continue

                    # Resize mask to model resolution
                    mask_t = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
                    mask_t = torch.nn.functional.interpolate(
                        mask_t,
                        size=(self.resolution, self.resolution),
                        mode="nearest"
                    )
                    segment = mask_t.squeeze() > 0.5  # [1008, 1008] boolean tensor

                except Exception as e:
                    print(f"Warning: Error processing mask for image {img_id}, ann {i}: {e}")
                    segment = None

            obj = Object(
                bbox=box_tensor,
                area=(box_tensor[2] * box_tensor[3]).item(),
                object_id=i,
                segment=segment
            )
            objects.append(obj)

        image_obj = Image(
            data=image_tensor,
            objects=objects,
            size=(self.resolution, self.resolution)
        )

        # Construct Queries - one per unique category
        # Each query maps to only the objects of that category
        from collections import defaultdict

        # Group object IDs by their class name
        class_to_object_ids = defaultdict(list)
        for obj, class_name in zip(objects, object_class_names):
            class_to_object_ids[class_name.lower()].append(obj.object_id)

        # Create one query per category
        queries = []
        if len(class_to_object_ids) > 0:
            for query_text, obj_ids in class_to_object_ids.items():
                query = FindQueryLoaded(
                    query_text=query_text,
                    image_id=0,
                    object_ids_output=obj_ids,
                    is_exhaustive=True,
                    query_processing_order=0,
                    inference_metadata=InferenceMetadata(
                        coco_image_id=img_id,
                        original_image_id=img_id,
                        original_category_id=0,
                        original_size=(orig_h, orig_w),
                        object_id=-1,
                        frame_index=-1
                    )
                )
                queries.append(query)
        else:
            # No annotations: create a single generic query
            query = FindQueryLoaded(
                query_text="object",
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=img_id,
                    original_image_id=img_id,
                    original_category_id=0,
                    original_size=(orig_h, orig_w),
                    object_id=-1,
                    frame_index=-1
                )
            )
            queries.append(query)

        return Datapoint(
            find_queries=queries,
            images=[image_obj],
            raw_images=[pil_image]
        )


def merge_overlapping_masks(binary_masks, scores, boxes, iou_threshold=0.3):
    """
    Merge overlapping masks that likely represent the same object.

    Args:
        binary_masks: Binary masks [N, H, W]
        scores: Confidence scores [N]
        boxes: Bounding boxes [N, 4]
        iou_threshold: IoU threshold for merging (default: 0.3)

    Returns:
        Tuple of (merged_masks, merged_scores, merged_boxes)
    """
    if len(binary_masks) == 0:
        return binary_masks, scores, boxes

    # Sort by score (highest first)
    sorted_indices = torch.argsort(scores, descending=True)
    binary_masks = binary_masks[sorted_indices]
    scores = scores[sorted_indices]
    boxes = boxes[sorted_indices]

    merged_masks = []
    merged_scores = []
    merged_boxes = []
    used = torch.zeros(len(binary_masks), dtype=torch.bool)

    for i in range(len(binary_masks)):
        if used[i]:
            continue

        current_mask = binary_masks[i].clone()
        current_score = scores[i].item()
        current_box = boxes[i]
        used[i] = True

        # Find overlapping masks and merge them
        for j in range(i + 1, len(binary_masks)):
            if used[j]:
                continue

            # Compute IoU
            intersection = (current_mask & binary_masks[j]).sum().item()
            union = (current_mask | binary_masks[j]).sum().item()
            iou = intersection / union if union > 0 else 0

            # If overlaps significantly, merge it
            if iou > iou_threshold:
                current_mask = current_mask | binary_masks[j]
                current_score = max(current_score, scores[j].item())
                used[j] = True

        merged_masks.append(current_mask)
        merged_scores.append(current_score)
        merged_boxes.append(current_box)

    if len(merged_masks) > 0:
        merged_masks = torch.stack(merged_masks)
        merged_scores = torch.tensor(merged_scores, device=scores.device)
        merged_boxes = torch.stack(merged_boxes)
    else:
        merged_masks = binary_masks[:0]
        merged_scores = scores[:0]
        merged_boxes = boxes[:0]

    return merged_masks, merged_scores, merged_boxes


def convert_predictions_to_coco_format(predictions_list, image_ids, resolution=288, score_threshold=0.0, merge_overlaps=True, iou_threshold=0.3, debug=False):
    """
    Convert model predictions to COCO format for evaluation.

    OPTIMIZATION: Keep masks at native model output resolution (288×288)
    GT is downsampled to match, so no upsampling needed!

    Args:
        predictions_list: List of prediction dictionaries from the model
        image_ids: List of image IDs corresponding to predictions
        resolution: Mask resolution for evaluation (default: 288, model's native output)
        score_threshold: Minimum score threshold for predictions
        merge_overlaps: Whether to merge overlapping predictions (default: True)
        iou_threshold: IoU threshold for merging overlaps (default: 0.3)
        debug: Print debug information

    Returns:
        List of prediction dictionaries in COCO format
    """
    coco_predictions = []
    pred_id = 0

    for img_id, preds in zip(image_ids, predictions_list):
        if preds is None or len(preds.get('pred_logits', [])) == 0:
            continue

        # Extract predictions
        logits = preds['pred_logits']  # [num_queries, 1]
        boxes = preds['pred_boxes']    # [num_queries, 4]
        masks = preds['pred_masks']    # [num_queries, H, W]

        scores = torch.sigmoid(logits).squeeze(-1)  # [num_queries]

        # Filter by score threshold
        valid_mask = scores > score_threshold
        num_before = len(scores)
        scores = scores[valid_mask]
        boxes = boxes[valid_mask]
        masks = masks[valid_mask]

        if debug and img_id == image_ids[0]:  # Debug first image only
            print(f"  Image {img_id}: {num_before} queries -> {len(scores)} after filtering (threshold={score_threshold})")

        # Convert masks to binary (apply sigmoid first, then threshold)
        binary_masks = (torch.sigmoid(masks) > 0.5).cpu()

        # Merge overlapping predictions to avoid over-segmentation penalty
        if merge_overlaps and len(binary_masks) > 0:
            num_before_merge = len(binary_masks)
            binary_masks, scores, boxes = merge_overlapping_masks(
                binary_masks, scores.cpu(), boxes.cpu(), iou_threshold=iou_threshold
            )
            if debug and img_id == image_ids[0]:
                print(f"  Merged {num_before_merge} predictions -> {len(binary_masks)} (IoU threshold={iou_threshold})")

        # Encode masks to RLE (at native resolution - much faster!)
        if len(binary_masks) > 0:
            # Check if masks have content
            mask_areas = binary_masks.flatten(1).sum(1)

            if debug and img_id == image_ids[0]:
                print(f"  Mask shape: {binary_masks.shape}")
                print(f"  Mask areas: min={mask_areas.min():.0f}, max={mask_areas.max():.0f}, mean={mask_areas.float().mean():.0f}")

            rles = rle_encode(binary_masks)

            for idx, (rle, score, box) in enumerate(zip(rles, scores.cpu().tolist(), boxes.cpu().tolist())):
                # Convert box from normalized [cx, cy, w, h] to [x, y, w, h] in pixel coordinates
                cx, cy, w, h = box
                x = (cx - w/2) * resolution
                y = (cy - h/2) * resolution
                w = w * resolution
                h = h * resolution

                coco_predictions.append({
                    'image_id': int(img_id),
                    'category_id': 1,  # Single category for instance segmentation
                    'segmentation': rle,
                    'bbox': [float(x), float(y), float(w), float(h)],
                    'score': float(score),
                    'id': pred_id
                })
                pred_id += 1

    return coco_predictions


def create_coco_gt_from_dataset(dataset, image_ids=None, mask_resolution=288):
    """
    Create COCO ground truth dictionary from SimpleSAM3Dataset.

    OPTIMIZATION: Downsample GT masks to match prediction resolution (288×288)
    instead of upsampling predictions to 1008×1008. Much faster!

    Args:
        dataset: SimpleSAM3Dataset instance
        image_ids: Optional list of specific image IDs to include
        mask_resolution: Resolution to downsample masks to (default: 288 to match model output)

    Returns:
        Dictionary in COCO format
    """
    coco_gt = {
        'info': {
            'description': 'SAM3 LoRA Validation Dataset',
            'version': '1.0',
            'year': 2024
        },
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'object'}]
    }

    ann_id = 0
    indices = range(len(dataset)) if image_ids is None else image_ids

    # Scale factor for boxes (masks will be at mask_resolution, boxes scaled accordingly)
    scale_factor = mask_resolution / dataset.resolution

    for idx in indices:
        # Add image entry at mask resolution
        coco_gt['images'].append({
            'id': int(idx),
            'width': mask_resolution,
            'height': mask_resolution,
            'is_instance_exhaustive': True  # Required for cgF1 evaluation
        })

        # Get datapoint
        datapoint = dataset[idx]

        # Add annotations
        for obj in datapoint.images[0].objects:
            # Convert normalized CxCyWH box to COCO [x, y, w, h] at mask_resolution
            cx, cy, bw, bh = (obj.bbox * mask_resolution).tolist()
            x, y, w, h = cx - bw / 2, cy - bh / 2, bw, bh

            ann = {
                'id': ann_id,
                'image_id': int(idx),
                'category_id': 1,
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0,
                'ignore': 0
            }

            # Add segmentation if available - downsample to mask_resolution
            if obj.segment is not None:
                # Downsample mask from 1008×1008 to mask_resolution×mask_resolution
                mask_tensor = obj.segment.unsqueeze(0).unsqueeze(0).float()
                downsampled_mask = torch.nn.functional.interpolate(
                    mask_tensor,
                    size=(mask_resolution, mask_resolution),
                    mode='bilinear',
                    align_corners=False
                ) > 0.5

                mask_np = downsampled_mask.squeeze().cpu().numpy().astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(mask_np))
                rle['counts'] = rle['counts'].decode('utf-8')
                ann['segmentation'] = rle

            coco_gt['annotations'].append(ann)
            ann_id += 1

    return coco_gt


def convert_predictions_to_coco_format_original_res(predictions_list, image_ids, dataset, model_resolution=288, score_threshold=0.0, merge_overlaps=True, iou_threshold=0.3, debug=False):
    """
    Convert model predictions to COCO format at ORIGINAL image resolution.

    This matches the inference approach (infer_sam.py) where:
    1. Masks are upsampled from 288x288 to original image size
    2. Boxes are scaled to original image size
    3. Evaluation happens at original resolution

    Args:
        predictions_list: List of predictions per image
        image_ids: List of image IDs (indices into dataset)
        dataset: Dataset to get original image sizes
        model_resolution: Model output resolution (default: 288)
        score_threshold: Confidence threshold
        merge_overlaps: Whether to merge overlapping predictions
        iou_threshold: IoU threshold for merging
        debug: Print debug info
    """
    coco_predictions = []
    pred_id = 0

    if debug:
        print(f"\n[DEBUG] Converting {len(predictions_list)} predictions to COCO format (ORIGINAL RESOLUTION)...")
        if merge_overlaps:
            print(f"[DEBUG] Overlapping segment merging ENABLED (IoU threshold={iou_threshold})")

    for img_id, preds in zip(image_ids, predictions_list):
        if preds is None or len(preds.get('pred_logits', [])) == 0:
            continue

        # Get original image size from dataset
        datapoint = dataset[img_id]
        orig_h, orig_w = datapoint.find_queries[0].inference_metadata.original_size

        logits = preds['pred_logits']
        boxes = preds['pred_boxes']
        masks = preds['pred_masks']  # [N, 288, 288]

        scores = torch.sigmoid(logits).squeeze(-1)

        # Filter by score threshold
        valid_mask = scores > score_threshold
        num_before = len(scores)
        scores = scores[valid_mask]
        boxes = boxes[valid_mask]
        masks = masks[valid_mask]

        if debug and img_id == image_ids[0]:
            print(f"[DEBUG] Image {img_id}: {num_before} queries -> {len(scores)} after filtering (threshold={score_threshold})")
            if len(scores) > 0:
                print(f"[DEBUG]   Original size: {orig_w}x{orig_h}")
                print(f"[DEBUG]   Filtered scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")

        if len(masks) == 0:
            continue

        # Upsample masks from 288x288 to original resolution (like infer_sam.py)
        # Process on GPU then immediately move to CPU to save memory
        masks_sigmoid = torch.sigmoid(masks)  # [N, 288, 288]
        masks_upsampled = torch.nn.functional.interpolate(
            masks_sigmoid.unsqueeze(1).float(),  # [N, 1, 288, 288]
            size=(orig_h, orig_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # [N, orig_h, orig_w]

        binary_masks = (masks_upsampled > 0.5).cpu()

        # Free GPU memory immediately after upsampling
        del masks_sigmoid, masks_upsampled
        torch.cuda.empty_cache()

        # Merge overlapping predictions
        if merge_overlaps and len(binary_masks) > 0:
            num_before_merge = len(binary_masks)
            binary_masks, scores, boxes = merge_overlapping_masks(
                binary_masks, scores.cpu(), boxes.cpu(), iou_threshold=iou_threshold
            )
            if debug and img_id == image_ids[0]:
                print(f"[DEBUG]   Merged {num_before_merge} predictions -> {len(binary_masks)} (IoU threshold={iou_threshold})")

        if len(binary_masks) > 0:
            mask_areas = binary_masks.flatten(1).sum(1)

            if debug and img_id == image_ids[0]:
                print(f"[DEBUG]   Upsampled mask shape: {binary_masks.shape}")
                print(f"[DEBUG]   Mask areas: min={mask_areas.min():.0f}, max={mask_areas.max():.0f}, mean={mask_areas.float().mean():.0f}")

            rles = rle_encode(binary_masks)

            for idx, (rle, score, box) in enumerate(zip(rles, scores.cpu().tolist(), boxes.cpu().tolist())):
                # Convert box from normalized [0,1] to original image coordinates
                cx, cy, w_norm, h_norm = box
                x = (cx - w_norm/2) * orig_w
                y = (cy - h_norm/2) * orig_h
                w = w_norm * orig_w
                h = h_norm * orig_h

                # Clamp coordinates to image bounds
                x = max(0, min(x, orig_w))
                y = max(0, min(y, orig_h))
                w = max(0, min(w, orig_w - x))
                h = max(0, min(h, orig_h - y))

                # Skip if box is too small after clamping
                if w < 1 or h < 1:
                    continue

                pred_dict = {
                    'image_id': int(img_id),
                    'category_id': 1,
                    'segmentation': rle,
                    'bbox': [float(x), float(y), float(w), float(h)],
                    'score': float(score),
                    'id': pred_id
                }

                if debug and img_id == image_ids[0] and idx == 0:
                    print(f"[DEBUG]   First prediction bbox (at {orig_w}x{orig_h}): {pred_dict['bbox']}")

                coco_predictions.append(pred_dict)
                pred_id += 1

    return coco_predictions


def create_coco_gt_from_dataset_original_res(dataset, image_ids=None, debug=False):
    """
    Create COCO ground truth dictionary from dataset at ORIGINAL resolution.

    This matches the inference approach (infer_sam.py) where GT is kept
    at original image size for evaluation.

    Args:
        dataset: Dataset with images and annotations
        image_ids: List of image IDs to include (None = all)
        debug: Print debug info
    """
    if debug:
        print(f"\n[DEBUG] Creating COCO ground truth (ORIGINAL RESOLUTION)...")

    coco_gt = {
        'info': {
            'description': 'SAM3 LoRA Validation Dataset',
            'version': '1.0',
            'year': 2024
        },
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'object'}]
    }

    ann_id = 0
    indices = range(len(dataset)) if image_ids is None else image_ids

    for idx in indices:
        datapoint = dataset[idx]

        # Get original image size
        orig_h, orig_w = datapoint.find_queries[0].inference_metadata.original_size

        coco_gt['images'].append({
            'id': int(idx),
            'width': orig_w,
            'height': orig_h,
            'is_instance_exhaustive': True
        })

        for obj in datapoint.images[0].objects:
            # Convert normalized CxCyWH box to COCO [x, y, w, h] at original size
            cx, cy, bw, bh = obj.bbox.tolist()
            w = bw * orig_w
            h = bh * orig_h
            x = cx * orig_w - w / 2
            y = cy * orig_h - h / 2

            ann = {
                'id': ann_id,
                'image_id': int(idx),
                'category_id': 1,
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0,
                'ignore': 0
            }

            if obj.segment is not None:
                # Upsample mask from 1008x1008 to original size
                mask_tensor = obj.segment.unsqueeze(0).unsqueeze(0).float()
                upsampled_mask = torch.nn.functional.interpolate(
                    mask_tensor,
                    size=(orig_h, orig_w),
                    mode='bilinear',
                    align_corners=False
                ) > 0.5

                mask_np = upsampled_mask.squeeze().cpu().numpy().astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(mask_np))
                rle['counts'] = rle['counts'].decode('utf-8')
                ann['segmentation'] = rle

            coco_gt['annotations'].append(ann)
            ann_id += 1

    if debug:
        print(f"[DEBUG] Created {len(coco_gt['images'])} images, {len(coco_gt['annotations'])} annotations")
        if len(coco_gt['annotations']) > 0:
            sample_gt = coco_gt['annotations'][0]
            sample_img = coco_gt['images'][0]
            print(f"[DEBUG] Sample GT: image_id={sample_gt['image_id']}, bbox={sample_gt['bbox']}, image_size={sample_img['width']}x{sample_img['height']}")

    return coco_gt


class SAM3TrainerNative:
    def __init__(
        self,
        config_path,
        multi_gpu=False,
        moe_config=None,
        patient_dataset_config=None,
        router_config=None,
        training_stage=None,
        svanet_config=None,
        resume_path=None,
        load_stage_dependencies=True,
        wandb_settings=None,
    ):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.patient_dataset_config = patient_dataset_config
        self.router_config = router_config or {}
        self.training_stage = training_stage
        self.svanet_config = svanet_config or {}
        self.stage_manager = None
        self.svanet_adapter = None
        self.last_refine_output = None
        self.last_metric_payload = None
        self.start_epoch = 0
        self.resume_batch_index = 0
        self.resume_rng_state = None
        self.resume_progress_state = {}
        self.resumed_global_step = None
        self.resume_best_metric = float("inf")
        self.resume_path = resume_path
        self.patient_datasets = []
        self.validation_patient_datasets = []
        self.wandb_settings = dict(wandb_settings or {})
        self.wandb_logger = None
        self.global_step = 0

        # Multi-GPU setup
        self.multi_gpu = multi_gpu
        self.local_rank = 0
        self.world_size = 1

        if self.multi_gpu:
            self.local_rank = setup_distributed()
            self.world_size = get_world_size()
            self.device = torch.device(f"cuda:{self.local_rank}")
            print_rank0(f"Multi-GPU training enabled with {self.world_size} GPUs")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build Model
        print_rank0("Building SAM3 model...")
        model_config = self.config.get("model", {}) or {}
        sam3_checkpoint = model_config.get("sam3_checkpoint") or None
        self.model = build_sam3_image_model(
            device=self.device.type,
            compile=False,
            checkpoint_path=sam3_checkpoint,
            load_from_HF=sam3_checkpoint is None,
            bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            eval_mode=False
        )

        # Apply LoRA
        print_rank0("Applying LoRA...")
        lora_cfg = self.config["lora"]
        lora_config = LoRAConfig(
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
            apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
            apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
            apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
            apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
            apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
        )
        self.model = apply_lora_to_model(self.model, lora_config)

        # Optional and deliberately opt-in: the original SAM3+LoRA path reaches
        # this point with moe_config=None and remains byte-for-byte equivalent in
        # model structure.  The MoE entry injects only decoder layers 4--6.
        self.moe_controller = None
        self.moe_aux_loss_weight = 0.0
        self.last_router_losses = {}
        if moe_config is not None and moe_config.get("enabled", True):
            from models.moe_injector import inject_hierarchical_moe

            self.moe_controller = inject_hierarchical_moe(
                self.model, moe_config, verbose=is_main_process()
            )
            self.moe_aux_loss_weight = float(
                moe_config.get("aux_loss_weight", 0.01)
            )
        loss_config = self.config.get("loss", {})
        self.router_loss_weights = {
            "modality_loss": float(loss_config.get("lambda_modality", 1.0)),
            "area_loss": float(loss_config.get("lambda_area", 1.0)),
            "boundary_router_loss": float(
                loss_config.get("lambda_boundary_router", 1.0)
            ),
            "load_balance_loss": float(
                loss_config.get("lambda_load_balance", 0.01)
            ),
        }
        self.moe_loss = None
        if self.moe_controller is not None:
            from models.moe_losses import HierarchicalMoELoss

            component_weights = {
                "sam3_loss": float(loss_config.get("lambda_sam3", 1.0)),
                "aux_loss": float(loss_config.get("lambda_aux", 1.0)),
                "modality_loss": float(loss_config.get("lambda_modality", 1.0)),
                "area_loss": float(loss_config.get("lambda_area", 1.0)),
                "area_reg_loss": float(loss_config.get("lambda_area_reg", 0.2)),
                "boundary_router_loss": float(
                    loss_config.get("lambda_boundary_router", 1.0)
                ),
                "boundary_seg_loss": float(
                    loss_config.get("lambda_boundary_seg", 0.5)
                ),
                "load_balance_loss": float(
                    loss_config.get("lambda_load_balance", 0.01)
                ),
                "refine_loss": float(loss_config.get("lambda_refine", 1.0)),
            }
            boundary_kernel = int(
                loss_config.get(
                    "boundary_kernel_size",
                    (self.patient_dataset_config or {})
                    .get("boundary_threshold_mode", {})
                    .get("boundary_band_width", 3),
                )
            )
            self.moe_loss = HierarchicalMoELoss(
                component_weights, boundary_kernel_size=boundary_kernel
            )

        if self.training_stage is not None:
            if self.moe_controller is None:
                raise ValueError("Staged training requires Hierarchical MoE")
            if self.training_stage in {4, 5}:
                if not self.svanet_config.get("enable", True):
                    raise ValueError("Stage 4/5 requires svanet.enable=true")
                from models.svanet_roi_adapter import (
                    SvANetROIAdapter, build_original_svanet,
                )

                source_root = Path(str(self.svanet_config.get("source_root", "../SvANet-main")))
                if not source_root.is_absolute():
                    source_root = (Path(config_path).resolve().parent / source_root).resolve()
                    if not (source_root / "opts.py").is_file():
                        source_root = (Path.cwd() / self.svanet_config.get("source_root", "../SvANet-main")).resolve()
                    if not (source_root / "opts.py").is_file():
                        source_root = Path(__file__).resolve().parent.parent / "SvANet-main"
                svanet = build_original_svanet(
                    source_root,
                    checkpoint=self.svanet_config.get("checkpoint") or None,
                    backbone_checkpoint=(
                        self.svanet_config.get("backbone_checkpoint") or None
                    ),
                    input_size=self.svanet_config.get("input_size", [512, 512]),
                    device=self.device,
                )
                self.svanet_adapter = SvANetROIAdapter(
                    svanet=svanet,
                    input_size=self.svanet_config.get("input_size", [512, 512]),
                    roi_expand_ratio=float(self.svanet_config.get("roi_expand_ratio", 0.25)),
                    min_roi_size=int(self.svanet_config.get("min_roi_size", 32)),
                    mask_threshold=float(self.svanet_config.get("mask_threshold", 0.5)),
                    empty_mask_fallback=self.svanet_config.get("empty_mask_fallback", "box_then_full_image"),
                    paste_mode=self.svanet_config.get("paste_mode", "replace_roi"),
                    outside_roi=self.svanet_config.get("outside_roi", "zero"),
                    train_trigger=self.svanet_config.get("train_trigger", "teacher_forcing"),
                ).to(self.device)

        stats = count_parameters(self.model)
        print_rank0(f"Trainable params: {stats['trainable_parameters']:,} ({stats['trainable_percentage']:.2f}%)")

        self.model.to(self.device)

        # Freeze/unfreeze before DDP construction so DDP registers exactly the
        # parameters owned by the selected stage.
        if self.training_stage is not None:
            from models.training_stages import StageTrainingManager

            optimizer_config = dict(self.config.get("optimizer") or {})
            optimizer_config.setdefault(
                "learning_rate", float(self.config["training"]["learning_rate"])
            )
            optimizer_config.setdefault(
                "weight_decay", float(self.config["training"]["weight_decay"])
            )
            self.stage_manager = StageTrainingManager(
                model=self.model,
                controller=self.moe_controller,
                stage=self.training_stage,
                optimizer_config=optimizer_config,
                svanet_adapter=self.svanet_adapter,
                stage_config=self.config.get("stages") or {},
            )
            loaded = (
                self.stage_manager.load_dependencies(self.config.get("model") or {})
                if load_stage_dependencies else []
            )
            if is_main_process():
                self.stage_manager.print_trainable_parameter_groups()
                if loaded:
                    print(f"Loaded stage dependencies: {loaded}")
            self.moe_loss.weights = self.stage_manager.active_loss_weights(
                self.moe_loss.weights
            )

        # Wrap model with DDP if multi-GPU
        if self.multi_gpu and any(p.requires_grad for p in self.model.parameters()):
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                # Top-1 MoE routing intentionally skips experts that are not
                # selected by the local batch. The selected experts can also
                # differ across ranks, so the set of parameters receiving a
                # gradient is dynamic and DDP must detect unused parameters.
                find_unused_parameters=True,
            )
            print_rank0(f"Model wrapped with DistributedDataParallel")

        # Store reference to unwrapped model for accessing custom methods
        self._unwrapped_model = self.model.module if isinstance(self.model, DDP) else self.model

        if (
            self.svanet_adapter is not None
            and self.multi_gpu
            and any(p.requires_grad for p in self.svanet_adapter.parameters())
        ):
            self.svanet_adapter = DDP(
                self.svanet_adapter,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
            )
        self._unwrapped_svanet = (
            self.svanet_adapter.module
            if isinstance(self.svanet_adapter, DDP)
            else self.svanet_adapter
        )

        # Optimizer
        if self.training_stage is not None:
            self.optimizer = self.stage_manager.build_optimizer()
        else:
            self.optimizer = AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=float(self.config["training"]["learning_rate"]),
                weight_decay=self.config["training"]["weight_decay"]
            )

        self.scheduler = None
        scheduler_config = self.config.get("training", {}).get("scheduler", {}) or {}
        if scheduler_config.get("enabled", False):
            scheduler_type = str(scheduler_config.get("type", "cosine")).lower()
            if scheduler_type == "cosine":
                epochs = int(
                    self.config["training"].get(
                        "epochs", self.config["training"].get("num_epochs", 1)
                    )
                )
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=max(int(scheduler_config.get("t_max", epochs)), 1),
                    eta_min=float(scheduler_config.get("eta_min", 0.0)),
                )
            elif scheduler_type == "step":
                self.scheduler = StepLR(
                    self.optimizer,
                    step_size=max(int(scheduler_config.get("step_size", 10)), 1),
                    gamma=float(scheduler_config.get("gamma", 0.1)),
                )
            else:
                raise ValueError(f"Unsupported scheduler.type: {scheduler_type}")

        configured_resume = self.resume_path or self.config.get("training", {}).get("resume")
        if configured_resume:
            if self.stage_manager is None:
                raise ValueError("Stage checkpoint resume is only available in MoE staged training")
            resumed = self.stage_manager.resume(
                configured_resume,
                self.optimizer,
                self.scheduler,
                restore_optimizer=bool(
                    self.config.get("training", {}).get("resume_optimizer", True)
                ),
            )
            self.start_epoch = int(resumed.get("epoch", 0))
            self.resume_batch_index = int(resumed.get("next_batch_index", 0))
            if self.resume_batch_index < 0:
                raise ValueError("Checkpoint next_batch_index must be non-negative")
            if self.resume_batch_index:
                self.resume_rng_state = resumed.get("rng_state") or None
                self.resume_progress_state = dict(
                    resumed.get("progress_state") or {}
                )
            self.resumed_global_step = resumed.get("global_step")
            self.resume_best_metric = float(
                resumed.get("best_metric", resumed.get("best_loss", float("inf")))
            )
            print_rank0(
                f"Resumed stage {self.training_stage} from {configured_resume} "
                f"at epoch {self.start_epoch}, next batch {self.resume_batch_index}"
            )
            if not resumed.get("optimizer_state_restored", False):
                print_rank0(
                    "WARNING: model and batch progress were restored, but the "
                    "optimizer is starting with fresh Adam state"
                )
        
        # Matcher & Loss
        self.matcher = BinaryHungarianMatcherV2(
            cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, focal=True
        )

        # Create loss functions with correct weights (from original SAM3 training config)
        # Note: These weights are for mask-based training
        loss_fns = [
            Boxes(weight_dict={
                "loss_bbox": 5.0,
                "loss_giou": 2.0
            }),
            IABCEMdetr(
                pos_weight=10.0,
                weight_dict={
                    "loss_ce": 20.0,
                    "presence_loss": 20.0
                },
                pos_focal=False,
                alpha=0.25,
                gamma=2,
                use_presence=True,
                pad_n_queries=200,
            ),
            Masks(
                weight_dict={
                    "loss_mask": 200.0,  # Much higher weight for mask loss!
                    "loss_dice": 10.0
                },
                focal_alpha=0.25,
                focal_gamma=2.0,
                compute_aux=False
            )
        ]

        # Create one-to-many matcher for auxiliary outputs
        o2m_matcher = BinaryOneToManyMatcher(
            alpha=0.3,
            threshold=0.4,
            topk=4
        )

        # Use Sam3LossWrapper for proper loss computation
        self.loss_wrapper = Sam3LossWrapper(
            loss_fns_find=loss_fns,
            matcher=self.matcher,
            o2m_matcher=o2m_matcher,
            o2m_weight=2.0,
            use_o2m_matcher_on_o2m_aux=False,
            normalization="local",  # Use local normalization (no distributed training)
            normalize_by_valid_object_num=False,
        )

        from models.wandb_logger import WandbLogger

        wandb_config = {
            "training_config": self.config,
            "training_stage": self.training_stage,
            "world_size": self.world_size,
        }
        self.wandb_logger = WandbLogger(
            self.wandb_settings,
            config=wandb_config,
            main_process=is_main_process(),
        )
        self.wandb_settings.pop("api_key", None)
        if (
            self.wandb_settings.get("watch_model", False)
            and self.wandb_logger.enabled
        ):
            self.wandb_logger.watch(self._unwrapped_model)
            if self._unwrapped_svanet is not None:
                self.wandb_logger.watch(self._unwrapped_svanet)

    def finish_wandb(self):
        """Finish the optional rank-zero W&B run without masking failures."""
        if self.wandb_logger is not None:
            self.wandb_logger.finish()

    def _add_moe_aux_loss(
        self, total_loss, outputs_list=None, find_targets=None,
        input_batch=None, metadata=None, epoch=0,
    ):
        self.last_refine_output = None
        self.last_metric_payload = None
        if self.moe_controller is None:
            return total_loss
        if outputs_list is None or find_targets is None or self.moe_loss is None:
            # Compatibility path for external callers that only provide a scalar.
            self.last_router_losses = self.moe_controller.routing_supervision_losses()
            for name, loss in self.last_router_losses.items():
                total_loss = total_loss + self.router_loss_weights[name] * loss
            return total_loss
        from models.moe_losses import extract_matched_masks

        if self.moe_controller.current_routes is None:
            raise RuntimeError("MoE routes are unavailable after SAM3 forward")
        if self.moe_controller.routing_targets is None:
            raise RuntimeError("MoE routing targets must be set before training forward")
        final_output = outputs_list.output[-1][-1]
        final_logits, gt_masks, aux_logits = extract_matched_masks(
            final_output,
            find_targets[-1],
            self.moe_controller.current_routes.get("coarse_mask_p3"),
        )
        routing_losses = self.moe_controller.routing_supervision_losses()
        # Preserve SAM3's complete native objective.  HierarchicalMoELoss adds
        # only the project-specific auxiliary/router/boundary terms around it;
        # it must not replace core_loss with a second Dice+BCE mask objective.
        total_loss, components = self.moe_loss(
            sam3_core_loss=total_loss,
            final_logits=final_logits,
            gt_masks=gt_masks,
            aux_logits=aux_logits,
            routes=self.moe_controller.current_routes,
            routing_losses=routing_losses,
            area_ratio_gt=self.moe_controller.routing_targets["area_ratio"],
        )
        if final_logits.shape[0] > 0 and input_batch is not None:
            batch_idx, _, target_idx = final_output["indices"]
            valid = find_targets[-1].get("is_valid_mask")
            if valid is not None:
                valid = valid if target_idx is None else valid[target_idx]
                batch_idx = batch_idx[valid]
            find_input = input_batch.find_inputs[-1]
            image_indices = find_input.img_ids.long()[batch_idx]
            images = input_batch.img_batch[image_indices]
            matched_metadata = (
                [metadata[index] for index in image_indices.detach().cpu().tolist()]
                if metadata is not None else []
            )
            refined_logits = final_logits

            routes = self.moe_controller.current_routes
            if self.svanet_adapter is not None:
                # SAM3 stores prompts as normalized CxCyWH. The ROI adapter
                # accepts normalized XYXY.
                from models.inference_utils import normalized_xyxy_prompts

                query_box_prompts = normalized_xyxy_prompts(find_input)
                box_prompts = [
                    query_box_prompts[query_index]
                    for query_index in batch_idx.tolist()
                ]
                teacher_area_mask = routes.get("teacher_area_mask")
                if (
                    self.svanet_adapter.training
                    and self.moe_controller.teacher_masks is not None
                    and "area" in self.moe_controller.teacher_masks
                ):
                    teacher_area_mask = self.moe_controller.teacher_masks["area"]
                if teacher_area_mask is not None:
                    teacher_area_mask = teacher_area_mask[batch_idx]
                adapter_output = self.svanet_adapter(
                    images=images,
                    sam3_logits=final_logits,
                    area_logits=routes["area_logits"][batch_idx],
                    area_labels=self.moe_controller.routing_targets["area"][batch_idx],
                    box_prompts=box_prompts,
                    gt_masks=gt_masks,
                    teacher_area_mask=teacher_area_mask,
                    use_gt_roi=(
                        bool(self.svanet_config.get("use_gt_roi_for_warmup", True))
                        and epoch < int(self.svanet_config.get("gt_roi_warmup_epochs", 2))
                        and self.svanet_adapter.training
                    ),
                )
                refine_loss = adapter_output["refine_loss"]
                total_loss = total_loss + self.moe_loss.weights.get("refine_loss", 0.0) * refine_loss
                components["refine_loss"] = refine_loss
                components["total_loss"] = total_loss
                self.last_refine_output = adapter_output
                refined_logits = adapter_output["final_logits"]
            self.last_metric_payload = {
                "base_logits": final_logits.detach(),
                "final_logits": refined_logits.detach(),
                "gt_masks": gt_masks.detach(),
                "metadata": matched_metadata,
            }
        self.last_router_losses = components
        return total_loss

    def _teacher_forcing_ratio(self, epoch):
        teacher = self.router_config.get("teacher_forcing", {})
        if not teacher.get("enabled", False):
            return 0.0
        start = float(teacher.get("start_ratio", 1.0))
        end = float(teacher.get("end_ratio", 0.0))
        if not 0.0 <= start <= 1.0 or not 0.0 <= end <= 1.0:
            raise ValueError("teacher forcing start_ratio/end_ratio must be in [0, 1]")
        decay_epochs = max(int(teacher.get("decay_epochs", 20)), 1)
        progress = min(max(float(epoch) / decay_epochs, 0.0), 1.0)
        return start + (end - start) * progress

    def _set_moe_routing_targets(self, metadata, input_batch, epoch, training):
        if self.moe_controller is None:
            return
        if not metadata or any(item is None for item in metadata):
            self.moe_controller.set_routing_targets(None)
            return
        image_targets = {
            "modality": torch.tensor(
                [int(item["modality_label"]) for item in metadata],
                device=self.device,
                dtype=torch.long,
            ),
            "area": torch.tensor(
                [int(item["area_label"]) for item in metadata],
                device=self.device,
                dtype=torch.long,
            ),
            "boundary": torch.tensor(
                [int(item["boundary_label"]) for item in metadata],
                device=self.device,
                dtype=torch.long,
            ),
            "area_ratio": torch.tensor(
                [float(item["area_ratio"]) for item in metadata],
                device=self.device,
                dtype=torch.float32,
            ),
        }
        for family, values in image_targets.items():
            if family == "area_ratio":
                if bool((~torch.isfinite(values) | (values < 0) | (values > 1)).any()):
                    raise ValueError(
                        f"Invalid area ratios in batch: {values.detach().cpu().tolist()}"
                    )
                continue
            classes = 2 if family == "modality" else 3
            if bool(((values < 0) | (values >= classes)).any()):
                raise ValueError(
                    f"Invalid {family} labels in batch: {values.detach().cpu().tolist()}"
                )
        img_ids = input_batch.find_inputs[0].img_ids.long()
        targets = {key: values[img_ids] for key, values in image_targets.items()}
        ratio = self._teacher_forcing_ratio(epoch) if training else 0.0
        common_mask = torch.rand(len(img_ids), device=self.device) < ratio
        use_gt = {
            "modality": bool(
                self.router_config.get("use_gt_modality_during_train", True)
            ),
            "area": bool(self.router_config.get("use_gt_area_during_warmup", True)),
            "boundary": bool(
                self.router_config.get("use_gt_boundary_during_warmup", True)
            ),
        }
        teacher_masks = {
            family: common_mask & enabled for family, enabled in use_gt.items()
        }
        self.moe_controller.set_routing_targets(targets, teacher_masks)

    def _set_moe_modality_targets(self, metadata, input_batch):
        """Compatibility wrapper: targets only, with no teacher routing."""
        self._set_moe_routing_targets(metadata, input_batch, epoch=0, training=False)

    def _save_adapter_weights(self, model, lora_path):
        save_lora_weights(model, str(lora_path))
        if self.moe_controller is not None:
            from models.moe_injector import save_moe_weights

            moe_name = lora_path.name.replace("lora_weights", "moe_weights")
            save_moe_weights(self.moe_controller, str(lora_path.with_name(moe_name)))

    @staticmethod
    def _capture_rng_state():
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng_state(state):
        if not state:
            return
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "torch" in state:
            torch.set_rng_state(state["torch"].cpu())
        if torch.cuda.is_available() and state.get("cuda") is not None:
            torch.cuda.set_rng_state_all(
                [value.cpu() for value in state["cuda"]]
            )

    def _save_stage_checkpoint(
        self,
        out_dir,
        epoch,
        loss,
        best=False,
        checkpoint_name=None,
        next_batch_index=0,
        progress_state=None,
        checkpoint_kind="epoch",
    ):
        if self.stage_manager is None:
            return
        from models.wandb_logger import sanitize_config

        name = checkpoint_name or (
            self.stage_manager.checkpoint_name
            if best else f"stage{self.training_stage}_{self.stage_manager.stage_config.get('last_suffix', 'last')}.pt"
        )
        self.stage_manager.save_checkpoint(
            Path(out_dir) / name,
            self.optimizer,
            epoch=epoch,
            best_loss=loss,
            scheduler=self.scheduler,
            selected_patient_ids=self._selected_patient_state(),
            area_thresholds=self._dataset_threshold_state("area_thresholds"),
            boundary_thresholds=self._dataset_threshold_state("boundary_thresholds"),
            config=sanitize_config(self.config, remove_paths=False),
            next_batch_index=next_batch_index,
            global_step=self.global_step,
            rng_state=(
                self._capture_rng_state() if checkpoint_kind == "step" else None
            ),
            progress_state=progress_state,
            checkpoint_kind=checkpoint_kind,
        )

    def _selected_patient_state(self):
        state = {"mr_patient_ids": [], "us_patient_ids": []}
        for dataset in self.patient_datasets:
            state[f"{dataset.modality.lower()}_patient_ids"] = list(dataset.patient_ids)
        return state

    def _dataset_threshold_state(self, attribute):
        for dataset in self.patient_datasets:
            value = getattr(dataset, attribute, None)
            if value:
                return dict(value)
        return {}

    def _merge_epoch_statistics(self, statistics):
        if not self.multi_gpu:
            return statistics
        from models.training_metrics import EpochStatistics

        gathered = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, statistics)
        merged = EpochStatistics()
        for item in gathered:
            merged.merge(item)
        return merged

    def _current_learning_rate_metrics(self):
        metrics = {}
        for index, group in enumerate(self.optimizer.param_groups):
            name = str(group.get("group_name", f"group_{index}"))
            metrics[f"train/lr_{name}"] = float(group["lr"])
            if index == 0:
                metrics["train/lr"] = float(group["lr"])
        return metrics

    def _reduce_scalar_metrics(self, metrics):
        """Average low-cost scalar batch metrics across DDP ranks."""
        if not self.multi_gpu or not metrics:
            return metrics
        keys = sorted(metrics)
        values = torch.tensor(
            [float(metrics[key]) for key in keys],
            device=self.device,
            dtype=torch.float64,
        )
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= self.world_size
        return {
            key: float(value)
            for key, value in zip(keys, values.detach().cpu().tolist())
        }

    @staticmethod
    def _wandb_report_metrics(prefix, report):
        """Map the existing globally merged epoch report to W&B scalars."""
        metrics = {}
        losses = report.get("loss", {})
        total_loss = losses.get("total_loss")
        if total_loss is not None:
            metrics[f"{prefix}/epoch_loss"] = float(total_loss)
            metrics[f"{prefix}/loss"] = float(total_loss)
        for name, value in losses.items():
            metrics[f"{prefix}/epoch_{name}"] = float(value)
            if name != "total_loss":
                metrics[f"{prefix}/{name}"] = float(value)
        for name, value in report.get("segmentation", {}).items():
            metrics[f"{prefix}/{name}"] = float(value)
        for name, value in report.get("router", {}).items():
            metrics[f"{prefix}/router/{name}"] = float(value)
        for name, value in report.get("svanet", {}).items():
            metrics[f"{prefix}/svanet/{name}"] = float(value)
        svanet = report.get("svanet", {})
        small_count = float(svanet.get("small_count", 0))
        trigger_count = float(svanet.get("trigger_count", 0))
        metrics[f"{prefix}/svanet/trigger_ratio"] = (
            trigger_count / small_count if small_count else 0.0
        )

        experts = report.get("experts", {})
        for name, value in experts.items():
            metrics[f"{prefix}/experts/{name}_count"] = float(value)
        for family in ("MR_area", "US_area", "MR_boundary", "US_boundary"):
            selected = {
                name: float(value)
                for name, value in experts.items()
                if name.startswith(f"{family}_")
            }
            total = sum(selected.values())
            for name, value in selected.items():
                metrics[f"{prefix}/experts/{name}_ratio"] = (
                    value / total if total else 0.0
                )
        area_total = sum(
            float(value) for name, value in experts.items() if "_area_" in name
        )
        boundary_total = sum(
            float(value)
            for name, value in experts.items()
            if "_boundary_" in name
        )
        for modality in ("MR", "US"):
            count = sum(
                float(value)
                for name, value in experts.items()
                if name.startswith(f"{modality}_area_")
            )
            metrics[f"{prefix}/router/modality_{modality}_ratio"] = (
                count / area_total if area_total else 0.0
            )
        for label in ("small", "medium", "large"):
            count = sum(
                float(value)
                for name, value in experts.items()
                if name.endswith(f"_area_{label}")
            )
            metrics[f"{prefix}/router/area_{label}_ratio"] = (
                count / area_total if area_total else 0.0
            )
        for label in ("clear", "fuzzy", "complex"):
            count = sum(
                float(value)
                for name, value in experts.items()
                if name.endswith(f"_boundary_{label}")
            )
            metrics[f"{prefix}/router/boundary_{label}_ratio"] = (
                count / boundary_total if boundary_total else 0.0
            )
        return metrics

    @staticmethod
    def _write_statistics_record(path, record):
        path = Path(path)
        records = []
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                records = loaded if isinstance(loaded, list) else []
            except (json.JSONDecodeError, OSError):
                records = []
        records.append(record)
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    def _build_patient_dataset(self, split, training):
        from data.patient_dataset import PatientDataset, save_selected_patients

        cfg = self.patient_dataset_config
        is_validation = (
            not training and str(split) == str(cfg.get("val_split", "val"))
        )
        sampling_key = (
            "patient_sampling" if training else "val_patient_sampling"
        )
        sampling = cfg.get(sampling_key, {}) or {} if (training or is_validation) else {}
        sampling_mode = sampling.get("mode", "random" if training else "sequential")
        seed = int(sampling.get("seed", 42))
        if training:
            explicit = cfg.get("patient_ids", {}) or {}
        elif is_validation:
            explicit = cfg.get("val_patient_ids", {}) or {}
        else:
            explicit = {}
        datasets = []
        def split_path(value):
            return value.format(split=split) if isinstance(value, str) else value

        for modality in ("mr", "us"):
            root = cfg.get(f"{modality}_root") or cfg.get(f"{modality}_image_root")
            if not root:
                continue
            coco_json = cfg.get(f"{modality}_{split}_coco_json") or split_path(
                cfg.get(f"{modality}_coco_json")
            )
            base_dataset = COCOSegmentDataset(
                data_dir=root,
                split=split,
                annotation_file=coco_json,
            )
            datasets.append(
                PatientDataset(
                    base_dataset=base_dataset,
                    modality_root=root,
                    split=split,
                    modality=modality.upper(),
                    num_patients=(
                        cfg.get(f"num_{modality}_patients")
                        if training
                        else (
                            cfg.get(f"num_{modality}_val_patients")
                            if is_validation else None
                        )
                    ),
                    patient_ids=explicit.get(modality),
                    sampling_mode=sampling_mode,
                    seed=seed,
                    mask_root=cfg.get(f"{modality}_mask_root"),
                    boxes_json=(
                        cfg.get(f"{modality}_{split}_boxes_json")
                        or split_path(cfg.get(f"{modality}_boxes_json"))
                    ),
                    strict_dataset_check=cfg.get("strict_dataset_check", True),
                    area_thresholds=cfg.get("area_threshold_file"),
                    boundary_thresholds=cfg.get("boundary_threshold_file"),
                    return_format=cfg.get("return_format", "sam3"),
                    max_slices_per_patient=(
                        cfg.get("max_slices_per_patient")
                        if training
                        else (
                            cfg.get("max_val_slices_per_patient")
                            if is_validation else None
                        )
                    ),
                )
            )
        if not datasets:
            raise ValueError("Patient dataset config must provide mr_root and/or us_root")
        output_dir = Path(self.config["output"]["output_dir"])
        if training:
            self.patient_datasets = datasets
            resample = bool(cfg.get("resample_patients_each_epoch", False))
            if resample:
                raise NotImplementedError(
                    "resample_patients_each_epoch=true is not supported yet; "
                    "the default fixed patient split is required for reproducibility"
                )
            manifest_path = cfg.get(
                "selected_patients_json",
                str(output_dir / "selected_patients.json"),
            )
            if is_main_process():
                save_selected_patients(
                    datasets,
                    manifest_path,
                    sampling_mode=sampling_mode,
                    seed=seed,
                    resample_patients_each_epoch=resample,
                )
        elif is_validation:
            self.validation_patient_datasets = datasets
            manifest_path = cfg.get(
                "selected_val_patients_json",
                str(output_dir / "selected_val_patients.json"),
            )
            if is_main_process():
                save_selected_patients(
                    datasets,
                    manifest_path,
                    sampling_mode=sampling_mode,
                    seed=seed,
                    resample_patients_each_epoch=False,
                )
        return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    def _log_validation_patient_selection(self):
        if not self.validation_patient_datasets or not is_main_process():
            return
        total_patients = 0
        total_slices = 0
        print("\nValidation patient selection:")
        for dataset in self.validation_patient_datasets:
            print(f"{dataset.modality} patients: {dataset.patient_ids}")
            print(f"{dataset.modality} slices: {dataset.num_slices}")
            total_patients += len(dataset.patient_ids)
            total_slices += dataset.num_slices
        print(f"Total validation patients: {total_patients}")
        print(f"Total validation slices: {total_slices}")

    def _log_patient_selection(self, epoch):
        if not self.patient_datasets or not is_main_process():
            return
        total_patients = 0
        total_slices = 0
        print(f"\nEpoch {epoch + 1} patient selection:")
        for dataset in self.patient_datasets:
            print(f"{dataset.modality} patients: {dataset.patient_ids}")
            print(f"{dataset.modality} slices: {dataset.num_slices}")
            total_patients += len(dataset.patient_ids)
            total_slices += dataset.num_slices
        print(f"Total patients: {total_patients}")
        print(f"Total slices: {total_slices}")
        
    def train(self):
        # Get data directory from config (should point to directory containing train/valid folders)
        data_dir = self.config["training"]["data_dir"]

        # Load datasets using COCO format
        print_rank0(f"\nLoading training data from {data_dir}...")
        if self.patient_dataset_config is not None:
            train_split = self.patient_dataset_config.get("train_split", "train")
            train_ds = self._build_patient_dataset(train_split, training=True)
        else:
            train_ds = COCOSegmentDataset(data_dir=data_dir, split="train")

        # Check if validation data exists
        has_validation = False
        val_ds = None

        try:
            print_rank0(f"\nLoading validation data from {data_dir}...")
            if self.patient_dataset_config is not None:
                val_split = self.patient_dataset_config.get("val_split", "val")
                val_ds = self._build_patient_dataset(val_split, training=False)
            else:
                val_ds = COCOSegmentDataset(data_dir=data_dir, split="valid")
            if len(val_ds) > 0:
                has_validation = True
                print_rank0(f"Found validation data: {len(val_ds)} images")
                if self.patient_dataset_config is not None:
                    self._log_validation_patient_selection()
            else:
                print_rank0(f"Validation dataset is empty.")
                val_ds = None
        except Exception as e:
            print_rank0(f"Could not load validation data: {e}")
            val_ds = None

        if not has_validation:
            val_ds = None

        def collate_fn(batch):
            metadata = [getattr(item, "patient_metadata", None) for item in batch]
            output = collate_fn_api(batch, dict_key="input", with_seg_masks=True)
            output["_patient_metadata"] = metadata
            return output

        # Create samplers for distributed training
        train_sampler = None
        val_sampler = None
        training_config = self.config.get("training", {}) or {}
        checkpoint_interval_steps = int(
            training_config.get("checkpoint_interval_steps", 0)
        )
        if checkpoint_interval_steps < 0:
            raise ValueError("training.checkpoint_interval_steps must be non-negative")
        if (checkpoint_interval_steps or self.resume_batch_index) and self.multi_gpu:
            raise ValueError(
                "Mid-epoch checkpointing currently supports single-GPU training only"
            )
        step_checkpoint_name = str(
            training_config.get(
                "step_checkpoint_name", f"stage{self.training_stage}_step_last.pt"
            )
        ).strip()
        if checkpoint_interval_steps and not step_checkpoint_name:
            raise ValueError("training.step_checkpoint_name must not be empty")
        data_order_seed = int(
            training_config.get(
                "data_order_seed",
                ((self.patient_dataset_config or {}).get("patient_sampling") or {}).get(
                    "seed", 42
                ),
            )
        )
        deterministic_step_resume = bool(
            checkpoint_interval_steps or self.resume_batch_index
        )
        train_generator = torch.Generator() if deterministic_step_resume else None
        resumable_batch_sampler = None

        if self.multi_gpu:
            train_sampler = DistributedSampler(
                train_ds,
                num_replicas=self.world_size,
                rank=get_rank(),
                shuffle=True
            )
            if has_validation:
                val_sampler = DistributedSampler(
                    val_ds,
                    num_replicas=self.world_size,
                    rank=get_rank(),
                    shuffle=False
                )

        loader_kwargs = {
            "dataset": train_ds,
            "collate_fn": collate_fn,
            "num_workers": self.config["training"].get("num_workers", 0),
            "pin_memory": True,
            "generator": train_generator,
        }
        if deterministic_step_resume:
            resumable_batch_sampler = ResumableRandomBatchSampler(
                dataset_size=len(train_ds),
                batch_size=self.config["training"]["batch_size"],
                seed=data_order_seed,
            )
            train_loader = DataLoader(
                batch_sampler=resumable_batch_sampler,
                **loader_kwargs,
            )
            total_train_batches = resumable_batch_sampler.total_batches
        else:
            train_loader = DataLoader(
                batch_size=self.config["training"]["batch_size"],
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                **loader_kwargs,
            )
            total_train_batches = len(train_loader)
        if self.resume_batch_index > total_train_batches:
            raise ValueError(
                f"Checkpoint next_batch_index={self.resume_batch_index} exceeds "
                f"the current train loader length {total_train_batches}"
            )
        if self.resumed_global_step is not None:
            self.global_step = int(self.resumed_global_step)
        else:
            self.global_step = self.start_epoch * total_train_batches

        if has_validation:
            val_loader = DataLoader(
                val_ds,
                batch_size=self.config["training"]["batch_size"],
                shuffle=False,
                sampler=val_sampler,
                collate_fn=collate_fn,
                num_workers=self.config["training"].get("num_workers", 0),
                pin_memory=True
            )
        else:
            val_loader = None

        if self.stage_manager is not None:
            self.stage_manager.set_module_modes(training=True)
        else:
            self.model.train()

        # Weights from a standard SAM config roughly
        weight_dict = {
            "loss_ce": 2.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_mask": 5.0,
            "loss_dice": 5.0
        }

        epochs = int(
            self.config["training"].get(
                "epochs", self.config["training"].get("num_epochs", 1)
            )
        )
        best_val_loss = self.resume_best_metric
        print_rank0(f"Starting training for {epochs} epochs...")

        if has_validation:
            print_rank0(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
        else:
            print_rank0(f"Training samples: {len(train_ds)}")
            print_rank0("WARNING: No validation data found - training without validation")

        if self.multi_gpu:
            print_rank0(f"Effective batch size: {self.config['training']['batch_size']} x {self.world_size} = {self.config['training']['batch_size'] * self.world_size}")

        # Helper to move BatchedDatapoint to device
        def move_to_device(obj, device):
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            elif isinstance(obj, list):
                return [move_to_device(x, device) for x in obj]
            elif isinstance(obj, tuple):
                return tuple(move_to_device(x, device) for x in obj)
            elif isinstance(obj, dict):
                return {k: move_to_device(v, device) for k, v in obj.items()}
            elif hasattr(obj, "__dataclass_fields__"):
                for field in obj.__dataclass_fields__:
                    val = getattr(obj, field)
                    setattr(obj, field, move_to_device(val, device))
                return obj
            return obj

        # Create output directory
        out_dir = Path(self.config["output"]["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        from models.training_metrics import EpochStatistics

        debug_mode = os.environ.get("SVANET_DEBUG_MODE", "").strip().lower()
        debug_max_batches = int(
            os.environ.get("SVANET_DEBUG_MAX_BATCHES", "0")
        )
        valid_debug_modes = {
            "",
            "forward_bn_eval",
            "forward_bn_train",
            "train_bn_eval",
        }
        if debug_mode not in valid_debug_modes:
            raise ValueError(
                f"Unknown SVANET_DEBUG_MODE={debug_mode!r}; expected one of "
                f"{sorted(mode for mode in valid_debug_modes if mode)}"
            )

        print_rank0(
            f"SvANet debug mode: {debug_mode or 'disabled'}, "
            f"max batches: {debug_max_batches or 'unlimited'}"
        )
        if checkpoint_interval_steps:
            print_rank0(
                f"Mid-epoch checkpointing: every {checkpoint_interval_steps} steps, "
                f"path={out_dir / step_checkpoint_name}"
            )
            if int(training_config.get("num_workers", 0)) != 0:
                print_rank0(
                    "WARNING: exact random augmentation replay is only guaranteed "
                    "with training.num_workers=0"
                )
        for epoch in range(self.start_epoch, epochs):
            if self.stage_manager is not None:
                self.stage_manager.set_module_modes(training=True)

                if debug_mode in {"forward_bn_eval", "train_bn_eval"}:
                    frozen_bn_count = 0
                    for module in self._unwrapped_svanet.modules():
                        if isinstance(
                            module,
                            torch.nn.modules.batchnorm._BatchNorm,
                        ):
                            module.eval()
                            frozen_bn_count += 1
                    print_rank0(
                        f"[BN-DEBUG] epoch={epoch + 1} "
                        f"frozen_batchnorm_modules={frozen_bn_count}"
                    )
            self._log_patient_selection(epoch)
            # Set epoch for distributed sampler (required for proper shuffling)
            if self.multi_gpu and train_sampler is not None:
                train_sampler.set_epoch(epoch)
            elif resumable_batch_sampler is not None:
                resumable_batch_sampler.set_epoch(
                    epoch,
                    start_batch=(
                        self.resume_batch_index
                        if epoch == self.start_epoch
                        else 0
                    ),
                )
                # Keep DataLoader worker seeding deterministic and independent
                # from the sampler's own generator.
                train_generator.manual_seed(data_order_seed + epoch)

            # Track training losses for this epoch
            train_losses = []
            train_statistics = EpochStatistics()
            epoch_learning_rates = {
                group.get("group_name", f"group_{index}"): group["lr"]
                for index, group in enumerate(self.optimizer.param_groups)
            }
            teacher_counts = {
                family: [0, 0] for family in ("modality", "area", "boundary")
            }
            resume_batch = (
                self.resume_batch_index if epoch == self.start_epoch else 0
            )
            if resume_batch:
                progress = self.resume_progress_state
                train_losses = [float(value) for value in progress.get("train_losses", [])]
                if progress.get("train_statistics"):
                    train_statistics.load_state_dict(progress["train_statistics"])
                restored_teacher = progress.get("teacher_counts", {})
                for family in teacher_counts:
                    values = restored_teacher.get(family, [0, 0])
                    teacher_counts[family] = [int(values[0]), int(values[1])]
                print_rank0(
                    f"Resuming epoch {epoch + 1} at batch "
                    f"{resume_batch}/{total_train_batches}"
                )
            if self.resume_rng_state is not None:
                # The resumable sampler does not read skipped samples, so restore
                # before DataLoader fetches the first required batch.
                self._restore_rng_state(self.resume_rng_state)
                self.resume_rng_state = None

            # Only show progress bar on rank 0
            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}",
                total=total_train_batches,
                initial=resume_batch,
                disable=not is_main_process(),
            )

            for batch_index, batch_dict in enumerate(pbar, start=resume_batch):
                if (
                    debug_max_batches > 0
                    and batch_index >= debug_max_batches
                ):
                    print_rank0(
                        f"Stopped after {debug_max_batches} debug batches"
                    )
                    break

                input_batch = batch_dict["input"]
                # Move to device
                input_batch = move_to_device(input_batch, self.device)
                self._set_moe_routing_targets(
                    batch_dict.get("_patient_metadata"),
                    input_batch,
                    epoch=epoch,
                    training=True,
                )

                # Forward pass
                # outputs_list is SAM3Output, we need to pass the whole thing to loss_wrapper
                outputs_list = self.model(input_batch)
                if self.moe_controller is not None:
                    for family, (used, total) in (
                        self.moe_controller.teacher_routing_statistics().items()
                    ):
                        teacher_counts[family][0] += used
                        teacher_counts[family][1] += total

                # Prepare targets for loss
                # input_batch.find_targets is a list of BatchedFindTarget (one per stage)
                find_targets = [self._unwrapped_model.back_convert(target) for target in input_batch.find_targets]

                # Move targets to device
                for targets in find_targets:
                    for k, v in targets.items():
                        if isinstance(v, torch.Tensor):
                            targets[k] = v.to(self.device)

                # Add matcher indices to outputs (required by Sam3LossWrapper)
                # Use SAM3Output.iteration_mode to properly iterate over outputs
                with SAM3Output.iteration_mode(
                    outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
                ) as outputs_iter:
                    for stage_outputs, stage_targets in zip(outputs_iter, find_targets):
                        # stage_targets is a single target dict, replicate for all steps
                        stage_targets_list = [stage_targets] * len(stage_outputs)
                        for outputs, targets in zip(stage_outputs, stage_targets_list):
                            # Compute indices for main output
                            outputs["indices"] = self.matcher(outputs, targets)

                            # Also add indices to auxiliary outputs if they exist
                            if "aux_outputs" in outputs:
                                for aux_out in outputs["aux_outputs"]:
                                    aux_out["indices"] = self.matcher(aux_out, targets)

                # Compute loss using Sam3LossWrapper
                # This handles num_boxes calculation and proper weighting
                loss_dict = self.loss_wrapper(outputs_list, find_targets)

                # Extract total loss
                total_loss = self._add_moe_aux_loss(
                    loss_dict[CORE_LOSS_KEY], outputs_list, find_targets,
                    input_batch=input_batch,
                    metadata=batch_dict.get("_patient_metadata"), epoch=epoch,
                )

                if self.moe_controller is not None:
                    batch_metadata = batch_dict.get("_patient_metadata") or []
                    train_statistics.update_losses(
                        self.last_router_losses,
                        weight=len(batch_metadata) or 1,
                    )
                    train_statistics.update_router(
                        self.moe_controller.current_routes,
                        self.moe_controller.routing_targets,
                    )
                    train_statistics.update_svanet(self.last_refine_output)
                    if self.last_metric_payload and self.last_metric_payload["metadata"]:
                        train_statistics.update_segmentation(**self.last_metric_payload)
                if not torch.isfinite(total_loss):
                    refine_loss = None

                    if self.last_refine_output is not None:
                        refine_loss = self.last_refine_output.get("refine_loss")

                    refine_value = (
                        refine_loss.detach().item()
                        if torch.is_tensor(refine_loss)
                        else refine_loss
                    )

                    raise FloatingPointError(
                        f"Non-finite forward loss at batch {batch_index}: "
                        f"total={total_loss.detach().item()}, "
                        f"refine={refine_value}"
                    )

                if debug_mode in {"forward_bn_eval", "forward_bn_train"}:
                    print_rank0(
                        f"[DEBUG][batch={batch_index}] "
                        f"forward-only total_loss={total_loss.detach().item():.8f}"
                    )
                    continue

                # Backward
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                if debug_mode == "train_bn_eval":
                    bad_gradients = []
                    largest_grad_name = None
                    largest_grad_value = 0.0
                    for name, parameter in self._unwrapped_svanet.named_parameters():
                        if not parameter.requires_grad or parameter.grad is None:
                            continue
                        if not torch.isfinite(parameter.grad).all():
                            bad_gradients.append(name)
                            continue
                        grad_abs_max = parameter.grad.detach().abs().max().item()
                        if grad_abs_max > largest_grad_value:
                            largest_grad_value = grad_abs_max
                            largest_grad_name = name
                    if bad_gradients:
                        raise FloatingPointError(
                            "Non-finite SvANet gradients: "
                            + ", ".join(bad_gradients[:20])
                        )
                    pre_clip_norm = torch.nn.utils.clip_grad_norm_(
                        self._unwrapped_svanet.parameters(),
                        max_norm=1.0,
                        error_if_nonfinite=True,
                    )
                    print_rank0(
                        f"[GRAD-DEBUG][batch={batch_index}] "
                        f"pre_clip_norm={float(pre_clip_norm):.8f} "
                        f"largest_grad={largest_grad_value:.8f} "
                        f"largest_grad_name={largest_grad_name}"
                    )
                self.optimizer.step()
                self.global_step += 1
                if debug_mode == "train_bn_eval":
                    bad_parameters = [
                        name
                        for name, parameter in self._unwrapped_svanet.named_parameters()
                        if not torch.isfinite(parameter.detach()).all()
                    ]
                    if bad_parameters:
                        raise FloatingPointError(
                            "Optimizer produced non-finite SvANet parameters: "
                            + ", ".join(bad_parameters[:20])
                        )
                    print_rank0(
                        f"[PARAM-DEBUG][batch={batch_index}] all_finite=True"
                    )

                # Track training loss
                train_losses.append(total_loss.item())
                pbar.set_postfix({"loss": total_loss.item()})
                should_log_wandb = (
                    self.wandb_settings.get("enabled", False)
                    and self.global_step % self.wandb_logger.log_interval == 0
                )
                epoch_batch_limit = (
                    min(total_train_batches, debug_max_batches)
                    if debug_max_batches > 0 else total_train_batches
                )
                if should_log_wandb and batch_index + 1 < epoch_batch_limit:
                    batch_metrics = {
                        "train/loss": float(total_loss.detach().float().item()),
                        "train/epoch": epoch + 1,
                        "train/iteration": batch_index + 1,
                        "train/global_step": self.global_step,
                    }
                    for name, value in self.last_router_losses.items():
                        if name == "total_loss":
                            continue
                        if torch.is_tensor(value) and value.numel() == 1:
                            batch_metrics[f"train/{name}"] = float(
                                value.detach().float().item()
                            )
                        elif isinstance(value, (int, float)):
                            batch_metrics[f"train/{name}"] = float(value)
                    batch_metrics.update(self._current_learning_rate_metrics())
                    batch_metrics = self._reduce_scalar_metrics(batch_metrics)
                    self.wandb_logger.log(batch_metrics, step=self.global_step)

                if (
                    checkpoint_interval_steps
                    and self.global_step % checkpoint_interval_steps == 0
                ):
                    progress_state = {
                        "train_losses": list(train_losses),
                        "train_statistics": train_statistics.state_dict(),
                        "teacher_counts": {
                            name: list(values) for name, values in teacher_counts.items()
                        },
                    }
                    self._save_stage_checkpoint(
                        out_dir,
                        epoch=epoch,
                        # Preserve the best completed validation metric. The
                        # current batch losses are stored in progress_state.
                        loss=best_val_loss,
                        checkpoint_name=step_checkpoint_name,
                        next_batch_index=batch_index + 1,
                        progress_state=progress_state,
                        checkpoint_kind="step",
                    )
                    print_rank0(
                        f"Saved mid-epoch checkpoint at epoch {epoch + 1}, "
                        f"next batch {batch_index + 1}: {out_dir / step_checkpoint_name}"
                    )

            self.resume_batch_index = 0
            self.resume_progress_state = {}

            # Calculate average training loss for this epoch
            avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0.0
            train_statistics = self._merge_epoch_statistics(train_statistics)
            train_report = train_statistics.report(self._teacher_forcing_ratio(epoch))
            if self.moe_controller is not None:
                avg_train_loss = train_report["loss"]["total_loss"]
            if self.moe_controller is not None and is_main_process():
                configured_ratio = self._teacher_forcing_ratio(epoch)
                actual = {
                    family: (used / total if total else 0.0)
                    for family, (used, total) in teacher_counts.items()
                }
                print(
                    f"Teacher routing epoch {epoch + 1}: configured={configured_ratio:.4f}, "
                    f"modality={actual['modality']:.4f}, area={actual['area']:.4f}, "
                    f"boundary={actual['boundary']:.4f}"
                )

            # Validation loss plus slice-, patient- and subgroup-level metrics.
            validation_interval = max(
                int(self.config.get("training", {}).get("validation_interval", 1)), 1
            )
            run_validation = (
                has_validation and val_loader is not None
                and (epoch + 1) % validation_interval == 0
            )
            if run_validation:
                if self.stage_manager is not None:
                    self.stage_manager.set_module_modes(training=False)
                else:
                    self.model.eval()
                val_losses = []
                val_statistics = EpochStatistics()

                with torch.no_grad():
                    val_pbar = tqdm(val_loader, desc=f"Validation", disable=not is_main_process())

                    for batch_dict in val_pbar:
                        input_batch = batch_dict["input"]
                        input_batch = move_to_device(input_batch, self.device)
                        self._set_moe_routing_targets(
                            batch_dict.get("_patient_metadata"),
                            input_batch,
                            epoch=epoch,
                            training=False,
                        )

                        # Forward pass
                        outputs_list = self.model(input_batch)

                        # Prepare targets
                        find_targets = [self._unwrapped_model.back_convert(target) for target in input_batch.find_targets]

                        # Move targets to device
                        for targets in find_targets:
                            for k, v in targets.items():
                                if isinstance(v, torch.Tensor):
                                    targets[k] = v.to(self.device)

                        # Add matcher indices to outputs (required by Sam3LossWrapper)
                        with SAM3Output.iteration_mode(
                            outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
                        ) as outputs_iter:
                            for stage_outputs, stage_targets in zip(outputs_iter, find_targets):
                                stage_targets_list = [stage_targets] * len(stage_outputs)
                                for outputs, targets in zip(stage_outputs, stage_targets_list):
                                    outputs["indices"] = self.matcher(outputs, targets)
                                    if "aux_outputs" in outputs:
                                        for aux_out in outputs["aux_outputs"]:
                                            aux_out["indices"] = self.matcher(aux_out, targets)

                        # Compute loss using Sam3LossWrapper
                        loss_dict = self.loss_wrapper(outputs_list, find_targets)
                        total_loss = self._add_moe_aux_loss(
                            loss_dict[CORE_LOSS_KEY], outputs_list, find_targets,
                            input_batch=input_batch,
                            metadata=batch_dict.get("_patient_metadata"), epoch=epoch,
                        )

                        if self.moe_controller is not None:
                            batch_metadata = batch_dict.get("_patient_metadata") or []
                            val_statistics.update_losses(
                                self.last_router_losses,
                                weight=len(batch_metadata) or 1,
                            )
                            val_statistics.update_router(
                                self.moe_controller.current_routes,
                                self.moe_controller.routing_targets,
                            )
                            val_statistics.update_svanet(self.last_refine_output)
                            if self.last_metric_payload and self.last_metric_payload["metadata"]:
                                val_statistics.update_segmentation(**self.last_metric_payload)

                        val_losses.append(total_loss.item())
                        val_pbar.set_postfix({"val_loss": total_loss.item()})

                avg_val_loss = sum(val_losses) / len(val_losses)
                val_statistics = self._merge_epoch_statistics(val_statistics)
                val_report = val_statistics.report(0.0)

                # Synchronize val_loss across all processes for consistent best model selection
                if self.multi_gpu:
                    val_loss_tensor = torch.tensor([avg_val_loss], device=self.device)
                    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
                    avg_val_loss = val_loss_tensor.item()

                print_rank0(f"\nEpoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
                if self.scheduler is not None:
                    self.scheduler.step()
                epoch_metrics = self._wandb_report_metrics("train", train_report)
                epoch_metrics.update(
                    self._wandb_report_metrics("val", val_report)
                )
                epoch_metrics.update(self._current_learning_rate_metrics())
                epoch_metrics.update({
                    "train/epoch": epoch + 1,
                    "train/iteration": len(train_losses),
                    "train/global_step": self.global_step,
                    "val/loss": avg_val_loss,
                })
                is_new_best = avg_val_loss < best_val_loss
                if is_new_best:
                    epoch_metrics.update({
                        "best/epoch": epoch + 1,
                        "best/val_loss": avg_val_loss,
                    })
                self.wandb_logger.log(epoch_metrics, step=self.global_step)
                if is_main_process() and self.moe_controller is not None:
                    record = {
                        "epoch": epoch + 1,
                        "stage": self.training_stage,
                        "learning_rates": epoch_learning_rates,
                        "train": train_report,
                        "validation": val_report,
                    }
                    self._write_statistics_record(
                        out_dir / "router_statistics.json", record
                    )
                    print("MoE epoch statistics:")
                    print(json.dumps(record, indent=2))

                # Save models based on validation loss (only on rank 0)
                if is_main_process():
                    # Get underlying model from DDP wrapper
                    model_to_save = self._unwrapped_model
                    self._save_adapter_weights(model_to_save, out_dir / "last_lora_weights.pt")
                    self._save_stage_checkpoint(
                        out_dir, epoch=epoch + 1, loss=avg_val_loss, best=False
                    )

                    if is_new_best:
                        best_val_loss = avg_val_loss
                        self._save_adapter_weights(model_to_save, out_dir / "best_lora_weights.pt")
                        self._save_stage_checkpoint(
                            out_dir, epoch=epoch + 1, loss=avg_val_loss, best=True
                        )
                        print(f"New best model saved (val_loss: {avg_val_loss:.6f})")

                    # Log to file
                    with open(out_dir / "val_stats.json", "a") as f:
                        f.write(json.dumps({
                            "epoch": epoch + 1,
                            "train_loss": avg_train_loss,
                            "val_loss": avg_val_loss
                        }) + "\n")

                torch.cuda.empty_cache()

                # Back to training mode
                if self.stage_manager is not None:
                    self.stage_manager.set_module_modes(training=True)
                else:
                    self.model.train()
            else:
                # No validation - just save model each epoch (only on rank 0)
                if self.scheduler is not None:
                    self.scheduler.step()
                epoch_metrics = self._wandb_report_metrics("train", train_report)
                epoch_metrics.update(self._current_learning_rate_metrics())
                epoch_metrics.update({
                    "train/epoch": epoch + 1,
                    "train/iteration": len(train_losses),
                    "train/global_step": self.global_step,
                })
                self.wandb_logger.log(epoch_metrics, step=self.global_step)
                if is_main_process():
                    model_to_save = self._unwrapped_model
                    self._save_adapter_weights(model_to_save, out_dir / "last_lora_weights.pt")
                    self._save_stage_checkpoint(
                        out_dir, epoch=epoch + 1, loss=avg_train_loss, best=False
                    )
                if is_main_process() and self.moe_controller is not None:
                    record = {
                        "epoch": epoch + 1,
                        "stage": self.training_stage,
                        "learning_rates": epoch_learning_rates,
                        "train": train_report,
                        "validation": None,
                    }
                    self._write_statistics_record(
                        out_dir / "router_statistics.json", record
                    )
                    print("MoE epoch statistics:")
                    print(json.dumps(record, indent=2))

        # Synchronize before final save
        if self.multi_gpu:
            dist.barrier()

        # Final save (only on rank 0)
        if is_main_process():
            if best_val_loss < float("inf"):
                print(f"\n{'='*80}")
                print("Training complete!")
                print(f"{'='*80}")
                print(f"Best validation loss: {best_val_loss:.6f}")
                print(f"\nModels saved to {out_dir}:")
                print(f"  - best_lora_weights.pt (best validation loss)")
                print(f"  - last_lora_weights.pt (last epoch)")
                print("\nTo compute full metrics (mAP, cgF1) with NMS:")
                print(f"   python validate_sam3_lora.py \\")
                print(f"     --config <config_path> \\")
                print(f"     --weights {out_dir}/best_lora_weights.pt \\")
                print(f"     --val_data_dir <data_dir>/valid")
                print(f"{'='*80}")
            else:
                # If no validation, copy last to best
                import shutil
                last_path = out_dir / "last_lora_weights.pt"
                best_path = out_dir / "best_lora_weights.pt"
                if last_path.exists():
                    shutil.copy(last_path, best_path)
                if self.stage_manager is not None:
                    last_stage = out_dir / f"stage{self.training_stage}_last.pt"
                    best_stage = out_dir / self.stage_manager.checkpoint_name
                    if last_stage.exists():
                        shutil.copy(last_stage, best_stage)

                print(f"\n{'='*80}")
                print("Training complete!")
                print(f"{'='*80}")
                print(f"\nModels saved to {out_dir}:")
                print(f"  - best_lora_weights.pt (copy of last epoch)")
                print(f"  - last_lora_weights.pt (last epoch)")
                print("\nNo validation data - consider adding data/valid/ for better model selection")
                print(f"{'='*80}")

        # Cleanup distributed training
        if self.multi_gpu:
            cleanup_distributed()

def launch_distributed_training(args):
    """Launch training with multiple GPUs using torchrun subprocess."""
    import subprocess
    import sys

    devices = args.device
    num_gpus = len(devices)
    device_str = ",".join(map(str, devices))

    print(f"Launching distributed training on GPUs: {devices}")
    print(f"Number of processes: {num_gpus}")

    # Build the command
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={num_gpus}",
        "--master_port", str(args.master_port),
        sys.argv[0],  # This script
        "--config", args.config,
        "--device", *map(str, devices),
        "--_launched_by_torchrun"  # Internal flag to indicate we're in subprocess
    ]
    if getattr(args, "stage", None) is not None:
        cmd.extend(["--stage", str(args.stage)])
    if getattr(args, "resume", None):
        cmd.extend(["--resume", str(args.resume)])

    # Set environment variable for visible devices
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device_str
    if getattr(args, "wandb_api_key", None):
        # Keep the key out of the child command line. The child resolves it
        # from WANDB_API_KEY after the explicit parent CLI value wins here.
        env["WANDB_API_KEY"] = str(args.wandb_api_key)
    if getattr(args, "use_wandb", None):
        cmd.append("--use-wandb")
    if getattr(args, "wandb_watch_model", None):
        cmd.append("--wandb-watch-model")
    for option, attribute in (
        ("--wandb-project", "wandb_project"),
        ("--wandb-entity", "wandb_entity"),
        ("--wandb-name", "wandb_name"),
        ("--wandb-group", "wandb_group"),
        ("--wandb-mode", "wandb_mode"),
        ("--wandb-dir", "wandb_dir"),
        ("--wandb-log-interval", "wandb_log_interval"),
        ("--wandb-resume", "wandb_resume"),
        ("--wandb-run-id", "wandb_run_id"),
    ):
        value = getattr(args, attribute, None)
        if value not in (None, ""):
            cmd.extend([option, str(value)])
    wandb_tags = getattr(args, "wandb_tags", None)
    if wandb_tags:
        cmd.append("--wandb-tags")
        cmd.extend(map(str, wandb_tags))

    # Run the subprocess
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SAM3 with LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single GPU (default GPU 0):
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml

  Single GPU (specific GPU):
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml --device 1

  Multi-GPU (GPUs 0 and 1):
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml --device 0 1

  Multi-GPU (GPUs 0, 2, 3):
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml --device 0 2 3

  Multi-GPU (all 4 GPUs):
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml --device 0 1 2 3
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/full_lora_config.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--device",
        type=int,
        nargs="+",
        default=[0],
        help="GPU device ID(s) to use. Single value for single GPU, multiple values for multi-GPU. "
             "Example: --device 0 (single GPU), --device 0 1 2 (3 GPUs)"
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=29500,
        help="Master port for distributed training (default: 29500)"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (set automatically by torchrun)"
    )
    parser.add_argument(
        "--_launched_by_torchrun",
        action="store_true",
        help=argparse.SUPPRESS  # Hidden argument for internal use
    )
    args = parser.parse_args()

    # Determine if multi-GPU training is requested
    num_devices = len(args.device)
    is_torchrun_subprocess = args._launched_by_torchrun or "LOCAL_RANK" in os.environ

    if num_devices > 1 and not is_torchrun_subprocess:
        # Multi-GPU requested but not yet in torchrun - launch it
        launch_distributed_training(args)
    else:
        # Single GPU or already in torchrun subprocess
        multi_gpu = num_devices > 1 and is_torchrun_subprocess

        if not multi_gpu and num_devices == 1:
            # Single GPU mode - set the device
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device[0])
            print(f"Using single GPU: {args.device[0]}")

        trainer = SAM3TrainerNative(args.config, multi_gpu=multi_gpu)
        trainer.train()
