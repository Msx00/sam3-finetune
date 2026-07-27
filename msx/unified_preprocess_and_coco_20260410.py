# -*- coding: utf-8 -*-
"""
Unified 3D preprocessing + 2D COCO export for prostate segmentation.

What changed compared with the original paired rigid-preprocessing script:
1) Removed STL loading / ICP / rigid affine transform between MR and US.
2) Each case is processed independently.
3) Only unify voxel<->mm relation by resampling all cases to the same spacing/size/direction.
4) Optionally center each case by label centroid (or image center) on the unified grid.
5) Integrates the 2D slice PNG + COCO JSON export logic.

Typical usage:
    python unified_preprocess_and_coco.py

Output structure (example):
    OUT_ROOT/
      unified_3d/
        mr/train/1/
          image.nrrd
          label0.nrrd ...
          label_combined.nrrd
          meta.json
        mr/train/index_mapping.json
        us/train/1/
          ...
      mr-2d/
        train/1/slice_0000.png
        train/2/slice_0000.png
        train.json / val.json / test.json
      mr-mask-2d/
        train/1/slice_0000.png
        ...
      us-2d/
        train/1/slice_0000.png
        ...
      us-mask-2d/
        train/1/slice_0000.png
        ...
"""

import gc
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
from PIL import Image
import random
import numpy as np

random.seed(0)
np.random.seed(0)

# ====================== 1) Config ======================
DATA_ROOT = r"./dataset-tcia-shixingma-v2"   # contains mr/ and us/
OUT_ROOT = r"D:/zdataset-tcia-unified-seg"

SUBSETS = ["train", "val", "test"]
MODALITIES = ["mr", "us"]
MODALITIES = ["us"]
LABEL_NAMES = [f"label{i}.nrrd" for i in range(6)]
USE_COMPRESSION = True

# unified 3D grid (voxel-mm relation is fixed by these)
TARGET_SIZE = (175, 175, 175)          # (x, y, z)
TARGET_SPACING = (1.0, 1.0, 1.0)       # mm
TARGET_DIRECTION = (
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
)

# How to place each case onto the unified grid:
#   "label" -> use label0 centroid, else union centroid, else image center
#   "image" -> always use image center
#   "none"  -> only resample to unified spacing/size using original origin (no centering)
CENTER_MODE = "label"

# intensity normalization
MR_UPPER_PERCENTILE = 99.0
US_UPPER_PERCENTILE = 99.0

# 2D export config
EXPORT_2D = True
VIEW_BY_MODALITY = {
    "mr": "axial",
    "us": "axial",   # keep your previous setting; change to "axial" if needed
}
FINAL_2D_SIZE = (504, 504)
ONLY_SAVE_FOREGROUND_SLICES = True
PATIENT_FOLDER_START_INDEX = 1


# ====================== 2) IO utils ======================
def load_lines(json_path: str) -> List[str]:
    """Support JSON list or plain text (one path per line)."""
    if not os.path.exists(json_path):
        print(f"[Warn] list file not found: {json_path}")
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                return [str(x).replace("\\", "/").strip() for x in data if str(x).strip()]
        except json.JSONDecodeError:
            pass

    out = []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().replace("\\", "/")
            if s:
                out.append(s)
    return out


def find_main_and_labels(case_dir: str) -> Tuple[Optional[str], List[str]]:
    """Find image*.nrrd and ordered label0..label5 in a case folder."""
    if not os.path.isdir(case_dir):
        return None, []

    all_nrrd = [f for f in os.listdir(case_dir) if f.lower().endswith(".nrrd")]
    if not all_nrrd:
        return None, []

    main = None
    labels = []
    for f in all_nrrd:
        lf = f.lower()
        full = os.path.join(case_dir, f)
        if "image" in lf:
            main = full
        if lf.startswith("label") and lf.endswith(".nrrd"):
            labels.append(full)

    labels_map = {os.path.basename(p): p for p in labels}
    labels_ordered = [labels_map[name] for name in LABEL_NAMES if name in labels_map]

    if main is None:
        main = os.path.join(case_dir, all_nrrd[0])

    return main, labels_ordered


def sanitize_rel_path(rel_path: str) -> str:
    rel_path = rel_path.replace("\\", "/").strip("/")
    return rel_path.replace("/", "__")


# ====================== 3) Normalization ======================
def _to_sitk_float32(arr: np.ndarray, ref_img: sitk.Image) -> sitk.Image:
    out = sitk.GetImageFromArray(arr.astype(np.float32))
    out.CopyInformation(ref_img)
    return sitk.Cast(out, sitk.sitkFloat32)


def normalize_ct_or_cbct_itk(img: sitk.Image) -> sitk.Image:
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    arr = np.clip(arr, -1000.0, 1000.0)
    arr = (arr + 1000.0) / 2000.0
    arr = np.clip(arr, 0.0, 1.0)
    return _to_sitk_float32(arr, img)


def normalize_mr_itk(img: sitk.Image, upper_percentile: float = 99.0) -> sitk.Image:
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    hi = float(np.percentile(arr, upper_percentile))
    lo = float(arr.min())
    if hi <= lo + 1e-6:
        arr_n = np.zeros_like(arr, dtype=np.float32)
    else:
        arr = np.clip(arr, lo, hi)
        arr_n = (arr - lo) / (hi - lo)
        arr_n = np.clip(arr_n, 0.0, 1.0)
    return _to_sitk_float32(arr_n, img)


def normalize_us_itk(img: sitk.Image, upper_percentile: float = 99.0) -> sitk.Image:
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    hi = float(np.percentile(arr, upper_percentile))
    lo = float(arr.min())
    if hi <= lo + 1e-6:
        arr_n = np.zeros_like(arr, dtype=np.float32)
    else:
        arr = np.clip(arr, lo, hi)
        arr_n = (arr - lo) / (hi - lo)
        arr_n = np.clip(arr_n, 0.0, 1.0)
    return _to_sitk_float32(arr_n, img)


def normalize_image_itk(img: sitk.Image, modality: str) -> sitk.Image:
    m = modality.lower()
    if m in ["ct", "cbct"]:
        return normalize_ct_or_cbct_itk(img)
    if m in ["mr", "mri", "t1w", "t1ce", "t2w", "t2", "flair", "dess", "dixon", "fa", "md"]:
        return normalize_mr_itk(img, upper_percentile=MR_UPPER_PERCENTILE)
    if m in ["us", "ultrasound", "trus"]:
        return normalize_us_itk(img, upper_percentile=US_UPPER_PERCENTILE)
    raise ValueError(f"Unknown modality for normalization: {modality}")


def get_normalization_info(modality: str) -> Dict:
    m = modality.lower()
    if m in ["ct", "cbct"]:
        return {
            "modality": m,
            "rule": "clip_to_-1000_1000_then_scale_to_0_1",
            "hu_clip_range": [-1000.0, 1000.0],
        }
    if m in ["mr", "mri", "t1w", "t1ce", "t2w", "t2", "flair", "dess", "dixon", "fa", "md"]:
        return {
            "modality": m,
            "rule": "clip_max_to_99th_percentile_then_scale_to_0_1",
            "upper_percentile": float(MR_UPPER_PERCENTILE),
        }
    if m in ["us", "ultrasound", "trus"]:
        return {
            "modality": m,
            "rule": "clip_max_to_99th_percentile_then_scale_to_0_1",
            "upper_percentile": float(US_UPPER_PERCENTILE),
        }
    return {"modality": m, "rule": "unknown"}


# ====================== 4) Geometry helpers ======================
def get_physical_center_from_mask(mask_img: sitk.Image) -> Optional[np.ndarray]:
    fg = sitk.Cast(mask_img > 0, sitk.sitkUInt8)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(fg)
    if not stats.HasLabel(1):
        return None
    return np.array(stats.GetCentroid(1), dtype=np.float64)  # (x, y, z)


def get_physical_center_of_image(img: sitk.Image) -> np.ndarray:
    size = np.array(img.GetSize(), dtype=np.float64)
    center_index = (size - 1.0) / 2.0
    return np.array(img.TransformContinuousIndexToPhysicalPoint(center_index.tolist()), dtype=np.float64)


def compute_case_center(img: sitk.Image, labels: Dict[str, sitk.Image], center_mode: str = "label") -> np.ndarray:
    if center_mode == "none":
        return np.zeros(3, dtype=np.float64)
    if center_mode == "image":
        return get_physical_center_of_image(img)

    center = None
    if "label0.nrrd" in labels:
        center = get_physical_center_from_mask(labels["label0.nrrd"])

    if center is None and len(labels) > 0:
        union = None
        for lab in labels.values():
            fg = sitk.Cast(lab > 0, sitk.sitkUInt8)
            union = fg if union is None else (union | fg)
        center = get_physical_center_from_mask(union)

    if center is None:
        center = get_physical_center_of_image(img)

    return center


def make_reference_grid(
    size_xyz: Tuple[int, int, int],
    spacing_xyz: Tuple[float, float, float],
    direction_9: Tuple[float, ...],
    center_voxel_at_world_origin: bool = True,
) -> sitk.Image:
    size_xyz = [int(x) for x in size_xyz]
    spacing_xyz = [float(x) for x in spacing_xyz]
    D = np.array(direction_9, dtype=np.float64).reshape(3, 3)
    half = np.array([
        (size_xyz[0] - 1) / 2.0 * spacing_xyz[0],
        (size_xyz[1] - 1) / 2.0 * spacing_xyz[1],
        (size_xyz[2] - 1) / 2.0 * spacing_xyz[2],
    ], dtype=np.float64)

    origin = -D.dot(half) if center_voxel_at_world_origin else np.zeros(3, dtype=np.float64)

    ref = sitk.Image(size_xyz, sitk.sitkFloat32)
    ref.SetSpacing(tuple(spacing_xyz))
    ref.SetDirection(tuple(direction_9))
    ref.SetOrigin(tuple(origin.tolist()))
    return ref


def translate_out2in(offset_xyz: np.ndarray) -> sitk.TranslationTransform:
    tfm = sitk.TranslationTransform(3)
    tfm.SetOffset([float(offset_xyz[0]), float(offset_xyz[1]), float(offset_xyz[2])])
    return tfm


def resample_like(
    moving: sitk.Image,
    fixed: sitk.Image,
    transform_out2in: sitk.Transform,
    is_label: bool,
    default_value: float = 0.0,
) -> sitk.Image:
    interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    out_type = sitk.sitkUInt8 if is_label else sitk.sitkFloat32
    return sitk.Resample(moving, fixed, transform_out2in, interp, default_value, out_type)


# ====================== 5) Read / combine labels ======================
def read_case(case_dir: str, modality: str) -> Tuple[sitk.Image, Dict[str, sitk.Image], Dict]:
    main_path, label_paths = find_main_and_labels(case_dir)
    if main_path is None or (not os.path.exists(main_path)):
        raise FileNotFoundError(f"Missing image.nrrd in {case_dir}")

    img = sitk.ReadImage(main_path)
    img_norm = normalize_image_itk(img, modality=modality)

    labels: Dict[str, sitk.Image] = {}
    for lp in label_paths:
        name = os.path.basename(lp)
        try:
            lab = sitk.ReadImage(lp)
            lab_aligned = sitk.Resample(
                lab,
                img,
                sitk.Transform(),
                sitk.sitkNearestNeighbor,
                0,
                sitk.sitkUInt8,
            )
            labels[name] = lab_aligned
        except Exception as e:
            print(f"  [Warn] label read/resample failed: {lp} | {e}")

    info = {
        "case_dir": case_dir,
        "modality": modality,
        "normalization": get_normalization_info(modality),
        "image_file": os.path.basename(main_path),
        "size_xyz": list(img.GetSize()),
        "spacing_xyz": list(img.GetSpacing()),
        "origin_xyz": list(img.GetOrigin()),
        "direction_3x3": np.array(img.GetDirection(), dtype=float).reshape(3, 3).tolist(),
        "labels_found": sorted(list(labels.keys())),
    }
    return img_norm, labels, info


def combine_labels_to_multiclass(labels: Dict[str, sitk.Image], ref_img: sitk.Image) -> Optional[sitk.Image]:
    if len(labels) == 0:
        return None

    combined = sitk.Image(ref_img.GetSize(), sitk.sitkUInt8)
    combined.CopyInformation(ref_img)

    value = 1
    for name in sorted(labels.keys()):
        lab = sitk.Cast(labels[name] > 0, sitk.sitkUInt8)
        combined = sitk.Mask(combined, lab == 0)
        combined = combined + lab * int(value)
        value += 1

    return sitk.Cast(combined, sitk.sitkUInt8)


# ====================== 6) 3D preprocessing ======================
def process_single_case(
    case_dir: str,
    modality: str,
    out_case_dir: str,
    rel_path: str,
    subset: str,
    patient_folder: str,
) -> bool:
    os.makedirs(out_case_dir, exist_ok=True)

    img, labels, info = read_case(case_dir, modality=modality)
    case_center_phys = compute_case_center(img, labels, center_mode=CENTER_MODE)

    ref = make_reference_grid(
        size_xyz=TARGET_SIZE,
        spacing_xyz=TARGET_SPACING,
        direction_9=TARGET_DIRECTION,
        center_voxel_at_world_origin=True,
    )

    if CENTER_MODE == "none":
        tfm_ref2in = sitk.Transform(3, sitk.sitkIdentity)
    else:
        # x_in = x_out + case_center_phys
        # so unified world origin samples from case_center_phys
        tfm_ref2in = translate_out2in(case_center_phys)

    img_out = resample_like(
        moving=img,
        fixed=ref,
        transform_out2in=tfm_ref2in,
        is_label=False,
        default_value=0.0,
    )

    labels_out: Dict[str, sitk.Image] = {}
    for name, lab in labels.items():
        labels_out[name] = resample_like(
            moving=lab,
            fixed=ref,
            transform_out2in=tfm_ref2in,
            is_label=True,
            default_value=0,
        )

    combined_out = combine_labels_to_multiclass(labels_out, img_out)

    sitk.WriteImage(img_out, os.path.join(out_case_dir, "image.nrrd"), useCompression=USE_COMPRESSION)
    for name, lab in labels_out.items():
        sitk.WriteImage(lab, os.path.join(out_case_dir, name), useCompression=USE_COMPRESSION)
    if combined_out is not None:
        sitk.WriteImage(combined_out, os.path.join(out_case_dir, "label_combined.nrrd"), useCompression=USE_COMPRESSION)

    meta = {
        "subset": subset,
        "modality": modality,
        "patient_folder": patient_folder,
        "relative_path": rel_path,
        "case_dir_raw": case_dir,
        "preprocess_type": "independent_case_resample_only",
        "removed_steps": [
            "stl_loading",
            "icp_rigid_registration",
            "cross_modality_affine_transform",
            "paired_mr_us_alignment",
        ],
        "center_mode": CENTER_MODE,
        "case_center_input_phys_xyz": case_center_phys.tolist() if CENTER_MODE != "none" else None,
        "input": info,
        "output": {
            "size_xyz": list(TARGET_SIZE),
            "spacing_xyz": list(TARGET_SPACING),
            "direction_3x3": np.array(TARGET_DIRECTION, dtype=float).reshape(3, 3).tolist(),
            "origin_xyz": list(ref.GetOrigin()),
            "voxel_to_mm": {
                "spacing_xyz_mm": list(TARGET_SPACING),
                "note": "1 voxel step along each axis corresponds to spacing_xyz_mm in physical space.",
            },
            "physical_extent_xyz_mm": [
                float((TARGET_SIZE[0] - 1) * TARGET_SPACING[0]),
                float((TARGET_SIZE[1] - 1) * TARGET_SPACING[1]),
                float((TARGET_SIZE[2] - 1) * TARGET_SPACING[2]),
            ],
            "center_voxel_phys_should_be": [0.0, 0.0, 0.0],
        },
        "labels_found": sorted(list(labels_out.keys())),
        "note": (
            "This case is processed independently. No rigid/affine registration between MR and US is used. "
            "The only geometric normalization is resampling to a fixed voxel spacing/size, plus optional centering."
        ),
    }

    with open(os.path.join(out_case_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return True


# ====================== 7) 2D export helpers ======================
def sitk_image_to_numpy(img: sitk.Image, view: str) -> np.ndarray:
    arr = sitk.GetArrayFromImage(img)  # (z, y, x)
    if arr.ndim != 3:
        raise ValueError("Only 3D images are supported.")

    if view == "axial":
        slices = arr                    # (z, y, x)
    elif view == "coronal":
        slices = arr.transpose(1, 0, 2) # (y, z, x)
    elif view == "sagittal":
        slices = arr.transpose(2, 0, 1) # (x, z, y)
    else:
        raise ValueError(f"view error: {view}")

    return slices.astype(np.float32)


def normalize_intensity_2d(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    vmin = np.percentile(arr, 0.5)
    vmax = np.percentile(arr, 99.5)
    if vmax <= vmin:
        return np.zeros_like(arr, dtype=np.float32)
    arr = (arr - vmin) / (vmax - vmin)
    return np.clip(arr, 0.0, 1.0)


def mask_to_bbox_and_area(mask: np.ndarray) -> Optional[Tuple[list, float]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    area = float(mask.astype(np.uint8).sum())
    return [float(x_min), float(y_min), float(width), float(height)], area


def create_coco_structure(category_name: str):
    return {
        "info": {
            "description": None,
            "url": None,
            "version": None,
            "year": datetime.now().year,
            "contributor": None,
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        },
        "licenses": [{"url": None, "id": 0, "name": None}],
        "images": [],
        "annotations": [],
        "categories": [
            {"supercategory": None, "id": 0, "name": "_background_"},
            {"supercategory": None, "id": 1, "name": category_name},
        ],
    }


def list_processed_case_dirs(processed_subset_dir: str) -> List[str]:
    if not os.path.isdir(processed_subset_dir):
        return []

    def _sort_key(name: str):
        return (0, int(name)) if str(name).isdigit() else (1, str(name))

    out = []
    for name in sorted(os.listdir(processed_subset_dir), key=_sort_key):
        p = os.path.join(processed_subset_dir, name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "image.nrrd")):
            out.append(p)
    return out

def make_patient_folder_name(index_zero_based: int, start_index: int = PATIENT_FOLDER_START_INDEX) -> str:
    """Convert 0-based case index to patient folder name: 1, 2, 3, ..."""
    return str(int(index_zero_based) + int(start_index))

def atomic_save_json(data, filepath: str):
    """Write JSON atomically to avoid partial files when interrupted."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    tmp_path = filepath + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    if os.path.exists(filepath):
        os.remove(filepath)
    os.replace(tmp_path, filepath)

def export_coco_from_processed_subset(
    processed_subset_dir: str,
    json_output_path: str,
    output_base_dir: str,
    mask_output_base_dir: str,
    view: str,
    image_ext: str = ".png",
    category_name: str = "prostate",
    target_size: Tuple[int, int] = (504, 504),
):
    subset_name = os.path.splitext(os.path.basename(json_output_path))[0]
    img_save_dir = os.path.join(output_base_dir, subset_name)
    os.makedirs(img_save_dir, exist_ok=True)

    mask_save_dir = os.path.join(mask_output_base_dir, subset_name)
    os.makedirs(mask_save_dir, exist_ok=True)

    json_filename = os.path.basename(json_output_path)
    mask_json_output_path = os.path.join(mask_output_base_dir, json_filename)

    coco_img = create_coco_structure(category_name)
    coco_mask = create_coco_structure(category_name)

    img_id = 0
    case_dirs = list_processed_case_dirs(processed_subset_dir)

    print(f"[COCO] Start | src={processed_subset_dir} | cases={len(case_dirs)} | view={view} | final={target_size}")

    for case_idx, case_dir in enumerate(case_dirs):
        case_id = os.path.basename(case_dir)
        img_path = os.path.join(case_dir, "image.nrrd")
        lab_path = os.path.join(case_dir, "label_combined.nrrd")

        img_sitk = sitk.ReadImage(img_path)
        img_np = normalize_intensity_2d(sitk_image_to_numpy(img_sitk, view=view))

        label_np = None
        if os.path.exists(lab_path):
            lab_sitk = sitk.ReadImage(lab_path)
            label_np = sitk_image_to_numpy(lab_sitk, view=view).astype(np.int64)

        num_slices = img_np.shape[0]
        patient_img_dir = os.path.join(img_save_dir, case_id)
        patient_mask_dir = os.path.join(mask_save_dir, case_id)
        os.makedirs(patient_img_dir, exist_ok=True)
        os.makedirs(patient_mask_dir, exist_ok=True)

        for s_idx in range(num_slices):
            file_basename = f"slice_{s_idx:04d}{image_ext}"

            img_slice_raw = img_np[s_idx]
            label_slice_raw = label_np[s_idx] if label_np is not None else None

            img_uint8 = (img_slice_raw * 255.0).clip(0, 255).astype(np.uint8)
            img_final_pil = Image.fromarray(img_uint8).resize(target_size, resample=Image.BILINEAR)

            mask_final_pil = None
            mask_final_bin = None
            if label_slice_raw is not None:
                mask_uint8 = ((label_slice_raw > 0).astype(np.uint8) * 255)
                mask_final_pil = Image.fromarray(mask_uint8).resize(target_size, resample=Image.NEAREST)
                mask_final_bin = (np.array(mask_final_pil) > 127).astype(np.uint8)

            bbox_area = None
            if mask_final_bin is not None:
                bbox_area = mask_to_bbox_and_area(mask_final_bin)

            if ONLY_SAVE_FOREGROUND_SLICES and bbox_area is None:
                continue

            img_abs_path = os.path.join(patient_img_dir, file_basename)
            img_final_pil.save(img_abs_path)
            img_rel_path = f"{subset_name}/{case_id}/{file_basename}"

            coco_img["images"].append({
                "id": img_id,
                "license": 0,
                "url": None,
                "file_name": img_rel_path,
                "height": target_size[0],
                "width": target_size[1],
                "date_captured": None,
            })

            if bbox_area is not None:
                bbox, area = bbox_area
                coco_img["annotations"].append({
                    "id": img_id,
                    "category_id": 1,
                    "file_name": img_rel_path,
                    "segmentation": [[]],
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                })

            if mask_final_pil is not None:
                mask_abs_path = os.path.join(patient_mask_dir, file_basename)
                mask_final_pil.save(mask_abs_path)
                coco_mask["images"].append({
                    "id": img_id,
                    "license": 0,
                    "url": None,
                    "file_name": img_rel_path,
                    "height": target_size[0],
                    "width": target_size[1],
                    "date_captured": None,
                })

            img_id += 1

        atomic_save_json(coco_img, json_output_path)
        if len(coco_mask["images"]) > 0:
            atomic_save_json(coco_mask, mask_json_output_path)

        print(f"[COCO] Progress: {case_idx + 1}/{len(case_dirs)} | {case_id}")
        gc.collect()

    print(f"[Done] Images JSON: {json_output_path} ({len(coco_img['images'])} imgs)")


# ====================== 8) Main pipeline ======================
def preprocess_all_cases():
    processed_root = os.path.join(OUT_ROOT, "unified_3d")
    os.makedirs(processed_root, exist_ok=True)

    for modality in MODALITIES:
        modality_base = os.path.join(DATA_ROOT, modality)
        if not os.path.isdir(modality_base):
            print(f"[Skip] modality base not found: {modality_base}")
            continue

        for subset in SUBSETS:
            rel_list = load_lines(os.path.join(modality_base, f"{subset}.json"))
            if not rel_list:
                print(f"[Skip] empty subset: {modality}/{subset}")
                continue

            print(f"\n{'=' * 18} {modality.upper()} | {subset} | cases={len(rel_list)} {'=' * 18}")
            out_subset_dir = os.path.join(processed_root, modality, subset)
            os.makedirs(out_subset_dir, exist_ok=True)

            index_mapping = []
            for i, rel_path in enumerate(rel_list):
                case_dir = os.path.normpath(os.path.join(modality_base, "data", rel_path))
                patient_folder = make_patient_folder_name(i)
                case_alias = sanitize_rel_path(rel_path)
                out_case_dir = os.path.join(out_subset_dir, patient_folder)

                print(f"[{i + 1:4d}/{len(rel_list):4d}] {modality}/{subset}/{patient_folder} | {case_alias}")
                try:
                    ok = process_single_case(
                        case_dir=case_dir,
                        modality=modality,
                        out_case_dir=out_case_dir,
                        rel_path=rel_path,
                        subset=subset,
                        patient_folder=patient_folder,
                    )
                    if ok:
                        index_mapping.append({
                            "patient_folder": patient_folder,
                            "relative_path": rel_path,
                            "case_alias": case_alias,
                            "raw_case_dir": case_dir,
                        })
                    print(f"  -> {'OK' if ok else 'FAIL'} | {out_case_dir}")
                except Exception as e:
                    print(f"  -> [FAIL] {case_dir} | {e}")

            if index_mapping:
                atomic_save_json(index_mapping, os.path.join(out_subset_dir, "index_mapping.json"))

    return processed_root


def export_all_coco(processed_root: str):
    if not EXPORT_2D:
        print("[Info] EXPORT_2D=False -> skip 2D PNG + COCO export.")
        return

    for modality in MODALITIES:
        view = VIEW_BY_MODALITY[modality]
        img_out = os.path.join(OUT_ROOT, f"{modality}-2d")
        mask_out = os.path.join(OUT_ROOT, f"{modality}-mask-2d")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(mask_out, exist_ok=True)

        for subset in SUBSETS:
            processed_subset_dir = os.path.join(processed_root, modality, subset)
            if not os.path.isdir(processed_subset_dir):
                continue

            out_json = os.path.join(img_out, f"{subset}.json")
            export_coco_from_processed_subset(
                processed_subset_dir=processed_subset_dir,
                json_output_path=out_json,
                output_base_dir=img_out,
                mask_output_base_dir=mask_out,
                view=view,
                image_ext=".png",
                category_name=f"{modality}_prostate",
                target_size=FINAL_2D_SIZE,
            )


def main():
    processed_root = preprocess_all_cases()
    # processed_root = os.path.join(OUT_ROOT, "unified_3d")
    export_all_coco(processed_root)
    print("\nAll done.")


if __name__ == "__main__":
    main()
