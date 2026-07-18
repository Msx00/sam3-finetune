#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 unified_preprocess_and_coco_20260410.py 生成的 COCO 风格 JSON
转换为更标准的 COCO bbox JSON，并额外生成 SAM3 XYXY box prompt JSON。

输入 JSON 中 bbox:
    [x, y, width, height]

标准 COCO 输出 bbox:
    [x, y, width, height]

SAM3 box prompt 输出:
    [x1, y1, x2, y2]

示例：
python convert_sam3_coco_json.py \
    --input-json "/data/mr-2d/test.json" \
    --output-json "/data/mr-2d/test_fixed.json" \
    --boxes-json "/data/mr-2d/test_boxes_xyxy.json"
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

_MEDSAM_ROOT = Path(__file__).resolve().parent / "MedSAM3-main"
if str(_MEDSAM_ROOT) not in sys.path:
    sys.path.insert(0, str(_MEDSAM_ROOT))
from data.sample_index import normalize_sample_key


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"找不到JSON文件：{path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("输入JSON顶层必须是字典。")

    return data


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    temp_path.replace(path)


def normalize_file_name(file_name: str) -> str:
    """
    将Windows路径分隔符转换为JSON中更通用的正斜杠。
    """
    return str(file_name).replace("\\", "/").lstrip("/")


def canonical_training_file_name(file_name: str) -> str:
    """Make a COCO path relative to its split image root.

    ``train/1/slice_0000.png`` becomes ``1/slice_0000.png``. Numeric patient
    folder names are deliberately preserved without patient_XXX rewriting.
    """
    return normalize_sample_key(file_name)


def encode_mask_rle(mask_path: Path) -> Tuple[Dict[str, Any], float]:
    """Encode a binary medical mask as JSON-compatible compressed COCO RLE."""
    if not mask_path.is_file():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    rle = {"size": [int(v) for v in encoded["size"]], "counts": counts}
    return rle, float(mask.sum())


def resolve_mask_path(mask_root: Path, source_file_name: str) -> Path:
    """Support both modality mask roots and already split-specific roots."""
    source_path = Path(normalize_file_name(source_file_name))
    with_split = mask_root / source_path
    if with_split.is_file():
        return with_split
    return mask_root / Path(canonical_training_file_name(source_file_name))


def xywh_to_xyxy(bbox: List[float]) -> List[float]:
    """
    COCO:
        [x, y, width, height]

    SAM3:
        [x1, y1, x2, y2]
    """
    if len(bbox) != 4:
        raise ValueError(f"bbox长度必须为4，当前bbox={bbox}")

    x, y, width, height = map(float, bbox)

    # 第二份代码的width和height使用：
    # width = x_max - x_min + 1
    # height = y_max - y_min + 1
    #
    # 如果SAM3内部按连续坐标处理，通常使用x+w和y+h。
    return [
        x,
        y,
        x + width,
        y + height,
    ]


def validate_xywh(
    bbox: List[float],
    image_width: int,
    image_height: int,
) -> Tuple[List[float], bool]:
    """
    检查bbox，并将框裁剪到图像范围内。

    返回：
        修正后的bbox
        是否有效
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return [], False

    try:
        x, y, width, height = map(float, bbox)
    except (TypeError, ValueError):
        return [], False

    if width <= 0 or height <= 0:
        return [], False

    x1 = max(0.0, min(x, float(image_width)))
    y1 = max(0.0, min(y, float(image_height)))
    x2 = max(0.0, min(x + width, float(image_width)))
    y2 = max(0.0, min(y + height, float(image_height)))

    fixed_width = x2 - x1
    fixed_height = y2 - y1

    if fixed_width <= 0 or fixed_height <= 0:
        return [], False

    return [x1, y1, fixed_width, fixed_height], True


def get_foreground_category(
    input_categories: List[Dict[str, Any]],
    category_name: Optional[str],
) -> Dict[str, Any]:
    """
    去除_background_类别，只保留一个前景类别。
    """
    foreground_categories = []

    for category in input_categories:
        category_id = int(category.get("id", -1))
        name = str(category.get("name", ""))

        if category_id == 0:
            continue

        if name.lower() in {
            "_background_",
            "background",
            "bg",
        }:
            continue

        foreground_categories.append(category)

    if category_name:
        return {
            "id": 1,
            "name": category_name,
            "supercategory": "object",
        }

    if foreground_categories:
        old = foreground_categories[0]
        return {
            "id": 1,
            "name": str(old.get("name", "prostate")),
            "supercategory": str(
                old.get("supercategory") or "object"
            ),
        }

    return {
        "id": 1,
        "name": "prostate",
        "supercategory": "object",
    }


def convert_json(
    input_json: Path,
    output_json: Path,
    boxes_json: Path,
    category_name: Optional[str] = None,
    mask_root: Optional[Path] = None,
) -> None:
    source = load_json(input_json)

    source_images = source.get("images", [])
    source_annotations = source.get("annotations", [])
    source_categories = source.get("categories", [])

    if not isinstance(source_images, list):
        raise ValueError("输入JSON中的images必须是列表。")

    if not isinstance(source_annotations, list):
        raise ValueError("输入JSON中的annotations必须是列表。")

    # -----------------------------------------------------
    # 1. 整理images，并建立多种查找方式
    # -----------------------------------------------------
    fixed_images: List[Dict[str, Any]] = []

    image_by_id: Dict[int, Dict[str, Any]] = {}
    image_id_by_file_name: Dict[str, int] = {}

    for image in source_images:
        source_file_name = normalize_file_name(
            image.get("file_name", "")
        )

        if not source_file_name:
            print(
                f"[Warn] 跳过缺少file_name的图像：{image}"
            )
            continue

        new_image_id = len(fixed_images)
        old_image_id = int(
            image.get("id", new_image_id)
        )
        file_name = canonical_training_file_name(source_file_name)

        width = int(image.get("width", 0))
        height = int(image.get("height", 0))

        if width <= 0 or height <= 0:
            raise ValueError(
                f"图像宽高无效：file_name={file_name}, "
                f"width={width}, height={height}"
            )

        fixed_image = {
            "id": new_image_id,
            "file_name": file_name,
            "width": width,
            "height": height,
            "license": int(image.get("license", 0)),
            "date_captured": image.get(
                "date_captured", None
            ),
        }

        fixed_images.append(fixed_image)

        image_by_id[old_image_id] = {
            **fixed_image,
            "new_id": new_image_id,
        }

        image_id_by_file_name[source_file_name] = new_image_id
        image_id_by_file_name[file_name] = new_image_id

    # -----------------------------------------------------
    # 2. 修复annotations
    # -----------------------------------------------------
    fixed_annotations: List[Dict[str, Any]] = []

    # SAM3 box prompt JSON
    boxes_xyxy: Dict[str, List[List[float]]] = {}

    skipped_annotations = 0

    for source_ann in source_annotations:
        file_name = normalize_file_name(
            source_ann.get("file_name", "")
        )

        image_id: Optional[int] = None
        image_item: Optional[Dict[str, Any]] = None

        # 优先通过已有image_id关联
        if "image_id" in source_ann:
            old_image_id = int(source_ann["image_id"])
            image_item = image_by_id.get(old_image_id)

            if image_item is not None:
                image_id = int(image_item["new_id"])
                file_name = image_item["file_name"]

        # 第二份代码缺少image_id，所以通过file_name关联
        if image_id is None and file_name:
            image_id = image_id_by_file_name.get(file_name)

            if image_id is not None:
                image_item = fixed_images[image_id]
                file_name = image_item["file_name"]

        # 还有一种情况：
        # annotation id与image id相同
        if image_id is None and "id" in source_ann:
            old_possible_image_id = int(source_ann["id"])
            image_item = image_by_id.get(
                old_possible_image_id
            )

            if image_item is not None:
                image_id = int(image_item["new_id"])
                file_name = image_item["file_name"]

        if image_id is None or image_item is None:
            skipped_annotations += 1
            print(
                "[Warn] 无法找到annotation对应的图像，跳过："
                f"{source_ann}"
            )
            continue

        bbox = source_ann.get("bbox", [])

        bbox_fixed, valid = validate_xywh(
            bbox=bbox,
            image_width=int(image_item["width"]),
            image_height=int(image_item["height"]),
        )

        if not valid:
            skipped_annotations += 1
            print(
                f"[Warn] 无效bbox，跳过："
                f"file={file_name}, bbox={bbox}"
            )
            continue

        x, y, width, height = bbox_fixed

        # 优先保留原JSON中的真实mask面积。
        # 如果没有area，退化为bbox面积。
        area = source_ann.get("area", None)

        try:
            area = float(area)
        except (TypeError, ValueError):
            area = float(width * height)

        if area <= 0:
            area = float(width * height)

        annotation_id = len(fixed_annotations)

        segmentation: Any = source_ann.get("segmentation", [])
        if mask_root is not None:
            # Initial annotations retain the split-prefixed path; use it when
            # available so /mask-root/train/1/slice.png resolves naturally.
            source_name = normalize_file_name(source_ann.get("file_name", file_name))
            mask_path = resolve_mask_path(mask_root, source_name)
            segmentation, mask_area = encode_mask_rle(mask_path)
            area = mask_area

        fixed_annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,
            "bbox": [
                float(x),
                float(y),
                float(width),
                float(height),
            ],
            "area": float(area),
            "segmentation": segmentation,
            "iscrowd": int(
                source_ann.get("iscrowd", 0)
            ),
        }

        fixed_annotations.append(fixed_annotation)

        xyxy = xywh_to_xyxy(
            fixed_annotation["bbox"]
        )

        # 保存标准路径键
        boxes_xyxy.setdefault(file_name, []).append(xyxy)

        # 再保存一个去掉train/val/test前缀的键。
        # 例如：
        # train/1/slice_0060.png
        # -> 1/slice_0060.png
        file_parts = Path(file_name).parts

        if (
            len(file_parts) >= 3
            and file_parts[0].lower()
            in {"train", "val", "test"}
        ):
            no_subset_name = Path(
                *file_parts[1:]
            ).as_posix()

            boxes_xyxy.setdefault(no_subset_name, []).append(xyxy)

    # -----------------------------------------------------
    # 3. 只保留前景类别
    # -----------------------------------------------------
    foreground_category = get_foreground_category(
        input_categories=source_categories,
        category_name=category_name,
    )

    fixed_coco = {
        "info": {
            "description": (
                "Standard COCO bbox annotations converted "
                "from unified prostate 2D export"
            ),
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        },
        "licenses": source.get(
            "licenses",
            [
                {
                    "id": 0,
                    "name": None,
                    "url": None,
                }
            ],
        ),
        "images": fixed_images,
        "annotations": fixed_annotations,
        "categories": [foreground_category],
    }

    save_json(fixed_coco, output_json)
    save_json(boxes_xyxy, boxes_json)

    print("\n[Done] 转换完成")
    print(f"输入JSON：{input_json}")
    print(f"COCO输出：{output_json}")
    print(f"SAM3框输出：{boxes_json}")
    print(f"图像数量：{len(fixed_images)}")
    print(f"标注数量：{len(fixed_annotations)}")
    print(f"跳过标注：{skipped_annotations}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "修复unified_preprocess_and_coco生成的JSON，"
            "输出标准COCO bbox JSON和SAM3 XYXY box JSON。"
        )
    )

    parser.add_argument(
        "--input-json",
        required=True,
        help="第二份代码生成的train.json/val.json/test.json",
    )

    parser.add_argument(
        "--mask-root",
        default=None,
        help=(
            "Mask root such as /data/mr-mask-2d. When provided, each mask PNG "
            "is encoded into annotations.segmentation as compressed COCO RLE."
        ),
    )

    parser.add_argument(
        "--output-json",
        required=True,
        help="修复后的标准COCO JSON",
    )

    parser.add_argument(
        "--boxes-json",
        required=True,
        help="输出SAM3 XYXY box prompt JSON",
    )

    parser.add_argument(
        "--category-name",
        default=None,
        help=(
            "前景类别名称，例如prostate。"
            "不设置时沿用输入JSON的前景类别名称。"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    convert_json(
        input_json=Path(args.input_json)
        .expanduser()
        .resolve(),
        output_json=Path(args.output_json)
        .expanduser()
        .resolve(),
        boxes_json=Path(args.boxes_json)
        .expanduser()
        .resolve(),
        category_name=args.category_name,
        mask_root=(
            Path(args.mask_root).expanduser().resolve()
            if args.mask_root is not None
            else None
        ),
    )


if __name__ == "__main__":
    main()
