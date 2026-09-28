#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
阳光玫瑰葡萄花穗完整性判断与分割挑战赛——官方结果评估脚本 V2

核心规则
--------
1. result.json:
   category_id=0 -> 完整花穗
   category_id=1 -> 不完整花穗

2. GT COCO:
   category_id=1, name="0" -> 完整花穗
   category_id=2, name="1" -> 不完整花穗

3. 主指标:
   COCO Mask mAP@[0.50:0.95]
   IoU 阈值固定为：
   [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
   Recall: 101 点
   maxDets 固定为 [1, 10, 100]，不允许调用者修改。

4. 类别无关 ignore_regions:
   - 普通 GT 始终优先匹配。
   - 对某一 IoU 阈值下“未匹配普通 GT”的预测，计算：
       IgnoreOverlap = area(pred ∩ union(ignore_regions)) / area(pred)
   - IgnoreOverlap >= 0.50 时，该预测在该 IoU 阈值下：
       不计 TP，也不计 FP。
   - ignore 区域不进入 GT 召回率/AP 分母。
   - 同一 ignore 区域对两个预测类别都生效。

5. 可迁移性:
   默认路径全部相对于本脚本所在目录，不包含任何用户本地绝对路径。

默认目录
--------
ground_truth/instances_test.json
ground_truth/ignore_regions_test.json
submission/result.json
output/score.json
output/error.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

if __package__:
    from .rle_validation import validate_compressed_rle
else:
    from rle_validation import validate_compressed_rle

try:
    from pycocotools import mask as mask_utils
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError as exc:
    raise SystemExit(
        "缺少依赖。请使用官方 requirements_evaluator.txt 安装固定版本依赖。"
    ) from exc


# ============================================================================
# 固定赛题参数
# ============================================================================

EXPECTED_VERSION = "1.0"

SUBMISSION_TO_GT_CATEGORY = {
    0: 1,  # 完整花穗
    1: 2,  # 不完整花穗
}

GT_TO_SUBMISSION_CATEGORY = {
    1: 0,
    2: 1,
}

SUBMISSION_CATEGORY_NAMES = {
    0: "完整花穗",
    1: "不完整花穗",
}

EXPECTED_GT_CATEGORY_NAMES = {
    1: "0",
    2: "1",
}

# 不使用 arange，避免 0.75 等阈值的浮点查找问题。
IOU_THRESHOLDS = np.asarray(
    [0.50, 0.55, 0.60, 0.65, 0.70,
     0.75, 0.80, 0.85, 0.90, 0.95],
    dtype=np.float64,
)

RECALL_THRESHOLDS = np.linspace(
    0.0,
    1.0,
    101,
    dtype=np.float64,
)

# 固定，不向命令行开放。
MAX_DETS = [1, 10, 100]

IGNORE_OVERLAP_METRIC = "intersection_over_prediction"
IGNORE_OVERLAP_THRESHOLD = 0.50


# ============================================================================
# 可迁移路径
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_GT_JSON = Path("ground_truth") / "instances_test.json"
DEFAULT_IGNORE_JSON = Path("ground_truth") / "ignore_regions_test.json"
DEFAULT_SUBMISSION_JSON = Path("submission") / "result.json"
DEFAULT_OUTPUT_JSON = Path("output") / "score.json"
DEFAULT_ERROR_JSON = Path("output") / "error.json"


def resolve_portable_path(path: Path) -> Path:
    """
    相对路径统一相对于评分脚本所在目录。
    显式传入绝对路径也可以运行，但代码本身不绑定任何本地绝对目录。
    """
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


# ============================================================================
# 异常
# ============================================================================

class SubmissionValidationError(ValueError):
    """选手提交 result.json 不符合赛题格式。"""


class GroundTruthValidationError(ValueError):
    """平台私有 GT 或 ignore_regions 文件不符合官方格式。"""


# ============================================================================
# JSON / 文件安全
# ============================================================================

def reject_duplicate_json_keys(
    pairs: Sequence[Tuple[str, Any]],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    for key, value in pairs:
        if key in result:
            raise SubmissionValidationError(
                f"JSON 中存在重复字段：{key!r}"
            )
        result[key] = value

    return result


def load_json_strict(
    path: Path,
    *,
    submission: bool = False,
) -> Any:
    if not path.exists():
        raise FileNotFoundError(
            f"文件不存在：{path}"
        )

    if not path.is_file():
        raise ValueError(
            f"路径不是文件：{path}"
        )

    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
        ) as file:
            return json.load(
                file,
                object_pairs_hook=reject_duplicate_json_keys,
            )

    except SubmissionValidationError:
        raise

    except json.JSONDecodeError as exc:
        prefix = "提交文件" if submission else "JSON 文件"

        raise SubmissionValidationError(
            f"{prefix}不是合法 JSON："
            f"第 {exc.lineno} 行，第 {exc.colno} 列，{exc.msg}"
        ) from exc

    except UnicodeDecodeError as exc:
        raise SubmissionValidationError(
            f"文件不是有效 UTF-8 编码：{path}"
        ) from exc


def json_safe(value: Any) -> Any:
    """
    在写 JSON 前递归转换 NumPy 类型，彻底解决 numpy.int64 /
    numpy.float64 等无法被标准 json 序列化的问题。
    """
    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(
                f"评分结果包含非有限浮点数：{number}"
            )
        return number

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.ndarray):
        return [
            json_safe(item)
            for item in value.tolist()
        ]

    if isinstance(value, Path):
        return value.as_posix()

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                f"评分结果包含非有限浮点数：{value}"
            )
        return value

    return value


def atomic_write_json(
    path: Path,
    data: Any,
) -> None:
    """
    先完成数据转换和临时文件写入，最后 os.replace 原子替换。
    即便写出失败，也不会留下“半截 score.json”。
    """
    safe_data = json_safe(data)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_name(
        f".{path.name}.tmp-{os.getpid()}"
    )

    try:
        with temp_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                safe_data,
                file,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )

            file.flush()
            os.fsync(
                file.fileno()
            )

        os.replace(
            temp_path,
            path,
        )

    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def safe_unlink(path: Path | None) -> None:
    if path is None:
        return

    try:
        if path.exists():
            path.unlink()
    except OSError:
        # 删除失败时让后续写入自行报错，避免静默吞掉。
        pass


# ============================================================================
# 通用校验
# ============================================================================

def is_int_not_bool(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
    )


def is_finite_number_not_bool(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def require_exact_keys(
    obj: Mapping[str, Any],
    required: set[str],
    location: str,
) -> None:
    actual = set(
        obj.keys()
    )

    missing = sorted(
        required - actual
    )

    extra = sorted(
        actual - required
    )

    messages: List[str] = []

    if missing:
        messages.append(
            f"缺少字段 {missing}"
        )

    if extra:
        messages.append(
            f"包含未允许字段 {extra}"
        )

    if messages:
        raise SubmissionValidationError(
            f"{location}："
            + "；".join(messages)
        )


# ============================================================================
# GT COCO 校验
# ============================================================================

def validate_gt_dataset(
    gt_data: Any,
) -> Tuple[
    Dict[str, Dict[str, Any]],
    Dict[int, Dict[str, Any]],
]:
    if not isinstance(gt_data, dict):
        raise GroundTruthValidationError(
            "真实标注 JSON 顶层必须是对象。"
        )

    for key in (
        "images",
        "annotations",
        "categories",
    ):
        if (
            key not in gt_data
            or not isinstance(
                gt_data[key],
                list,
            )
        ):
            raise GroundTruthValidationError(
                f"真实标注缺少列表字段：{key}"
            )

    images_by_name: Dict[
        str,
        Dict[str, Any],
    ] = {}

    images_by_id: Dict[
        int,
        Dict[str, Any],
    ] = {}

    # ------------------------------------------------------------------
    # images
    # ------------------------------------------------------------------
    for index, image in enumerate(
        gt_data["images"]
    ):
        if not isinstance(image, dict):
            raise GroundTruthValidationError(
                f"真实标注 images[{index}] 必须是对象。"
            )

        for key in (
            "id",
            "file_name",
            "width",
            "height",
        ):
            if key not in image:
                raise GroundTruthValidationError(
                    f"真实标注 images[{index}] 缺少字段 {key}"
                )

        image_id = image["id"]
        file_name = image["file_name"]
        width = image["width"]
        height = image["height"]

        if not is_int_not_bool(image_id):
            raise GroundTruthValidationError(
                f"真实标注 images[{index}].id 必须是整数。"
            )

        if (
            not isinstance(file_name, str)
            or not file_name
        ):
            raise GroundTruthValidationError(
                f"真实标注 images[{index}].file_name 无效。"
            )

        if (
            not is_int_not_bool(width)
            or width <= 0
        ):
            raise GroundTruthValidationError(
                f"真实标注 {file_name} 的 width 无效。"
            )

        if (
            not is_int_not_bool(height)
            or height <= 0
        ):
            raise GroundTruthValidationError(
                f"真实标注 {file_name} 的 height 无效。"
            )

        if image_id in images_by_id:
            raise GroundTruthValidationError(
                f"真实标注存在重复 image id：{image_id}"
            )

        if file_name in images_by_name:
            raise GroundTruthValidationError(
                f"真实标注存在重复 file_name：{file_name}"
            )

        images_by_id[image_id] = image
        images_by_name[file_name] = image

    # ------------------------------------------------------------------
    # categories
    # ------------------------------------------------------------------
    categories_by_id: Dict[
        int,
        Dict[str, Any],
    ] = {}

    for index, category in enumerate(
        gt_data["categories"]
    ):
        if not isinstance(category, dict):
            raise GroundTruthValidationError(
                f"真实标注 categories[{index}] 必须是对象。"
            )

        if (
            "id" not in category
            or "name" not in category
        ):
            raise GroundTruthValidationError(
                f"真实标注 categories[{index}] 缺少 id 或 name。"
            )

        category_id = category["id"]

        if not is_int_not_bool(category_id):
            raise GroundTruthValidationError(
                f"真实标注 categories[{index}].id 必须是整数。"
            )

        if category_id in categories_by_id:
            raise GroundTruthValidationError(
                f"真实标注存在重复 category id：{category_id}"
            )

        categories_by_id[
            category_id
        ] = category

    if set(categories_by_id) != set(
        EXPECTED_GT_CATEGORY_NAMES
    ):
        raise GroundTruthValidationError(
            "真实标注 categories 必须且只能包含 category_id=1、2。"
        )

    for (
        category_id,
        expected_name,
    ) in EXPECTED_GT_CATEGORY_NAMES.items():

        actual_name = str(
            categories_by_id[
                category_id
            ].get(
                "name",
                "",
            )
        ).strip()

        if actual_name != expected_name:
            raise GroundTruthValidationError(
                f"真实标注 category_id={category_id} 的 name 应为 "
                f"{expected_name!r}，实际为 {actual_name!r}"
            )

    # ------------------------------------------------------------------
    # annotations
    # ------------------------------------------------------------------
    annotation_ids: set[int] = set()

    for index, annotation in enumerate(
        gt_data["annotations"]
    ):
        if not isinstance(annotation, dict):
            raise GroundTruthValidationError(
                f"真实标注 annotations[{index}] 必须是对象。"
            )

        for key in (
            "id",
            "image_id",
            "category_id",
            "segmentation",
            "area",
            "bbox",
            "iscrowd",
        ):
            if key not in annotation:
                raise GroundTruthValidationError(
                    f"真实标注 annotations[{index}] 缺少字段 {key}"
                )

        annotation_id = annotation["id"]
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]

        if not is_int_not_bool(
            annotation_id
        ):
            raise GroundTruthValidationError(
                f"真实标注 annotations[{index}].id 必须是整数。"
            )

        if annotation_id in annotation_ids:
            raise GroundTruthValidationError(
                f"真实标注存在重复 annotation id：{annotation_id}"
            )

        annotation_ids.add(
            annotation_id
        )

        if image_id not in images_by_id:
            raise GroundTruthValidationError(
                f"真实标注 annotation id={annotation_id} "
                f"引用了不存在的 image_id={image_id}"
            )

        if category_id not in EXPECTED_GT_CATEGORY_NAMES:
            raise GroundTruthValidationError(
                f"真实标注 annotation id={annotation_id} "
                f"使用未允许类别 category_id={category_id}"
            )

        if int(annotation["iscrowd"]) != 0:
            raise GroundTruthValidationError(
                f"真实普通标注 annotation id={annotation_id} "
                "必须 iscrowd=0；忽略区域应放入 ignore_regions 文件。"
            )

    return (
        images_by_name,
        images_by_id,
    )


# ============================================================================
# Ignore regions 校验与 RLE 合并
# ============================================================================

def segmentation_to_rle(
    segmentation: Any,
    height: int,
    width: int,
    location: str,
) -> Dict[str, Any]:
    """
    支持：
    - COCO polygon list
    - compressed / uncompressed RLE object
    """
    if isinstance(
        segmentation,
        list,
    ):
        if not segmentation:
            raise GroundTruthValidationError(
                f"{location}: segmentation polygon 为空。"
            )

        try:
            rles = mask_utils.frPyObjects(
                segmentation,
                height,
                width,
            )
        except Exception as exc:
            raise GroundTruthValidationError(
                f"{location}: polygon 无法转换为 RLE：{exc}"
            ) from exc

        if isinstance(rles, list):
            if not rles:
                raise GroundTruthValidationError(
                    f"{location}: polygon 转换结果为空。"
                )

            return mask_utils.merge(
                rles,
                intersect=False,
            )

        return rles

    if isinstance(
        segmentation,
        dict,
    ):
        size = segmentation.get(
            "size"
        )

        counts = segmentation.get(
            "counts"
        )

        if size != [
            height,
            width,
        ]:
            raise GroundTruthValidationError(
                f"{location}: RLE size 应为 [{height}, {width}]，"
                f"实际为 {size!r}。"
            )

        # compressed RLE
        if isinstance(
            counts,
            str,
        ):
            return {
                "size": [
                    height,
                    width,
                ],
                "counts": counts.encode(
                    "utf-8"
                ),
            }

        if isinstance(
            counts,
            bytes,
        ):
            return {
                "size": [
                    height,
                    width,
                ],
                "counts": counts,
            }

        # uncompressed RLE
        if isinstance(
            counts,
            list,
        ):
            try:
                return mask_utils.frPyObjects(
                    segmentation,
                    height,
                    width,
                )
            except Exception as exc:
                raise GroundTruthValidationError(
                    f"{location}: uncompressed RLE 无效：{exc}"
                ) from exc

        raise GroundTruthValidationError(
            f"{location}: RLE counts 类型无效。"
        )

    raise GroundTruthValidationError(
        f"{location}: segmentation 必须是 polygon list 或 RLE object。"
    )


def validate_and_build_ignore_unions(
    ignore_data: Any,
    images_by_id: Mapping[
        int,
        Dict[str, Any],
    ],
) -> Tuple[
    Dict[int, Dict[str, Any]],
    Dict[str, Any],
]:
    if not isinstance(
        ignore_data,
        dict,
    ):
        raise GroundTruthValidationError(
            "ignore_regions JSON 顶层必须是对象。"
        )

    if (
        "ignore_regions" not in ignore_data
        or not isinstance(
            ignore_data["ignore_regions"],
            list,
        )
    ):
        raise GroundTruthValidationError(
            "ignore_regions JSON 必须包含列表字段 ignore_regions。"
        )

    if "overlap_metric" in ignore_data:
        if (
            ignore_data["overlap_metric"]
            != IGNORE_OVERLAP_METRIC
        ):
            raise GroundTruthValidationError(
                "ignore_regions.overlap_metric 必须为 "
                f"{IGNORE_OVERLAP_METRIC!r}。"
            )

    if "overlap_threshold" in ignore_data:
        threshold = ignore_data[
            "overlap_threshold"
        ]

        if not is_finite_number_not_bool(
            threshold
        ):
            raise GroundTruthValidationError(
                "ignore_regions.overlap_threshold 必须是数值。"
            )

        if abs(
            float(threshold)
            - IGNORE_OVERLAP_THRESHOLD
        ) > 1e-12:
            raise GroundTruthValidationError(
                "ignore_regions.overlap_threshold 必须固定为 0.50。"
            )

    # 如果文件带 images 元数据，则与 GT 严格核对。
    if "images" in ignore_data:
        if not isinstance(
            ignore_data["images"],
            list,
        ):
            raise GroundTruthValidationError(
                "ignore_regions.images 必须是列表。"
            )

        ignore_image_ids = set()

        for index, image in enumerate(
            ignore_data["images"]
        ):
            if not isinstance(
                image,
                dict,
            ):
                raise GroundTruthValidationError(
                    f"ignore_regions.images[{index}] 必须是对象。"
                )

            image_id = image.get(
                "id"
            )

            if image_id not in images_by_id:
                raise GroundTruthValidationError(
                    f"ignore_regions.images[{index}] "
                    f"引用不存在的 image_id={image_id}"
                )

            if image_id in ignore_image_ids:
                raise GroundTruthValidationError(
                    f"ignore_regions.images 中 image_id={image_id} 重复。"
                )

            ignore_image_ids.add(
                image_id
            )

            gt_image = images_by_id[
                image_id
            ]

            for key in (
                "file_name",
                "width",
                "height",
            ):
                if (
                    key in image
                    and image[key] != gt_image[key]
                ):
                    raise GroundTruthValidationError(
                        f"ignore_regions.images[{index}].{key} "
                        "与 GT COCO 不一致。"
                    )

        if ignore_image_ids != set(
            images_by_id
        ):
            raise GroundTruthValidationError(
                "ignore_regions.images 必须与 GT COCO images 完全一致。"
            )

    region_ids: set[int] = set()

    rles_by_image: Dict[
        int,
        List[Dict[str, Any]],
    ] = {}

    images_with_ignore: set[int] = set()

    for index, region in enumerate(
        ignore_data["ignore_regions"]
    ):
        location = (
            f"ignore_regions[{index}]"
        )

        if not isinstance(
            region,
            dict,
        ):
            raise GroundTruthValidationError(
                f"{location} 必须是对象。"
            )

        for key in (
            "id",
            "image_id",
            "file_name",
            "segmentation",
        ):
            if key not in region:
                raise GroundTruthValidationError(
                    f"{location} 缺少字段 {key}。"
                )

        region_id = region["id"]
        image_id = region["image_id"]

        if not is_int_not_bool(
            region_id
        ):
            raise GroundTruthValidationError(
                f"{location}.id 必须是整数。"
            )

        if region_id in region_ids:
            raise GroundTruthValidationError(
                f"ignore region id 重复：{region_id}"
            )

        region_ids.add(
            region_id
        )

        if image_id not in images_by_id:
            raise GroundTruthValidationError(
                f"{location} 引用不存在的 image_id={image_id}"
            )

        gt_image = images_by_id[
            image_id
        ]

        if (
            region["file_name"]
            != gt_image["file_name"]
        ):
            raise GroundTruthValidationError(
                f"{location}.file_name 与 GT COCO 不一致。"
            )

        height = int(
            gt_image["height"]
        )

        width = int(
            gt_image["width"]
        )

        rle = segmentation_to_rle(
            region["segmentation"],
            height,
            width,
            location,
        )

        try:
            area = float(
                mask_utils.area(
                    rle
                )
            )
        except Exception as exc:
            raise GroundTruthValidationError(
                f"{location}: ignore mask 无法计算 area：{exc}"
            ) from exc

        if (
            not math.isfinite(area)
            or area <= 0.0
        ):
            raise GroundTruthValidationError(
                f"{location}: ignore mask 面积必须 > 0。"
            )

        # 若文件中给了 area / bbox，则顺带校验，避免私有真值损坏。
        if "area" in region:
            if abs(
                float(region["area"])
                - area
            ) > max(
                1e-3,
                area * 1e-5,
            ):
                raise GroundTruthValidationError(
                    f"{location}.area 与 segmentation 不一致。"
                )

        if "bbox" in region:
            calc_bbox = np.asarray(
                mask_utils.toBbox(
                    rle
                ),
                dtype=np.float64,
            ).reshape(
                -1
            )

            given_bbox = np.asarray(
                region["bbox"],
                dtype=np.float64,
            ).reshape(
                -1
            )

            if (
                calc_bbox.size != 4
                or given_bbox.size != 4
                or not np.allclose(
                    calc_bbox,
                    given_bbox,
                    atol=1e-3,
                    rtol=0.0,
                )
            ):
                raise GroundTruthValidationError(
                    f"{location}.bbox 与 segmentation 不一致。"
                )

        rles_by_image.setdefault(
            image_id,
            [],
        ).append(
            rle
        )

        images_with_ignore.add(
            image_id
        )

    ignore_unions: Dict[
        int,
        Dict[str, Any],
    ] = {}

    for (
        image_id,
        rles,
    ) in rles_by_image.items():

        if len(rles) == 1:
            ignore_unions[
                image_id
            ] = rles[0]
        else:
            ignore_unions[
                image_id
            ] = mask_utils.merge(
                rles,
                intersect=False,
            )

    summary = {
        "overlap_metric": IGNORE_OVERLAP_METRIC,
        "overlap_threshold": IGNORE_OVERLAP_THRESHOLD,
        "num_ignore_regions": len(
            ignore_data["ignore_regions"]
        ),
        "num_images_with_ignore": len(
            images_with_ignore
        ),
    }

    return (
        ignore_unions,
        summary,
    )


# ============================================================================
# result.json 校验
# ============================================================================

def validate_and_convert_submission(
    submission_data: Any,
    images_by_name: Mapping[
        str,
        Dict[str, Any],
    ],
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    if not isinstance(
        submission_data,
        dict,
    ):
        raise SubmissionValidationError(
            "result.json 顶层必须是对象。"
        )

    require_exact_keys(
        submission_data,
        {
            "version",
            "results",
        },
        "result.json 顶层",
    )

    if (
        submission_data["version"]
        != EXPECTED_VERSION
    ):
        raise SubmissionValidationError(
            f"version 必须为 {EXPECTED_VERSION!r}，"
            f"实际为 {submission_data['version']!r}"
        )

    results = submission_data[
        "results"
    ]

    if not isinstance(
        results,
        list,
    ):
        raise SubmissionValidationError(
            "results 必须是数组。"
        )

    expected_names = set(
        images_by_name
    )

    submitted_names: set[str] = set()

    detections: List[
        Dict[str, Any]
    ] = []

    per_category_count = {
        0: 0,
        1: 0,
    }

    per_image_count: Dict[
        str,
        int,
    ] = {}

    for (
        result_index,
        result_item,
    ) in enumerate(
        results
    ):
        location = (
            f"results[{result_index}]"
        )

        if not isinstance(
            result_item,
            dict,
        ):
            raise SubmissionValidationError(
                f"{location} 必须是对象。"
            )

        require_exact_keys(
            result_item,
            {
                "image_id",
                "instances",
            },
            location,
        )

        image_name = result_item[
            "image_id"
        ]

        instances = result_item[
            "instances"
        ]

        if (
            not isinstance(
                image_name,
                str,
            )
            or not image_name
        ):
            raise SubmissionValidationError(
                f"{location}.image_id 必须是非空字符串。"
            )

        if image_name in submitted_names:
            raise SubmissionValidationError(
                f"测试图像重复出现：{image_name}"
            )

        if image_name not in expected_names:
            raise SubmissionValidationError(
                f"提交包含测试集之外的图像：{image_name}"
            )

        if not isinstance(
            instances,
            list,
        ):
            raise SubmissionValidationError(
                f"{location}.instances 必须是数组。"
            )

        submitted_names.add(
            image_name
        )

        gt_image = images_by_name[
            image_name
        ]

        gt_image_id = int(
            gt_image["id"]
        )

        expected_height = int(
            gt_image["height"]
        )

        expected_width = int(
            gt_image["width"]
        )

        per_image_count[
            image_name
        ] = len(
            instances
        )

        for (
            instance_index,
            instance,
        ) in enumerate(
            instances
        ):
            instance_location = (
                f"{location}.instances[{instance_index}]"
            )

            if not isinstance(
                instance,
                dict,
            ):
                raise SubmissionValidationError(
                    f"{instance_location} 必须是对象。"
                )

            require_exact_keys(
                instance,
                {
                    "category_id",
                    "score",
                    "segmentation",
                },
                instance_location,
            )

            category_id = instance[
                "category_id"
            ]

            score = instance[
                "score"
            ]

            segmentation = instance[
                "segmentation"
            ]

            if (
                not is_int_not_bool(
                    category_id
                )
                or category_id not in (
                    0,
                    1,
                )
            ):
                raise SubmissionValidationError(
                    f"{instance_location}.category_id "
                    "只能是整数 0 或 1。"
                )

            if not is_finite_number_not_bool(
                score
            ):
                raise SubmissionValidationError(
                    f"{instance_location}.score 必须是有限数值。"
                )

            score_float = float(
                score
            )

            if not (
                0.0
                <= score_float
                <= 1.0
            ):
                raise SubmissionValidationError(
                    f"{instance_location}.score 必须位于 [0,1]，"
                    f"实际为 {score_float}。"
                )

            if not isinstance(
                segmentation,
                dict,
            ):
                raise SubmissionValidationError(
                    f"{instance_location}.segmentation 必须是对象。"
                )

            require_exact_keys(
                segmentation,
                {
                    "size",
                    "counts",
                },
                f"{instance_location}.segmentation",
            )

            size = segmentation[
                "size"
            ]

            counts = segmentation[
                "counts"
            ]

            if (
                not isinstance(
                    size,
                    list,
                )
                or len(size) != 2
                or not all(
                    is_int_not_bool(
                        value
                    )
                    for value in size
                )
            ):
                raise SubmissionValidationError(
                    f"{instance_location}.segmentation.size "
                    "必须是 [height,width] 两个整数。"
                )

            if size != [
                expected_height,
                expected_width,
            ]:
                raise SubmissionValidationError(
                    f"{instance_location}.segmentation.size "
                    f"与 {image_name} 尺寸不一致："
                    f"应为 [{expected_height}, {expected_width}]，"
                    f"实际为 {size}。"
                )

            if (
                not isinstance(
                    counts,
                    str,
                )
                or not counts
            ):
                raise SubmissionValidationError(
                    f"{instance_location}.segmentation.counts "
                    "必须是非空 COCO compressed RLE 字符串。"
                )

            # Do not call any native RLE routine before validating the full
            # run stream: area/toBbox may accept an undersized stream, while
            # merge can subsequently hang on it.
            try:
                validate_compressed_rle(counts, expected_height, expected_width)
            except ValueError as exc:
                raise SubmissionValidationError(
                    f"{instance_location}.segmentation 不是有效 compressed RLE：{exc}"
                ) from exc

            rle = {
                "size": [
                    expected_height,
                    expected_width,
                ],
                "counts": counts.encode(
                    "utf-8"
                ),
            }

            try:
                area = float(
                    mask_utils.area(
                        rle
                    )
                )

                bbox = np.asarray(
                    mask_utils.toBbox(
                        rle
                    ),
                    dtype=np.float64,
                ).reshape(
                    -1
                )

            except Exception as exc:
                raise SubmissionValidationError(
                    f"{instance_location}.segmentation "
                    f"不是有效 compressed RLE：{exc}"
                ) from exc

            if (
                not math.isfinite(
                    area
                )
                or area <= 0.0
            ):
                raise SubmissionValidationError(
                    f"{instance_location}.segmentation "
                    "解码后为空掩码。"
                )

            if (
                bbox.size != 4
                or not np.all(
                    np.isfinite(
                        bbox
                    )
                )
            ):
                raise SubmissionValidationError(
                    f"{instance_location}.segmentation "
                    "无法生成有效 bbox。"
                )

            detections.append(
                {
                    "image_id": gt_image_id,
                    "category_id": (
                        SUBMISSION_TO_GT_CATEGORY[
                            category_id
                        ]
                    ),
                    "segmentation": rle,
                    "score": score_float,
                }
            )

            per_category_count[
                category_id
            ] += 1

    missing_images = sorted(
        expected_names
        - submitted_names
    )

    if missing_images:
        preview = ", ".join(
            missing_images[
                :10
            ]
        )

        suffix = (
            "……"
            if len(missing_images) > 10
            else ""
        )

        raise SubmissionValidationError(
            f"提交遗漏 {len(missing_images)} 张测试图像："
            f"{preview}{suffix}"
        )

    if len(
        submitted_names
    ) != len(
        expected_names
    ):
        raise SubmissionValidationError(
            "results 中的图像数量与测试集不一致。"
        )

    summary = {
        "version": EXPECTED_VERSION,
        "num_images": len(
            submitted_names
        ),
        "num_predictions": len(
            detections
        ),
        "predictions_per_category": {
            "0": {
                "name": "完整花穗",
                "count": per_category_count[
                    0
                ],
            },
            "1": {
                "name": "不完整花穗",
                "count": per_category_count[
                    1
                ],
            },
        },
        "max_predictions_in_one_image": max(
            per_image_count.values(),
            default=0,
        ),
    }

    return (
        detections,
        summary,
    )


# ============================================================================
# Ignore-aware COCOeval
# ============================================================================

def create_empty_coco_results(
    coco_gt: COCO,
) -> COCO:
    """
    COCO.loadRes([]) 在部分 pycocotools 版本中行为不稳定，
    空预测显式构造空 COCO results。
    """
    coco_dt = COCO()

    coco_dt.dataset = {
        "images": deepcopy(
            coco_gt.dataset.get(
                "images",
                [],
            )
        ),
        "categories": deepcopy(
            coco_gt.dataset.get(
                "categories",
                [],
            )
        ),
        "annotations": [],
    }

    coco_dt.createIndex()

    return coco_dt


def build_detection_ignore_overlap(
    coco_dt: COCO,
    ignore_unions: Mapping[
        int,
        Dict[str, Any],
    ],
) -> Dict[int, float]:
    """
    计算每个预测与当前图像类别无关 ignore union 的：
        area(pred ∩ ignore) / area(pred)

    全程使用 RLE 运算，不需要展开整幅高分辨率二值 mask。
    """
    overlap_by_dt_id: Dict[
        int,
        float,
    ] = {}

    for (
        dt_id,
        detection,
    ) in coco_dt.anns.items():

        image_id = int(
            detection["image_id"]
        )

        if image_id not in ignore_unions:
            overlap_by_dt_id[
                int(dt_id)
            ] = 0.0
            continue

        pred_rle = detection[
            "segmentation"
        ]

        pred_area = float(
            mask_utils.area(
                pred_rle
            )
        )

        if pred_area <= 0.0:
            overlap_by_dt_id[
                int(dt_id)
            ] = 0.0
            continue

        intersection_rle = mask_utils.merge(
            [
                pred_rle,
                ignore_unions[
                    image_id
                ],
            ],
            intersect=True,
        )

        intersection_area = float(
            mask_utils.area(
                intersection_rle
            )
        )

        overlap = (
            intersection_area
            / pred_area
        )

        # 数值保护
        overlap = min(
            1.0,
            max(
                0.0,
                overlap,
            ),
        )

        overlap_by_dt_id[
            int(dt_id)
        ] = float(
            overlap
        )

    return overlap_by_dt_id


class IgnoreAwareCOCOeval(COCOeval):
    """
    在标准 COCOeval 普通 GT 匹配之后，给“未匹配普通 GT”的预测
    添加类别无关 ignore 决策。

    重要：
    - 不在 COCO 匹配前粗暴删除预测；
    - 因此一个既接近正常 GT、又触碰 ignore 区域的正确预测，
      仍然优先作为 TP；
    - 只有当前 IoU 阈值下未匹配普通 GT 时，ignore 才生效。
    """

    def __init__(
        self,
        coco_gt: COCO,
        coco_dt: COCO,
        *,
        ignore_overlap_by_dt_id: Mapping[
            int,
            float,
        ],
        ignore_threshold: float,
    ):
        super().__init__(
            coco_gt,
            coco_dt,
            iouType="segm",
        )

        self.ignore_overlap_by_dt_id = {
            int(key): float(value)
            for key, value
            in ignore_overlap_by_dt_id.items()
        }

        self.ignore_threshold = float(
            ignore_threshold
        )

    def evaluateImg(
        self,
        imgId,
        catId,
        aRng,
        maxDet,
    ):
        result = super().evaluateImg(
            imgId,
            catId,
            aRng,
            maxDet,
        )

        if result is None:
            return None

        dt_ids = result.get(
            "dtIds",
            [],
        )

        if not dt_ids:
            return result

        dt_matches = np.asarray(
            result["dtMatches"]
        )

        dt_ignore = np.asarray(
            result["dtIgnore"],
            dtype=bool,
        ).copy()

        for (
            detection_index,
            dt_id,
        ) in enumerate(
            dt_ids
        ):
            overlap = (
                self.ignore_overlap_by_dt_id.get(
                    int(dt_id),
                    0.0,
                )
            )

            if (
                overlap
                + 1e-12
                < self.ignore_threshold
            ):
                continue

            # 每个 IoU 阈值独立：
            # 普通 GT 已匹配 => 保持 TP；
            # 普通 GT 未匹配 => 标记为 ignore。
            unmatched = (
                dt_matches[
                    :,
                    detection_index,
                ]
                == 0
            )

            dt_ignore[
                unmatched,
                detection_index,
            ] = True

        result[
            "dtIgnore"
        ] = dt_ignore

        return result


# ============================================================================
# 指标
# ============================================================================

def clean_metric(
    value: Any,
) -> float:
    number = float(
        value
    )

    # 完美/空预测回归测试更稳定，避免 0.9999999999999998。
    if abs(
        number
    ) <= 1e-12:
        return 0.0

    if abs(
        number - 1.0
    ) <= 1e-12:
        return 1.0

    return number


def mean_valid(
    values: np.ndarray,
) -> float:
    valid = values[
        values > -1
    ]

    if valid.size == 0:
        return 0.0

    return clean_metric(
        np.mean(
            valid
        )
    )


def extract_per_category_metrics(
    evaluator: COCOeval,
    gt_category_ids: Sequence[int],
) -> Dict[
    str,
    Dict[str, Any],
]:
    precision = evaluator.eval[
        "precision"
    ]

    recall = evaluator.eval[
        "recall"
    ]

    area_all_index = 0

    # 固定 maxDets [1,10,100]，最后一维索引 2 就是 100。
    max_dets_index = 2

    metrics: Dict[
        str,
        Dict[str, Any],
    ] = {}

    iou_50_index = int(
        np.where(
            evaluator.params.iouThrs
            == 0.50
        )[0][0]
    )

    iou_75_index = int(
        np.where(
            evaluator.params.iouThrs
            == 0.75
        )[0][0]
    )

    for (
        category_index,
        raw_gt_category_id,
    ) in enumerate(
        gt_category_ids
    ):
        # 关键修复：COCOeval 可能把 catIds 转为 np.int64。
        gt_category_id = int(
            raw_gt_category_id
        )

        submission_category_id = (
            GT_TO_SUBMISSION_CATEGORY[
                gt_category_id
            ]
        )

        ap = mean_valid(
            precision[
                :,
                :,
                category_index,
                area_all_index,
                max_dets_index,
            ]
        )

        ap50 = mean_valid(
            precision[
                iou_50_index,
                :,
                category_index,
                area_all_index,
                max_dets_index,
            ]
        )

        ap75 = mean_valid(
            precision[
                iou_75_index,
                :,
                category_index,
                area_all_index,
                max_dets_index,
            ]
        )

        ar = mean_valid(
            recall[
                :,
                category_index,
                area_all_index,
                max_dets_index,
            ]
        )

        metrics[
            str(
                submission_category_id
            )
        ] = {
            "name": SUBMISSION_CATEGORY_NAMES[
                submission_category_id
            ],
            "submission_category_id": int(
                submission_category_id
            ),
            "gt_category_id": int(
                gt_category_id
            ),
            "AP_50_95": ap,
            "AP50": ap50,
            "AP75": ap75,
            "AR_50_95": ar,
        }

    return metrics


def summarize_ignore_application(
    evaluator: IgnoreAwareCOCOeval,
) -> Dict[str, Any]:
    """
    统计 area=all、maxDets=100 下，每个 IoU 阈值实际因为
    类别无关 ignore 规则而被忽略的“未匹配预测”数量。
    """
    counts = np.zeros(
        len(
            evaluator.params.iouThrs
        ),
        dtype=np.int64,
    )

    all_area = evaluator.params.areaRng[
        0
    ]

    for result in evaluator.evalImgs:
        if result is None:
            continue

        if list(
            result["aRng"]
        ) != list(
            all_area
        ):
            continue

        dt_ids = result.get(
            "dtIds",
            [],
        )

        if not dt_ids:
            continue

        dt_matches = np.asarray(
            result["dtMatches"]
        )

        for (
            detection_index,
            dt_id,
        ) in enumerate(
            dt_ids
        ):
            overlap = (
                evaluator.ignore_overlap_by_dt_id.get(
                    int(dt_id),
                    0.0,
                )
            )

            if (
                overlap
                + 1e-12
                < evaluator.ignore_threshold
            ):
                continue

            unmatched = (
                dt_matches[
                    :,
                    detection_index,
                ]
                == 0
            )

            counts += unmatched.astype(
                np.int64
            )

    return {
        "ignored_unmatched_predictions_by_iou": {
            f"{float(iou):.2f}": int(
                count
            )
            for (
                iou,
                count,
            ) in zip(
                evaluator.params.iouThrs,
                counts,
            )
        }
    }


def evaluate_coco(
    gt_json_path: Path,
    detections: List[
        Dict[str, Any]
    ],
    ignore_unions: Mapping[
        int,
        Dict[str, Any],
    ],
) -> Tuple[
    Dict[str, Any],
    str,
]:
    coco_gt = COCO(
        str(
            gt_json_path
        )
    )

    if detections:
        coco_dt = coco_gt.loadRes(
            detections
        )
    else:
        coco_dt = create_empty_coco_results(
            coco_gt
        )

    ignore_overlap_by_dt_id = (
        build_detection_ignore_overlap(
            coco_dt,
            ignore_unions,
        )
    )

    evaluator = IgnoreAwareCOCOeval(
        coco_gt,
        coco_dt,
        ignore_overlap_by_dt_id=(
            ignore_overlap_by_dt_id
        ),
        ignore_threshold=(
            IGNORE_OVERLAP_THRESHOLD
        ),
    )

    evaluator.params.imgIds = sorted(
        int(value)
        for value
        in coco_gt.getImgIds()
    )

    evaluator.params.catIds = [
        1,
        2,
    ]

    evaluator.params.iouThrs = (
        IOU_THRESHOLDS.copy()
    )

    evaluator.params.recThrs = (
        RECALL_THRESHOLDS.copy()
    )

    # 固定，不允许外部改变。
    evaluator.params.maxDets = (
        MAX_DETS.copy()
    )

    evaluator.evaluate()
    evaluator.accumulate()

    buffer = io.StringIO()

    with contextlib.redirect_stdout(
        buffer
    ):
        evaluator.summarize()

    summary_text = buffer.getvalue()

    print(
        summary_text,
        end="",
    )

    stats = evaluator.stats

    primary_score = clean_metric(
        stats[0]
    )

    if not (
        0.0
        <= primary_score
        <= 1.0
    ):
        raise RuntimeError(
            f"主 Mask mAP 得到非法值 {primary_score}。"
            "请检查 GT、pycocotools 版本和固定 maxDets 配置。"
        )

    overall = {
        "Mask_mAP_50_95": primary_score,
        "Mask_AP50": clean_metric(
            stats[1]
        ),
        "Mask_AP75": clean_metric(
            stats[2]
        ),
        "Mask_AP_small": clean_metric(
            stats[3]
        ),
        "Mask_AP_medium": clean_metric(
            stats[4]
        ),
        "Mask_AP_large": clean_metric(
            stats[5]
        ),
        "Mask_AR_maxDets_1": clean_metric(
            stats[6]
        ),
        "Mask_AR_maxDets_10": clean_metric(
            stats[7]
        ),
        "Mask_AR_maxDets_100": clean_metric(
            stats[8]
        ),
        "Mask_AR_small": clean_metric(
            stats[9]
        ),
        "Mask_AR_medium": clean_metric(
            stats[10]
        ),
        "Mask_AR_large": clean_metric(
            stats[11]
        ),
    }

    per_category = (
        extract_per_category_metrics(
            evaluator,
            [
                int(value)
                for value
                in evaluator.params.catIds
            ],
        )
    )

    ignore_application = (
        summarize_ignore_application(
            evaluator
        )
    )

    evaluation = {
        "metric": "COCO Mask mAP@[0.50:0.95]",
        "score": primary_score,
        "score_percent": (
            primary_score
            * 100.0
        ),
        "iou_thresholds": [
            float(value)
            for value
            in evaluator.params.iouThrs
        ],
        "recall_points": int(
            len(
                evaluator.params.recThrs
            )
        ),
        "max_dets": [
            1,
            10,
            100,
        ],
        "overall": overall,
        "per_category": per_category,
        "ignore_application": (
            ignore_application
        ),
    }

    return (
        evaluation,
        summary_text,
    )


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "严格校验选手 result.json，并使用类别无关 ignore_regions "
            "计算 COCO Mask mAP。默认路径均相对于评分脚本所在目录。"
        )
    )

    parser.add_argument(
        "--gt_json",
        type=Path,
        default=DEFAULT_GT_JSON,
        help=(
            "测试集普通 COCO 真值。默认："
            "ground_truth/instances_test.json"
        ),
    )

    parser.add_argument(
        "--ignore_json",
        type=Path,
        default=DEFAULT_IGNORE_JSON,
        help=(
            "类别无关忽略区域。默认："
            "ground_truth/ignore_regions_test.json"
        ),
    )

    parser.add_argument(
        "--submission_json",
        type=Path,
        default=DEFAULT_SUBMISSION_JSON,
        help=(
            "选手提交结果。默认："
            "submission/result.json"
        ),
    )

    parser.add_argument(
        "--output_json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help=(
            "评分成功输出。默认："
            "output/score.json"
        ),
    )

    parser.add_argument(
        "--error_json",
        type=Path,
        default=DEFAULT_ERROR_JSON,
        help=(
            "评分失败输出。默认："
            "output/error.json"
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    gt_json_path = resolve_portable_path(
        args.gt_json
    )

    ignore_json_path = resolve_portable_path(
        args.ignore_json
    )

    submission_json_path = resolve_portable_path(
        args.submission_json
    )

    output_json_path = resolve_portable_path(
        args.output_json
    )

    error_json_path = resolve_portable_path(
        args.error_json
    )

    if (
        output_json_path.resolve()
        == error_json_path.resolve()
    ):
        print(
            "错误：--output_json 与 --error_json 不能指向同一文件。",
            file=sys.stderr,
        )
        return 2

    # ------------------------------------------------------------------
    # 每次评测开始前清理旧状态，避免平台误读上一轮成绩。
    # ------------------------------------------------------------------
    safe_unlink(
        output_json_path
    )

    safe_unlink(
        error_json_path
    )

    try:
        gt_data = load_json_strict(
            gt_json_path
        )

        (
            images_by_name,
            images_by_id,
        ) = validate_gt_dataset(
            gt_data
        )

        ignore_data = load_json_strict(
            ignore_json_path
        )

        (
            ignore_unions,
            ignore_summary,
        ) = validate_and_build_ignore_unions(
            ignore_data,
            images_by_id,
        )

        submission_data = load_json_strict(
            submission_json_path,
            submission=True,
        )

        (
            detections,
            validation,
        ) = validate_and_convert_submission(
            submission_data,
            images_by_name,
        )

        print(
            "提交文件校验通过。"
        )

        print(
            f"测试图像数量：{validation['num_images']}"
        )

        print(
            f"预测实例数量：{validation['num_predictions']}"
        )

        print(
            "单张图像最大预测实例数："
            f"{validation['max_predictions_in_one_image']}"
        )

        print(
            "Ignore regions："
            f"{ignore_summary['num_ignore_regions']} 个，"
            f"覆盖 {ignore_summary['num_images_with_ignore']} 张图"
        )

        print(
            "\n开始执行 Ignore-aware COCO Mask 评测……"
        )

        (
            evaluation,
            coco_summary,
        ) = evaluate_coco(
            gt_json_path,
            detections,
            ignore_unions,
        )

        output = {
            "status": "success",
            # 只记录调用时提供的路径，不写入脚本解析后的本机绝对目录。
            "gt_json": args.gt_json.as_posix(),
            "ignore_json": args.ignore_json.as_posix(),
            "submission_json": args.submission_json.as_posix(),
            "validation": validation,
            "ignore_policy": ignore_summary,
            "evaluation": evaluation,
            "coco_summary": coco_summary,
        }

        atomic_write_json(
            output_json_path,
            output,
        )

        # 成功状态下确保不存在旧 error.json。
        safe_unlink(
            error_json_path
        )

        print(
            "=" * 72
        )

        print(
            "最终比赛得分 Mask mAP@[0.50:0.95]："
            f"{evaluation['score']:.6f}"
        )

        print(
            f"百分制显示：{evaluation['score_percent']:.4f}"
        )

        print(
            f"评分结果已保存：{args.output_json.as_posix()}"
        )

        return 0

    except (
        SubmissionValidationError,
        GroundTruthValidationError,
        FileNotFoundError,
        ValueError,
        KeyError,
    ) as exc:

        # 再次确保失败时绝对没有旧 score.json。
        safe_unlink(
            output_json_path
        )

        error = {
            "status": "error",
            "error_type": type(
                exc
            ).__name__,
            "message": str(
                exc
            ),
        }

        print(
            f"评分失败：{exc}",
            file=sys.stderr,
        )

        try:
            atomic_write_json(
                error_json_path,
                error,
            )
        except Exception as write_exc:
            print(
                f"错误 JSON 写入失败：{write_exc}",
                file=sys.stderr,
            )

        return 2

    except Exception as exc:
        safe_unlink(
            output_json_path
        )

        error = {
            "status": "error",
            "error_type": type(
                exc
            ).__name__,
            "message": str(
                exc
            ),
            "traceback": traceback.format_exc(),
        }

        print(
            "评分程序发生未预期错误。",
            file=sys.stderr,
        )

        traceback.print_exc()

        try:
            atomic_write_json(
                error_json_path,
                error,
            )
        except Exception:
            pass

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
