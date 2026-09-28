# -*- coding: utf-8 -*-

"""
result.json 提交格式检查工具

用途：
    选手提交前本地检查 result.json

特点：
    不读取测试真实标签
    不计算比赛得分

检查：
    1. JSON结构
    2. image_id完整性
    3. category_id
    4. score
    5. COCO RLE
    6. mask尺寸
"""


import json
import argparse
import math
from pathlib import Path

from PIL import Image
from pycocotools import mask as mask_utils



# ==========================
# 参数
# ==========================

parser = argparse.ArgumentParser()


parser.add_argument(
    "--submission",
    required=True,
    help="result.json路径"
)


parser.add_argument(
    "--image_dir",
    required=True,
    help="测试图片目录"
)

parser.add_argument(
    "--example",
    action="store_true",
    help="示例模式：仅验证提交中包含的图片"
)


args = parser.parse_args()



submission_file = Path(args.submission)

image_dir = Path(args.image_dir)



# ==========================
# 工具函数
# ==========================

def error(msg):

    raise RuntimeError(
        "\nERROR:\n" + msg
    )



# ==========================
# 读取submission
# ==========================


try:

    with open(
        submission_file,
        "r",
        encoding="utf-8"
    ) as f:

        submission=json.load(f)


except Exception as e:

    error(
        f"JSON解析失败: {e}"
    )



# ==========================
# 基础结构
# ==========================


if set(submission.keys()) - {"version", "results"}:

    error(
        f"包含未允许顶层字段: {sorted(set(submission.keys()) - {'version', 'results'})}"
    )


if "version" not in submission:

    error(
        "缺少 version 字段"
    )


if submission["version"] != "1.0":

    error(
        f"version必须为'1.0'，实际为 {submission['version']!r}"
    )


if "results" not in submission:

    error(
        "缺少 results 字段"
    )



results=submission["results"]


if not isinstance(results,list):

    error(
        "results必须为list"
    )



# ==========================
# 图片列表
# ==========================


image_files=sorted(

    [
        x.name
        for x in image_dir.iterdir()
        if x.suffix.lower()
        in [
            ".jpg",
            ".jpeg",
            ".png"
        ]
    ]

)



submit_ids=[]


total_instances=0



# ==========================
# 检查results
# ==========================


for item in results:


    allowed_result_keys = {
        "image_id",
        "instances"
    }

    extra_result_keys = set(item.keys()) - allowed_result_keys

    if extra_result_keys:
        error(
            f"result项包含未允许字段: {sorted(extra_result_keys)}"
        )


    if "image_id" not in item:

        error(
            "存在结果缺少image_id"
        )


    image_id=item["image_id"]


    submit_ids.append(
        image_id
    )


    if image_id not in image_files:

        error(
            f"提交了不存在图片: {image_id}"
        )



    if "instances" not in item:

        error(
            f"{image_id}缺少instances"
        )


    instances=item["instances"]


    if not isinstance(
        instances,
        list
    ):

        error(
            f"{image_id}.instances必须为list"
        )



    # 图片尺寸

    img_path=image_dir/image_id


    with Image.open(img_path) as img:

        width,height=img.size



    for idx,ins in enumerate(instances):


        total_instances+=1


        allowed_instance_keys = {
            "category_id",
            "score",
            "segmentation"
        }

        extra_keys = set(ins.keys()) - allowed_instance_keys

        if extra_keys:
            error(
                f"{image_id} instance {idx}包含未允许字段: {sorted(extra_keys)}"
            )


        # -------------------
        # category
        # -------------------

        if "category_id" not in ins:

            error(
                f"{image_id} instance {idx}缺少category_id"
            )


        cid=ins["category_id"]


        if type(cid) is not int or cid not in [0,1]:

            error(
                f"{image_id} instance {idx}: "
                f"category_id必须为整数0/1，实际={cid!r}"
            )



        # -------------------
        # score
        # -------------------

        if "score" not in ins:

            error(
                f"{image_id} instance {idx}缺少score"
            )


        score=ins["score"]


        if type(score) not in [int, float]:

            error(
                f"{image_id} score必须为数字，实际={score!r}"
            )


        if (
            not math.isfinite(score)
            or score<0
            or score>1
        ):

            error(
                f"{image_id} score范围错误:{score}"
            )



        # -------------------
        # segmentation
        # -------------------

        if "segmentation" not in ins:

            error(
                f"{image_id}缺少segmentation"
            )


        rle=ins["segmentation"]


        if not isinstance(
            rle,
            dict
        ):

            error(
                f"{image_id} segmentation必须为RLE对象"
            )


        allowed_segmentation_keys = {
            "size",
            "counts"
        }

        extra_segmentation_keys = (
            set(rle.keys())
            -
            allowed_segmentation_keys
        )

        if extra_segmentation_keys:
            error(
                f"{image_id} segmentation包含未允许字段: "
                f"{sorted(extra_segmentation_keys)}"
            )


        if "size" not in rle:

            error(
                f"{image_id} RLE缺少size"
            )


        if "counts" not in rle:

            error(
                f"{image_id} RLE缺少counts"
            )



        size=rle["size"]


        if size != [height,width]:

            error(
                f"{image_id} mask尺寸错误:"
                f"提交{size}, 图片{[height,width]}"
            )



        counts=rle["counts"]


        if not isinstance(
            counts,
            str
        ):

            error(
                f"{image_id} counts必须为字符串"
            )



        # -------------------
        # RLE decode
        # -------------------

        try:

            mask=mask_utils.decode(
                rle
            )


            if mask.shape != (
                height,
                width
            ):

                error(
                    f"{image_id} decode尺寸错误"
                )


            # 解码后必须为非空掩码。
            # 与正式评分器保持一致：面积为0的RLE直接拒绝。
            if not mask.any():

                error(
                    f"{image_id} instance {idx}: "
                    f"segmentation 解码后为空掩码"
                )


        except Exception as e:

            error(
                f"{image_id} RLE无法解码:{e}"
            )



# ==========================
# 图片完整性
# ==========================


if len(set(submit_ids)) != len(submit_ids):

    error(
        "存在重复image_id"
    )


if not args.example:

    missing=set(image_files)-set(submit_ids)

    extra=set(submit_ids)-set(image_files)


    if missing:

        error(
            "缺少图片:\n"
            + "\n".join(sorted(missing))
        )


    if extra:

        error(
            "多余图片:\n"
            + "\n".join(sorted(extra))
        )



# ==========================
# 成功
# ==========================


print("="*60)

print(
    "Submission validation passed"
)

print(
    "Images checked:",
    len(submit_ids)
)

print(
    "Instances checked:",
    total_instances
)

print(
    "RLE decode: OK"
)

print("="*60)