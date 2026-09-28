#!/usr/bin/env python3
"""Validate a B submission using public image metadata and the platform parser."""

import argparse
import json
import tempfile
from pathlib import Path

if __package__:
    from .official_evaluate import load_json_strict, validate_and_convert_submission
    from .submission import read_submission
else:
    from official_evaluate import load_json_strict, validate_and_convert_submission
    from submission import read_submission


def load_image_metadata(metadata_path):
    rows = load_json_strict(Path(metadata_path), submission=True)
    if not isinstance(rows, list) or len(rows) != 106:
        raise ValueError("请使用官方 images_test_b.json：应包含 B 榜全部 106 张图片")
    images = {}
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or set(row) != {"image_id", "width", "height"}:
            raise ValueError("图像信息条目必须包含 image_id、width、height")
        name = row["image_id"]
        if not isinstance(name, str) or not name or name in images:
            raise ValueError("图像信息包含空、重复或非法 image_id")
        if any(type(row[key]) is not int or row[key] <= 0 for key in ("width", "height")):
            raise ValueError("图像尺寸必须是正整数")
        images[name] = {"id": index, "file_name": name, "width": row["width"], "height": row["height"]}
    return images


def validate_submission(zip_path, metadata_path):
    result_bytes, commit = read_submission(zip_path)
    images = load_image_metadata(metadata_path)
    with tempfile.TemporaryDirectory(prefix="b_submission_check_") as temporary:
        result_path = Path(temporary) / "result.json"
        result_path.write_bytes(result_bytes)
        result = load_json_strict(result_path, submission=True)
        _, summary = validate_and_convert_submission(result, images)
    return {"solution_commit": commit, "validation": summary}


def main():
    parser = argparse.ArgumentParser(description="B 榜提交包本地格式检查（不读取真实标注，不评分）")
    parser.add_argument("submission", help="b_submission.zip")
    parser.add_argument("image_metadata", help="官方 images_test_b.json")
    args = parser.parse_args()
    try:
        details = validate_submission(args.submission, args.image_metadata)
    except Exception as exc:
        parser.exit(1, "校验失败：{}\n".format(exc))
    print("格式校验通过；真实模型 SHA、字节数及结果复现将在赛后审核。")
    print(json.dumps(details, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
