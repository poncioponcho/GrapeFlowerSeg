#!/usr/bin/env python3
"""Generate solution_commit.txt from the original solution.zip bytes."""

import argparse
import hashlib
import zipfile
from pathlib import Path

if __package__:
    from .submission import B_DATA_VERSION, COMMIT_FIELDS, parse_solution_commit
else:
    from submission import B_DATA_VERSION, COMMIT_FIELDS, parse_solution_commit


def generate_commit(solution_path, output_path):
    solution_path, output_path = Path(solution_path), Path(output_path)
    if solution_path.name != "solution.zip" or not solution_path.is_file():
        raise ValueError("请提供已完成的原始 solution.zip 文件")
    if output_path.resolve() == solution_path.resolve():
        raise ValueError("输出路径不能覆盖 solution.zip")
    if not zipfile.is_zipfile(solution_path):
        raise ValueError("solution.zip 不是有效的 ZIP 压缩包")
    digest, size = hashlib.sha256(), 0
    with solution_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    fields = dict(zip(COMMIT_FIELDS, (
        "solution.zip", "SHA-256", digest.hexdigest(), str(size), B_DATA_VERSION,
    )))
    content = "".join("{}={}\n".format(key, fields[key]) for key in COMMIT_FIELDS)
    parse_solution_commit(content.encode("utf-8"))
    output_path.write_text(content, encoding="utf-8")
    return fields


def main():
    parser = argparse.ArgumentParser(description="对原始 solution.zip 计算真实 SHA-256 和字节数")
    parser.add_argument("solution", help="本次完整推理包 solution.zip")
    parser.add_argument("--output", default="solution_commit.txt")
    args = parser.parse_args()
    fields = generate_commit(args.solution, args.output)
    print("已生成 {}：SHA-256={}，字节数={}".format(
        args.output, fields["solution_sha256"], fields["solution_size"]
    ))


if __name__ == "__main__":
    main()
