#!/usr/bin/env python3
"""Fixed B-board archive and model-declaration contract (standard library only)."""

import re
import stat
import zipfile
from pathlib import Path

B_DATA_VERSION = "B-v1"
MAX_ARCHIVE_BYTES = 1 << 30
MAX_RESULT_BYTES = 1 << 30
MAX_COMMIT_BYTES = 65_536
COMMIT_FIELDS = (
    "solution_name", "hash_algorithm", "solution_sha256",
    "solution_size", "b_data_version",
)


def parse_solution_commit(raw):
    """Check syntax now; verify the declared original model bytes after ranking."""
    if len(raw) > MAX_COMMIT_BYTES:
        raise ValueError("solution_commit.txt 超过 65,536 字节安全上限")
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("solution_commit.txt 必须使用 UTF-8 编码") from exc
    fields = {}
    for line_no, line in enumerate(content.splitlines(), 1):
        if "=" not in line:
            raise ValueError("solution_commit.txt 第 {} 行不是 key=value 格式".format(line_no))
        key, value = line.split("=", 1)
        if key not in COMMIT_FIELDS or key in fields or not value:
            raise ValueError("solution_commit.txt 包含未知、重复或空字段")
        fields[key] = value
    if set(fields) != set(COMMIT_FIELDS):
        raise ValueError("solution_commit.txt 缺少字段：{}".format(", ".join(sorted(set(COMMIT_FIELDS) - set(fields)))))
    if fields["solution_name"] != "solution.zip":
        raise ValueError("solution_name 必须为 solution.zip")
    if fields["hash_algorithm"] != "SHA-256":
        raise ValueError("hash_algorithm 必须为 SHA-256")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", fields["solution_sha256"]):
        raise ValueError("solution_sha256 必须是 64 位十六进制 SHA-256")
    if not re.fullmatch(r"[1-9][0-9]*", fields["solution_size"]):
        raise ValueError("solution_size 必须是无前导零的正整数字节数")
    if fields["b_data_version"] != B_DATA_VERSION:
        raise ValueError("b_data_version 必须为 {}".format(B_DATA_VERSION))
    fields["solution_sha256"] = fields["solution_sha256"].lower()
    return fields


def read_submission(zip_path):
    """Read exact whitelisted members without extracting any untrusted paths."""
    zip_path = Path(zip_path)
    if zip_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("B 榜提交 ZIP 超过 1 GiB 安全上限")
    if not zipfile.is_zipfile(zip_path):
        raise ValueError("B 榜只接受 ZIP 提交包，不接受裸 result.json")
    limits = {"result.json": MAX_RESULT_BYTES, "solution_commit.txt": MAX_COMMIT_BYTES}
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.infolist()
        if len(members) != 2 or {member.filename for member in members} != set(limits):
            raise ValueError("提交包根目录必须且只能包含 result.json 和 solution_commit.txt 两个文件")
        for member in members:
            if member.orig_filename != member.filename:
                raise ValueError("ZIP 包含非法文件名")
            mode = (member.external_attr >> 16) & 0xFFFF
            if member.is_dir() or stat.S_IFMT(mode) not in (0, stat.S_IFREG):
                raise ValueError("提交包只允许普通文件，不允许目录或符号链接")
            if member.flag_bits & 1:
                raise ValueError("提交包不允许加密")
            if member.file_size == 0 or member.file_size > limits[member.filename]:
                raise ValueError("{} 为空或解压后超过安全上限".format(member.filename))
        # read() validates CRC; the small declaration is checked before the large JSON.
        commit = parse_solution_commit(archive.read("solution_commit.txt"))
        result = archive.read("result.json")
        if len(result) > MAX_RESULT_BYTES:
            raise ValueError("result.json 解压后超过 1 GiB 安全上限")
    return result, commit
