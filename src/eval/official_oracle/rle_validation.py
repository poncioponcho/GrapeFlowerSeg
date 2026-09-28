"""Validate COCO's compressed run stream before calling its native decoder.

pycocotools' area/toBbox/decode do not consistently reject truncated masks.
In particular, a stream whose runs do not cover the declared image may hang
native RLE merging. Keep this check shared by the scorer and public validator.
"""

from __future__ import annotations


def validate_compressed_rle(counts: str, height: int, width: int) -> None:
    """Reject invalid encoding, negative/overflowed runs and wrong pixel totals.

COCO stores signed 5-bit chunks (ASCII 48..111); runs after the third are
differences from the run two positions earlier. Decode in Python, without
allocating a mask or handing any unvalidated bytes to pycocotools.
"""
    if not isinstance(counts, str) or not counts:
        raise ValueError("counts 必须是非空 COCO compressed RLE 字符串")
    if type(height) is not int or type(width) is not int or min(height, width) <= 0:
        raise ValueError("RLE size 必须是两个正整数")

    pixel_count = height * width
    max_run = (1 << 32) - 1  # pycocotools stores each expanded run as uint32.
    if pixel_count > max_run:
        raise ValueError("RLE 图像像素数超过支持范围")

    position = 0
    run_index = 0
    two_back = one_back = 0
    total = 0
    while position < len(counts):
        value = 0
        shift = 0
        while True:
            if position >= len(counts):
                raise ValueError("RLE counts 编码被截断：末尾缺少结束字符")
            chunk = ord(counts[position]) - 48
            position += 1
            if not 0 <= chunk <= 63:
                raise ValueError("RLE counts 包含非法字符（须为 ASCII 48..111）")
            # Seven chunks cover every signed difference of uint32 runs.
            # A larger shift is unnecessary and is unsafe in the C decoder.
            if shift >= 35:
                raise ValueError("RLE counts 单个 run 编码过长")
            value |= (chunk & 0x1F) << shift
            shift += 5
            if not chunk & 0x20:
                if chunk & 0x10:
                    value -= 1 << shift
                break

        if run_index > 2:
            value += two_back
        if value < 0 or value > max_run:
            raise ValueError("RLE counts 包含负数或溢出的 run")
        total += value
        if total > pixel_count:
            raise ValueError(f"RLE run 总长度超过图像像素数 {pixel_count}")
        two_back, one_back = one_back, value
        run_index += 1

    if total != pixel_count:
        raise ValueError(
            f"RLE run 总长度必须等于 height×width={pixel_count}，实际为 {total}"
        )
