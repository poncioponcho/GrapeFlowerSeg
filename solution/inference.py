#!/usr/bin/env python3
"""Unified inference entry point (official contract).

    python inference.py --input_dir test_images/ --weights model/fold0.pt ... \
        --output_path reproduced_result.json

Reads every image under ``--input_dir``, predicts grape-panicle instances, and
writes a ``result.json`` in the submission format:

* ``category_id`` 0 = 完整花穗 (complete panicle), 1 = 不完整花穗 (incomplete)
* ``segmentation`` = COCO compressed RLE ``{"size": [h, w], "counts": str}``

Every image gets exactly one record; images with no detection get
``"instances": []``.

The inference resolution defaults to the set the submitted ``result.json`` was
produced with (see ``model.model.DEFAULT_INFER_SHORT_SIDES`` - a 5-fold x 2-scale
ensemble); pass ``--short_side`` to override it. Reproducing the submitted score
requires the default, so do not change it without re-running the submission.

This file ships inside ``solution.zip`` and must reproduce the submitted
predictions offline. Keep it self-contained: no network access, no imports from
outside the archive.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def load_metadata(manifest_path: Path | None) -> dict[str, tuple[int, int]]:
    """Optional public manifest giving per-image width/height."""
    if manifest_path is None:
        return {}
    rows = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return {row["image_id"]: (int(row["width"]), int(row["height"])) for row in rows}


def build_predictor(weights, device: str, short_side: int | None = None):
    """Return a callable ``(image_path, width, height) -> list[instance]``.

    Falls back to an empty predictor only when no weights are supplied, which is
    the "no model yet" state: the output is format-valid and scores 0. Weights
    that were given but are missing are a hard error - silently emitting empty
    predictions would fake a reproduction.
    """
    if weights is None:
        def empty_predictor(_path, _width, _height):
            return []
        return empty_predictor

    paths = list(weights) if isinstance(weights, (list, tuple)) else [weights]
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise SystemExit(f"weights not found: {missing}")

    try:
        import torch
        from model.model import (  # packaged alongside this file
            DEFAULT_INFER_SHORT_SIDES,
            PanicleSegmenter,
        )
    except ImportError as exc:
        raise SystemExit(
            f"--weights given but the model implementation is unavailable: {exc}"
        ) from exc

    if short_side is None:
        resolved = [int(v) for v in DEFAULT_INFER_SHORT_SIDES]
    elif isinstance(short_side, (list, tuple)):
        resolved = [int(v) for v in short_side]
    else:
        resolved = [int(p) for p in str(short_side).split(",") if p.strip()]
    if not resolved:
        raise SystemExit("--short_side must select at least one resolution")

    model = PanicleSegmenter.load(paths, device=device, short_side=resolved)
    model.eval()
    print(f"inference short sides = {resolved}", file=sys.stderr)

    def predictor(image_path, width, height):
        return model.predict(image_path, width, height)

    return predictor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, type=Path,
                        help="directory of test images")
    parser.add_argument("--weights", nargs="+", default=None, type=Path,
                        help="model weights, one or more (multiple checkpoints "
                             "are ensembled by per-class NMS; omit to emit "
                             "empty predictions)")
    parser.add_argument("--output_path", required=True, type=Path)
    parser.add_argument("--manifest", default=None, type=Path,
                        help="optional images_test_b.json for exact dimensions")
    parser.add_argument("--score_threshold", type=float, default=0.05)
    parser.add_argument("--short_side", default=None,
                        help="inference resolution(s), e.g. 1024 or 640,1024; "
                             "defaults to the resolutions the submitted "
                             "result.json was produced with")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"input_dir not found: {args.input_dir}")

    metadata = load_metadata(args.manifest)
    images = sorted(
        path for path in args.input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise SystemExit(f"no images found under {args.input_dir}")

    predict = build_predictor(args.weights, args.device, args.short_side)

    results = []
    total_instances = 0
    for index, image_path in enumerate(images, 1):
        name = image_path.name
        width, height = metadata.get(name, (0, 0))
        if not width or not height:
            from PIL import Image
            with Image.open(image_path) as handle:
                width, height = handle.size
        instances = predict(image_path, width, height)
        instances = [item for item in instances if item["score"] >= args.score_threshold]
        instances.sort(key=lambda item: item["score"], reverse=True)
        total_instances += len(instances)
        results.append({"image_id": name, "instances": instances})
        if index % 20 == 0 or index == len(images):
            print(f"  {index}/{len(images)} images", file=sys.stderr)

    payload = {"version": "1.0", "results": results}
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    print(f"wrote {args.output_path} ({len(results)} images, {total_instances} instances)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
