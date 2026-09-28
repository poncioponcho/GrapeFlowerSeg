"""Generate a synthetic dataset that mirrors the competition's *shape*.

⚠️ This exists ONLY so the engineering skeleton can be tested and the
``generate -> verify -> pack`` flow can be exercised before the real data
lands. Numbers computed on this data are **not** competition results and must
never be reported as such.

What it reproduces faithfully (because these are what the code must handle):

* many berry crops per source image (果穗图), so grouped splitting matters;
* labels correlated *within* a source image plus per-source colour cast, which
  is exactly why a random per-image split leaks and inflates the score;
* an ordinal colour axis, so adjacent classes (0<->1, 1<->2) are the confusable
  pairs;
* the official split sizes from ``configs/default.yaml``.

Usage
-----
>>> python -m data.make_sample_data --out-dir data/sample
>>> python -m data.make_sample_data --out-dir data/sample --with-images
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from common.config import load_config
from common.io_utils import ensure_dir, write_csv_rows

LABELS_CSV_FIELDS = ("image_id", "file_name", "label")
SUBMISSION_FIELDS = ("image_id", "label")


def _berry_count(rng: random.Random) -> int:
    """Crops per source image: right-skewed, mean tuned to the configured ratio.

    The mean is derived from ``configs/default.yaml`` (train images / train
    sources) rather than hardcoded, so the synthetic shape tracks the real one.
    """
    return max(1, int(rng.gauss(_MEAN_CROPS_PER_SOURCE, 5.0)))


_MEAN_CROPS_PER_SOURCE = 13.6


def generate(
    out_dir: Path,
    *,
    n_sources: int,
    n_test_a: int,
    n_test_b: int,
    seed: int,
    with_images: bool,
    image_size: int,
) -> dict:
    rng = random.Random(seed)
    raw_dir = ensure_dir(out_dir / "raw")
    image_dir = ensure_dir(raw_dir / "images")

    # Each source image sits at a position on the ordinal colour axis, so crops
    # from one source mostly share a class - the correlation that grouping must
    # neutralise.
    source_rows: list[tuple[str, float]] = []
    for index in range(n_sources):
        source_id = f"src{index:04d}"
        centre = rng.uniform(0.0, 2.0)
        source_rows.append((source_id, centre))

    rows: list[dict[str, object]] = []
    counter = 0
    for source_id, centre in source_rows:
        for berry_index in range(_berry_count(rng)):
            # Ordinal jitter around the source's colour state; per-source cast
            # adds a shared offset so a source's crops are mutually similar.
            cast = rng.gauss(0.0, 0.28)
            value = centre + cast + rng.gauss(0.0, 0.55)
            label = int(min(2, max(0, round(value))))
            file_name = f"{source_id}_{berry_index:03d}.jpg"
            rows.append(
                {
                    "image_id": f"{counter:06d}",
                    "file_name": file_name,
                    "label": label,
                }
            )
            counter += 1

    labels_csv = write_csv_rows(raw_dir / "labels.csv", rows, LABELS_CSV_FIELDS)

    # Held-out ids for A/B, disjoint from training ids by construction.
    test_a_ids = [f"a{counter + i:06d}" for i in range(n_test_a)]
    test_b_ids = [f"b{counter + n_test_a + i:06d}" for i in range(n_test_b)]
    test_a_dir = ensure_dir(out_dir / "testA")
    test_b_dir = ensure_dir(out_dir / "testB")
    # sample_submission.csv ships placeholder labels; only the ids matter here.
    write_csv_rows(
        test_a_dir / "sample_submission.csv",
        ({"image_id": i, "label": 0} for i in test_a_ids),
        SUBMISSION_FIELDS,
    )
    write_csv_rows(
        test_b_dir / "sample_submission.csv",
        ({"image_id": i, "label": 0} for i in test_b_ids),
        SUBMISSION_FIELDS,
    )
    (test_a_dir / "test_ids.txt").write_text("\n".join(test_a_ids) + "\n", encoding="utf-8")
    (test_b_dir / "test_ids.txt").write_text("\n".join(test_b_ids) + "\n", encoding="utf-8")

    if with_images:
        _write_placeholder_images(image_dir, rows, image_size)

    return {
        "labels_csv": str(labels_csv),
        "n_train_images": len(rows),
        "n_sources": n_sources,
        "n_test_a": n_test_a,
        "n_test_b": n_test_b,
        "sample_submission_a": str(test_a_dir / "sample_submission.csv"),
        "sample_submission_b": str(test_b_dir / "sample_submission.csv"),
        "images_written": with_images,
    }


def _write_placeholder_images(
    image_dir: Path, rows: list[dict[str, object]], image_size: int
) -> None:
    """Tiny synthetic crops whose mean colour encodes the label (smoke tests)."""
    from PIL import Image  # imported lazily so the generator works without Pillow

    palette = {0: (150, 200, 90), 1: (215, 140, 120), 2: (140, 35, 55)}
    for row in rows:
        base = palette[int(row["label"])]
        jittered = tuple(
            max(0, min(255, channel + ((int(row["image_id"]) * 37 + index * 11) % 41) - 20))
            for index, channel in enumerate(base)
        )
        image = Image.new("RGB", (image_size, image_size), jittered)
        image.save(image_dir / str(row["file_name"]), quality=70)


def simulate_predictions(
    labels_csv: Path,
    out_csv: Path,
    *,
    accuracy: float,
    adjacent_error_share: float,
    seed: int,
) -> Path:
    """Simulate model output so the submit pipeline can be exercised offline.

    Errors are drawn to be *ordinal*: with probability
    ``adjacent_error_share`` a mistake moves to the neighbouring class rather
    than jumping 0->2, mirroring the real confusion structure.
    """
    import csv as _csv

    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    with open(labels_csv, newline="", encoding="utf-8") as handle:
        for record in _csv.DictReader(handle):
            truth = int(record["label"])
            if rng.random() < accuracy:
                predicted = truth
            else:
                if rng.random() < adjacent_error_share:
                    step = -1 if (truth == 2 or (truth == 1 and rng.random() < 0.5)) else 1
                else:
                    step = rng.choice([-2, -1, 1, 2])
                predicted = int(min(2, max(0, truth + step)))
            rows.append({"image_id": record["image_id"], "label": predicted})
    return write_csv_rows(out_csv, rows, SUBMISSION_FIELDS)


def main() -> None:  # pragma: no cover - CLI glue
    import json

    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/sample"))
    # Defaults come from configs/default.yaml (single source of truth).
    parser.add_argument("--n-sources", type=int, default=int(cfg.competition.n_train_sources))
    parser.add_argument("--n-test-a", type=int, default=int(cfg.competition.n_test_a))
    parser.add_argument("--n-test-b", type=int, default=int(cfg.competition.n_test_b))
    parser.add_argument("--seed", type=int, default=int(cfg.project.seed))
    parser.add_argument("--with-images", action="store_true")
    parser.add_argument("--image-size", type=int, default=32)
    args = parser.parse_args()

    summary = generate(
        args.out_dir,
        n_sources=args.n_sources,
        n_test_a=args.n_test_a,
        n_test_b=args.n_test_b,
        seed=args.seed,
        with_images=args.with_images,
        image_size=args.image_size,
    )
    summary["note"] = "SYNTHETIC data - not competition data; results are pipeline self-tests"
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
