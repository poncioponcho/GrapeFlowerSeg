"""Manifest construction: ``labels.csv`` -> typed records + source-image grouping.

Why a manifest at all (HardLane ``src/data/manifest.py`` pattern): every
downstream stage (splitting, training, OOF, submission) reads one validated,
fingerprinted table instead of re-parsing raw CSVs with slightly different
rules. One parse, one truth.

The critical domain fact for this competition: the organiser split
train/A-board/B-board **by original source image (果穗图)**, so every berry
cropped from one source image lives entirely inside a single subset. Local
validation must reproduce that grouping or the score is inflated. That is why
``source_id`` is a first-class manifest field, not an afterthought.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from common.checksum import sha256_file
from common.config import Config, load_config
from common.io_utils import read_csv_rows, write_csv_rows

MANIFEST_FIELDS = (
    "image_id",
    "file_name",
    "label",
    "source_id",
    "split",
    "fold",
)

_SUFFIX_RE = re.compile(r"\.(jpg|jpeg|png|bmp|tif|tiff)$", re.IGNORECASE)


class ManifestError(ValueError):
    """Raised when labels.csv violates the competition contract."""


@dataclass(frozen=True)
class ManifestRecord:
    image_id: str
    file_name: str
    label: int
    source_id: str
    split: str = "train"
    fold: int = -1

    def to_row(self) -> dict[str, object]:
        return asdict(self)


def derive_source_id(file_name: str, cfg: Config) -> str:
    """Derive the source-image (果穗) group key from a crop's file name.

    Resolution order:

    1. A directory component wins, e.g. ``src_017/berry_003.jpg`` -> ``src_017``.
       This is the only unambiguous signal when present.
    2. ``source_key.regex``, if configured. The pattern must expose a named
       group ``source``, e.g. ``^(?P<source>src_\\d+)_``.
    3. Fallback: the substring before the **first** delimiter in
       ``source_key.delimiters``, e.g. ``src0007_012.jpg`` -> ``src0007``.

    ⚠️ Rules 2-3 are guesses until real file names are inspected. Note the
    fallback is wrong for names whose source token itself contains a delimiter
    (``src_017_berry_003.jpg`` -> ``src``); use ``regex`` for those. The
    diagnostic :func:`source_grouping_report` is how this gets validated: on
    the real training set ``n_sources`` must land near the official
    ``competition.n_train_sources`` in ``configs/default.yaml``, not equal to
    the number of images.
    """
    source_cfg = cfg.data.source_key
    normalized = str(file_name).replace("\\", "/").strip()
    if source_cfg.mode == "column":
        raise ManifestError(
            "derive_source_id called in column mode; read the column instead"
        )

    parent = str(Path(normalized).parent)
    if parent not in (".", "", "/"):
        return parent

    stem = _SUFFIX_RE.sub("", Path(normalized).name)

    pattern = source_cfg.get("regex")
    if pattern:
        match = re.search(str(pattern), stem)
        if match is None or "source" not in (match.groupdict() or {}):
            raise ManifestError(
                f"source_key.regex {pattern!r} did not match a named 'source' group "
                f"in {stem!r}"
            )
        return match.group("source")

    for delimiter in source_cfg.delimiters:
        if delimiter in stem:
            return stem.split(delimiter, 1)[0]
    return stem


def read_labels(
    cfg: Config | None = None,
    *,
    labels_csv: str | Path | None = None,
    require_images: bool = False,
) -> list[ManifestRecord]:
    """Parse ``labels.csv`` into manifest records with validation.

    Enforces: non-empty, unique ``image_id``, integer label inside the declared
    label set, and non-empty file name. With ``require_images=True`` every
    referenced image must exist on disk.
    """
    cfg = cfg or load_config()
    labels_path = Path(labels_csv) if labels_csv is not None else cfg.path("data.labels_csv")
    if not labels_path.is_file():
        raise FileNotFoundError(
            f"labels.csv not found at {labels_path}. "
            "Download the official training data into data/raw/ first "
            "(see docs/recon_findings.md)."
        )

    columns = cfg.data.columns
    id_col, name_col, label_col = (
        columns.image_id,
        columns.file_name,
        columns.label,
    )
    valid_labels = {int(value) for value in cfg.competition.labels}
    image_dir = cfg.path("data.image_dir")

    rows = read_csv_rows(labels_path)
    if not rows:
        raise ManifestError(f"{labels_path} is empty")
    missing_columns = {id_col, name_col, label_col} - set(rows[0])
    if missing_columns:
        raise ManifestError(
            f"{labels_path} missing required column(s): {sorted(missing_columns)}; "
            f"found {sorted(rows[0])}"
        )

    records: list[ManifestRecord] = []
    seen_ids: set[str] = set()
    use_column = cfg.data.source_key.mode == "column"
    source_col = cfg.data.source_key.column

    for line_number, row in enumerate(rows, start=2):  # header is line 1
        image_id = (row.get(id_col) or "").strip()
        file_name = (row.get(name_col) or "").strip()
        raw_label = (row.get(label_col) or "").strip()

        if not image_id:
            raise ManifestError(f"{labels_path}:L{line_number}: empty image_id")
        if image_id in seen_ids:
            raise ManifestError(f"{labels_path}:L{line_number}: duplicate image_id {image_id!r}")
        if not file_name:
            raise ManifestError(f"{labels_path}:L{line_number}: empty file_name")
        try:
            label = int(raw_label)
        except ValueError as exc:
            raise ManifestError(
                f"{labels_path}:L{line_number}: label {raw_label!r} is not an integer"
            ) from exc
        if label not in valid_labels:
            raise ManifestError(
                f"{labels_path}:L{line_number}: label {label} outside {sorted(valid_labels)}"
            )

        if use_column:
            source_id = (row.get(source_col) or "").strip()
            if not source_id:
                raise ManifestError(
                    f"{labels_path}:L{line_number}: empty source column {source_col!r}"
                )
        else:
            source_id = derive_source_id(file_name, cfg)

        if require_images and not (image_dir / file_name).is_file():
            raise ManifestError(
                f"{labels_path}:L{line_number}: image not found: {image_dir / file_name}"
            )

        seen_ids.add(image_id)
        records.append(
            ManifestRecord(
                image_id=image_id,
                file_name=file_name,
                label=label,
                source_id=source_id,
            )
        )
    return records


def source_grouping_report(records: Iterable[ManifestRecord]) -> dict:
    """Diagnostics to validate the source-id convention against real data.

    If ``n_sources`` equals ``n_images`` the derivation failed to group anything
    and the local split would leak. Compare ``n_sources`` with the official
    count recorded in ``configs/default.yaml``.
    """
    records = list(records)
    groups: dict[str, list[int]] = {}
    for record in records:
        groups.setdefault(record.source_id, []).append(record.label)
    sizes = sorted((len(values) for values in groups.values()), reverse=True)
    label_histogram: dict[int, int] = {}
    for values in groups.values():
        for label in values:
            label_histogram[label] = label_histogram.get(label, 0) + 1
    multi_label_sources = sum(1 for values in groups.values() if len(set(values)) > 1)
    return {
        "n_images": len(records),
        "n_sources": len(groups),
        "images_per_source_min": sizes[-1] if sizes else 0,
        "images_per_source_max": sizes[0] if sizes else 0,
        "images_per_source_mean": round(len(records) / len(groups), 2) if groups else 0.0,
        "sources_with_multiple_labels": multi_label_sources,
        "label_histogram": dict(sorted(label_histogram.items())),
        "grouping_looks_broken": len(groups) == len(records) and len(records) > 1,
    }


def write_manifest(path: str | Path, records: Sequence[ManifestRecord]) -> Path:
    return write_csv_rows(path, (r.to_row() for r in records), MANIFEST_FIELDS)


def read_manifest(path: str | Path) -> list[ManifestRecord]:
    rows = read_csv_rows(path)
    return [
        ManifestRecord(
            image_id=row["image_id"],
            file_name=row["file_name"],
            label=int(row["label"]),
            source_id=row["source_id"],
            split=row.get("split") or "train",
            fold=int(row["fold"]) if str(row.get("fold", "")).strip() not in ("", "-1") else -1,
        )
        for row in rows
    ]


def build_manifest(cfg: Config | None = None, *, require_images: bool = False) -> Path:
    """CLI entry: labels.csv -> data/processed/manifest.csv (+ fingerprint)."""
    cfg = cfg or load_config()
    records = read_labels(cfg, require_images=require_images)
    out = cfg.path("data.manifest_path")
    write_manifest(out, records)
    report = source_grouping_report(records)
    report["labels_csv_sha256"] = sha256_file(cfg.path("data.labels_csv"))
    report["manifest_path"] = str(out)
    return out


def main() -> None:  # pragma: no cover - CLI glue
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Build manifest from labels.csv")
    parser.add_argument("--labels-csv", type=Path, default=None)
    parser.add_argument("--require-images", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config()
    records = read_labels(cfg, labels_csv=args.labels_csv, require_images=args.require_images)
    out = args.out or cfg.path("data.manifest_path")
    write_manifest(out, records)
    report = source_grouping_report(records)
    report["labels_csv_sha256"] = sha256_file(args.labels_csv or cfg.path("data.labels_csv"))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
