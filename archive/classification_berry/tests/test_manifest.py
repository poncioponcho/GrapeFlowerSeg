"""Manifest parsing: contract violations must fail loudly at parse time."""
from __future__ import annotations

from pathlib import Path

import pytest

from data.manifest import (
    ManifestError,
    derive_source_id,
    read_labels,
    read_manifest,
    source_grouping_report,
    write_manifest,
)


def _write_labels(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_reads_valid_labels_csv(sandbox_cfg) -> None:
    records = read_labels(sandbox_cfg)
    assert records
    assert all(record.label in {0, 1, 2} for record in records)
    assert all(record.source_id for record in records)
    assert len({record.image_id for record in records}) == len(records)


def test_missing_file_raises_actionable_error(sandbox_cfg, tmp_path: Path) -> None:
    sandbox_cfg["data"]["labels_csv"] = str(tmp_path / "nope.csv")
    with pytest.raises(FileNotFoundError, match="labels.csv not found"):
        read_labels(sandbox_cfg)


def test_missing_required_column_is_rejected(sandbox_cfg, tmp_path: Path) -> None:
    path = _write_labels(tmp_path / "labels.csv", "image_id,label\n000,0\n")
    sandbox_cfg["data"]["labels_csv"] = str(path)
    with pytest.raises(ManifestError, match="missing required column"):
        read_labels(sandbox_cfg)


def test_duplicate_image_id_is_rejected(sandbox_cfg, tmp_path: Path) -> None:
    path = _write_labels(
        tmp_path / "labels.csv",
        "image_id,file_name,label\n000,src1_0.jpg,0\n000,src1_1.jpg,1\n",
    )
    sandbox_cfg["data"]["labels_csv"] = str(path)
    with pytest.raises(ManifestError, match="duplicate image_id"):
        read_labels(sandbox_cfg)


def test_non_integer_label_is_rejected(sandbox_cfg, tmp_path: Path) -> None:
    path = _write_labels(
        tmp_path / "labels.csv",
        "image_id,file_name,label\n000,src1_0.jpg,early\n",
    )
    sandbox_cfg["data"]["labels_csv"] = str(path)
    with pytest.raises(ManifestError, match="not an integer"):
        read_labels(sandbox_cfg)


def test_out_of_range_label_is_rejected(sandbox_cfg, tmp_path: Path) -> None:
    path = _write_labels(
        tmp_path / "labels.csv",
        "image_id,file_name,label\n000,src1_0.jpg,3\n",
    )
    sandbox_cfg["data"]["labels_csv"] = str(path)
    with pytest.raises(ManifestError, match="outside"):
        read_labels(sandbox_cfg)


def test_empty_image_id_is_rejected(sandbox_cfg, tmp_path: Path) -> None:
    path = _write_labels(
        tmp_path / "labels.csv",
        "image_id,file_name,label\n,src1_0.jpg,0\n",
    )
    sandbox_cfg["data"]["labels_csv"] = str(path)
    with pytest.raises(ManifestError, match="empty image_id"):
        read_labels(sandbox_cfg)


def test_require_images_flag(sandbox_cfg, tmp_path: Path) -> None:
    """With no images on disk, require_images must fail (guards silent 0-image runs)."""
    with pytest.raises(ManifestError, match="image not found"):
        read_labels(sandbox_cfg, require_images=True)


def test_manifest_roundtrip(sandbox_cfg, tmp_path: Path) -> None:
    records = read_labels(sandbox_cfg)
    path = write_manifest(tmp_path / "manifest.csv", records)
    reloaded = read_manifest(path)
    assert [r.to_row() for r in reloaded] == [r.to_row() for r in records]


def test_grouping_report_shape(sandbox_cfg) -> None:
    records = read_labels(sandbox_cfg)
    report = source_grouping_report(records)
    assert report["n_images"] == len(records)
    assert report["n_sources"] < report["n_images"]
    assert report["grouping_looks_broken"] is False
    assert set(report["label_histogram"]) <= {0, 1, 2}


def test_derive_source_id_handles_all_suffixes_and_delimiters(sandbox_cfg) -> None:
    assert derive_source_id("src0007_012.JPG", sandbox_cfg) == "src0007"
    assert derive_source_id("src0007-012.jpeg", sandbox_cfg) == "src0007"
    assert derive_source_id("nested/src0007/012.png", sandbox_cfg) == "nested/src0007"
    assert derive_source_id("lonely.png", sandbox_cfg) == "lonely"


def test_column_mode_reads_source_id(sandbox_cfg, tmp_path: Path) -> None:
    sandbox_cfg["data"]["source_key"]["mode"] = "column"
    path = _write_labels(
        tmp_path / "labels.csv",
        "image_id,file_name,label,source_image_id\n"
        "000,a.jpg,0,SRC_A\n"
        "001,b.jpg,1,SRC_A\n",
    )
    sandbox_cfg["data"]["labels_csv"] = str(path)
    records = read_labels(sandbox_cfg)
    assert {r.source_id for r in records} == {"SRC_A"}


def test_column_mode_rejects_empty_source(sandbox_cfg, tmp_path: Path) -> None:
    sandbox_cfg["data"]["source_key"]["mode"] = "column"
    path = _write_labels(
        tmp_path / "labels.csv",
        "image_id,file_name,label,source_image_id\n000,a.jpg,0,\n",
    )
    sandbox_cfg["data"]["labels_csv"] = str(path)
    with pytest.raises(ManifestError, match="empty source column"):
        read_labels(sandbox_cfg)
