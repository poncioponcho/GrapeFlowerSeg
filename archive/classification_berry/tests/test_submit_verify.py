"""The five interception classes of the submission safety net.

Acceptance criterion #1 requires "verify_submit 对 5 类格式错误的拦截测试".
Each class below has an explicit, named test so a regression shows up as a
failing test rather than as an invalid submission on the leaderboard.

1. not valid UTF-8
2. missing header row / wrong column names
3. ``image_id`` missing, duplicated, or outside the test set
4. ``label`` not an integer, or outside {0,1,2}
5. archive root not a flat, exact file set (multi-level directory / extra file)
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from submit.prepare_submit import (
    PrepareError,
    canonicalize_predictions,
    prepare_submission,
    read_raw_predictions,
    write_predictions_csv,
)
from submit.verify_submit import read_expected_ids, verify_predictions_bytes, verify_submission

EXPECTED = ["img000", "img001", "img002", "img003"]
HEADER = b"image_id,label\n"


def _csv(*rows: str) -> bytes:
    return HEADER + "".join(f"{row}\n" for row in rows).encode("utf-8")


def _zip(tmp_path: Path, name: str, members: dict[str, bytes]) -> Path:
    path = tmp_path / name
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname, payload in members.items():
            zf.writestr(arcname, payload)
    return path


def _good_csv() -> bytes:
    return _csv("img000,0", "img001,1", "img002,2", "img003,0")


# ---------------------------------------------------------------- happy path


def test_valid_predictions_pass(sandbox_cfg) -> None:
    result = verify_predictions_bytes(_good_csv(), EXPECTED, sandbox_cfg)
    assert result.ok, result.report
    assert result.n_rows == 4


def test_valid_submit_zip_passes(sandbox_cfg, tmp_path: Path) -> None:
    archive = _zip(tmp_path, "submit.zip", {"predictions.csv": _good_csv()})
    result = verify_submission(archive, EXPECTED, sandbox_cfg, kind="a")
    assert result.ok, result.report
    assert result.n_rows == 4


# ------------------------------------------------- class 1: encoding is UTF-8


def test_class1_non_utf8_is_rejected(sandbox_cfg) -> None:
    payload = HEADER + "img000,0\nimg001,1\nimg002,2\nimg003,\xff\xfe\n".encode("latin-1")
    result = verify_predictions_bytes(payload, EXPECTED, sandbox_cfg)
    assert not result.ok
    assert any("not valid UTF-8" in err for err in result.errors)


def test_class1_utf8_bom_is_rejected(sandbox_cfg) -> None:
    result = verify_predictions_bytes(b"\xef\xbb\xbf" + _good_csv(), EXPECTED, sandbox_cfg)
    assert not result.ok
    assert any("BOM" in err for err in result.errors)


# ------------------------------------------- class 2: header row and columns


def test_class2_missing_header_is_rejected(sandbox_cfg) -> None:
    payload = b"img000,0\nimg001,1\nimg002,2\nimg003,0\n"
    result = verify_predictions_bytes(payload, EXPECTED, sandbox_cfg)
    assert not result.ok
    assert any("header mismatch" in err for err in result.errors)


def test_class2_wrong_column_names_are_rejected(sandbox_cfg) -> None:
    payload = b"id,class\n" + b"img000,0\nimg001,1\nimg002,2\nimg003,0\n"
    result = verify_predictions_bytes(payload, EXPECTED, sandbox_cfg)
    assert not result.ok
    assert any("header mismatch" in err for err in result.errors)


def test_class2_empty_file_is_rejected(sandbox_cfg) -> None:
    result = verify_predictions_bytes(b"", EXPECTED, sandbox_cfg)
    assert not result.ok
    assert any("empty" in err for err in result.errors)


# ---------------------------------------------------- class 3: image_id set


def test_class3_missing_image_id_is_rejected(sandbox_cfg) -> None:
    payload = _csv("img000,0", "img001,1", "img002,2")
    result = verify_predictions_bytes(payload, EXPECTED, sandbox_cfg)
    assert not result.ok
    assert any("missing from predictions" in err for err in result.errors)
    assert any("img003" in err for err in result.errors)


def test_class3_duplicate_image_id_is_rejected(sandbox_cfg) -> None:
    payload = _csv("img000,0", "img001,1", "img002,2", "img002,0", "img003,0")
    result = verify_predictions_bytes(payload, EXPECTED, sandbox_cfg)
    assert not result.ok
    assert any("duplicated image_id" in err for err in result.errors)


def test_class3_extra_image_id_is_rejected(sandbox_cfg) -> None:
    payload = _csv("img000,0", "img001,1", "img002,2", "img003,0", "img999,1")
    result = verify_predictions_bytes(payload, EXPECTED, sandbox_cfg)
    assert not result.ok
    assert any("not in the test set" in err for err in result.errors)


def test_class3_malformed_row_shape_is_rejected(sandbox_cfg) -> None:
    payload = _csv("img000,0", "img001,1", "img002,2", "img003")
    result = verify_predictions_bytes(payload, EXPECTED, sandbox_cfg)
    assert not result.ok
    assert any("malformed row" in err for err in result.errors)


# ------------------------------------------------------- class 4: label value


def test_class4_non_integer_label_is_rejected(sandbox_cfg) -> None:
    payload = _csv("img000,0", "img001,1", "img002,2", "img003,early")
    result = verify_predictions_bytes(payload, EXPECTED, sandbox_cfg)
    assert not result.ok
    assert any("not an integer" in err for err in result.errors)


def test_class4_out_of_range_label_is_rejected(sandbox_cfg) -> None:
    payload = _csv("img000,0", "img001,1", "img002,2", "img003,3")
    result = verify_predictions_bytes(payload, EXPECTED, sandbox_cfg)
    assert not result.ok
    assert any("outside [0, 1, 2]" in err for err in result.errors)


def test_class4_negative_label_is_rejected(sandbox_cfg) -> None:
    payload = _csv("img000,0", "img001,1", "img002,2", "img003,-1")
    result = verify_predictions_bytes(payload, EXPECTED, sandbox_cfg)
    assert not result.ok
    assert any("outside [0, 1, 2]" in err for err in result.errors)


def test_class4_float_label_is_rejected(sandbox_cfg) -> None:
    payload = _csv("img000,0", "img001,1", "img002,2", "img003,1.0")
    result = verify_predictions_bytes(payload, EXPECTED, sandbox_cfg)
    assert not result.ok
    assert any("not an integer" in err for err in result.errors)


# ------------------------------------------------------- class 5: zip layout


def test_class5_multilevel_directory_is_rejected(sandbox_cfg, tmp_path: Path) -> None:
    archive = _zip(
        tmp_path, "submit.zip", {"submit/predictions.csv": _good_csv()}
    )
    result = verify_submission(archive, EXPECTED, sandbox_cfg, kind="a")
    assert not result.ok
    assert any("not flat" in err for err in result.errors)
    assert any("missing required root file" in err for err in result.errors)


def test_class5_deeply_nested_directory_is_rejected(sandbox_cfg, tmp_path: Path) -> None:
    archive = _zip(
        tmp_path, "submit.zip", {"a/b/c/predictions.csv": _good_csv()}
    )
    result = verify_submission(archive, EXPECTED, sandbox_cfg, kind="a")
    assert not result.ok
    assert any("not flat" in err for err in result.errors)


def test_class5_extra_root_file_is_rejected(sandbox_cfg, tmp_path: Path) -> None:
    archive = _zip(
        tmp_path,
        "submit.zip",
        {"predictions.csv": _good_csv(), "notes.txt": b"oops"},
    )
    result = verify_submission(archive, EXPECTED, sandbox_cfg, kind="a")
    assert not result.ok
    assert any("extra root file" in err for err in result.errors)


def test_class5_renamed_predictions_file_is_rejected(sandbox_cfg, tmp_path: Path) -> None:
    archive = _zip(tmp_path, "submit.zip", {"predictions(1).csv": _good_csv()})
    result = verify_submission(archive, EXPECTED, sandbox_cfg, kind="a")
    assert not result.ok
    assert any("missing required root file" in err for err in result.errors)


def test_corrupt_archive_is_rejected(sandbox_cfg, tmp_path: Path) -> None:
    archive = tmp_path / "submit.zip"
    archive.write_bytes(b"this is not a zip")
    result = verify_submission(archive, EXPECTED, sandbox_cfg, kind="a")
    assert not result.ok
    assert any("not a readable zip" in err for err in result.errors)


# ------------------------------------------------------------ B-board rules


def test_b_board_valid_archive_passes(sandbox_cfg, tmp_path: Path) -> None:
    inner = _zip(tmp_path, "submit.zip", {"predictions.csv": _good_csv()})
    outer = _zip(
        tmp_path,
        "b_submission.zip",
        {"submit.zip": inner.read_bytes(), "solution_commit.txt": b"model: convnext_tiny\n"},
    )
    result = verify_submission(outer, EXPECTED, sandbox_cfg, kind="b")
    assert result.ok, result.report
    assert result.n_rows == 4


def test_b_board_missing_solution_commit_is_rejected(sandbox_cfg, tmp_path: Path) -> None:
    inner = _zip(tmp_path, "submit.zip", {"predictions.csv": _good_csv()})
    outer = _zip(tmp_path, "b_submission.zip", {"submit.zip": inner.read_bytes()})
    result = verify_submission(outer, EXPECTED, sandbox_cfg, kind="b")
    assert not result.ok
    assert any("missing required root file" in err for err in result.errors)


def test_b_board_empty_solution_commit_is_rejected(sandbox_cfg, tmp_path: Path) -> None:
    inner = _zip(tmp_path, "submit.zip", {"predictions.csv": _good_csv()})
    outer = _zip(
        tmp_path,
        "b_submission.zip",
        {"submit.zip": inner.read_bytes(), "solution_commit.txt": b"   \n"},
    )
    result = verify_submission(outer, EXPECTED, sandbox_cfg, kind="b")
    assert not result.ok
    assert any("solution_commit.txt is empty" in err for err in result.errors)


def test_b_board_propagates_nested_errors(sandbox_cfg, tmp_path: Path) -> None:
    bad_inner = _zip(tmp_path, "submit.zip", {"predictions.csv": _csv("img000,0")})
    outer = _zip(
        tmp_path,
        "b_submission.zip",
        {"submit.zip": bad_inner.read_bytes(), "solution_commit.txt": b"model: x\n"},
    )
    result = verify_submission(outer, EXPECTED, sandbox_cfg, kind="b")
    assert not result.ok
    assert any("submit.zip" in err and "missing" in err for err in result.errors)


def test_b_board_nested_multilevel_is_rejected(sandbox_cfg, tmp_path: Path) -> None:
    bad_inner = _zip(tmp_path, "submit.zip", {"submit/predictions.csv": _good_csv()})
    outer = _zip(
        tmp_path,
        "b_submission.zip",
        {"submit.zip": bad_inner.read_bytes(), "solution_commit.txt": b"model: x\n"},
    )
    result = verify_submission(outer, EXPECTED, sandbox_cfg, kind="b")
    assert not result.ok
    assert any("not flat" in err for err in result.errors)


# ----------------------------------------------------------- helper behaviour


def test_read_expected_ids_from_csv_and_txt(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample_submission.csv"
    csv_path.write_text("image_id,label\nimg000,0\nimg001,0\n", encoding="utf-8")
    assert read_expected_ids(csv_path) == ["img000", "img001"]

    txt_path = tmp_path / "ids.txt"
    txt_path.write_text("img000\n\nimg001\n", encoding="utf-8")
    assert read_expected_ids(txt_path) == ["img000", "img001"]

    with pytest.raises(ValueError, match="duplicates"):
        read_expected_ids(["a", "a"])


def test_canonicalize_rejects_extra_ids(sandbox_cfg) -> None:
    with pytest.raises(PrepareError, match="not in the test set"):
        canonicalize_predictions(
            {"img000": 0, "img001": 1, "img002": 2, "img003": 0, "ghost": 1},
            EXPECTED,
            sandbox_cfg,
        )


def test_canonicalize_rejects_missing_without_fill(sandbox_cfg) -> None:
    with pytest.raises(PrepareError, match="has no prediction"):
        canonicalize_predictions({"img000": 0}, EXPECTED, sandbox_cfg)


def test_canonicalize_fill_missing_is_explicit_and_ordered(sandbox_cfg) -> None:
    rows, filled = canonicalize_predictions(
        {"img002": 2, "img000": 0}, EXPECTED, sandbox_cfg, fill_missing=1
    )
    assert [row[0] for row in rows] == EXPECTED
    assert filled == ["img001", "img003"]


def test_canonicalize_rejects_out_of_range_label(sandbox_cfg) -> None:
    with pytest.raises(PrepareError, match="outside"):
        canonicalize_predictions({"img000": 7}, EXPECTED, sandbox_cfg)


def test_write_predictions_csv_is_utf8_lf_no_bom(sandbox_cfg, tmp_path: Path) -> None:
    path = write_predictions_csv(
        [("img000", 0), ("img001", 2)], tmp_path / "predictions.csv", sandbox_cfg
    )
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in raw
    assert raw == b"image_id,label\nimg000,0\nimg001,2\n"


def test_read_raw_predictions_ignores_extra_columns(tmp_path: Path) -> None:
    path = tmp_path / "raw.csv"
    path.write_text("image_id,label,score\nimg000,1,0.93\n", encoding="utf-8")
    assert read_raw_predictions(path) == {"img000": 1}


# --------------------------------------------------- end-to-end prepare chain


def test_prepare_submission_a_end_to_end(sandbox_cfg, sample_submission_a: Path) -> None:
    raw = {image_id: index % 3 for index, image_id in enumerate(read_expected_ids(sample_submission_a))}
    result = prepare_submission(raw, sample_submission_a, sandbox_cfg, kind="a")
    assert result.ok, result.verify.report if result.verify else ""
    archive = Path(result.archive)
    assert archive.name == sandbox_cfg.submit.a_zip_name
    with zipfile.ZipFile(archive) as zf:
        assert zf.namelist() == ["predictions.csv"]


def test_prepare_submission_b_end_to_end(sandbox_cfg, sample_submission_b: Path) -> None:
    raw = {image_id: index % 3 for index, image_id in enumerate(read_expected_ids(sample_submission_b))}
    result = prepare_submission(
        raw,
        sample_submission_b,
        sandbox_cfg,
        kind="b",
        solution_commit={"model": "convnext_tiny", "commit": "deadbeef"},
    )
    assert result.ok, result.verify.report if result.verify else ""
    with zipfile.ZipFile(result.archive) as zf:
        assert sorted(zf.namelist()) == ["solution_commit.txt", "submit.zip"]
        assert b"deadbeef" in zf.read("solution_commit.txt")


def test_prepare_fails_loudly_on_extra_ids(sandbox_cfg, sample_submission_a: Path) -> None:
    ids = read_expected_ids(sample_submission_a)
    raw = {image_id: 0 for image_id in ids}
    raw["not_a_test_id"] = 1
    with pytest.raises(PrepareError):
        prepare_submission(raw, sample_submission_a, sandbox_cfg, kind="a")
