"""result.json construction and the five-field SHA declaration.

Both files are checked by the platform with hard failures (no partial credit),
so every rejection rule gets a named test here.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from submit.result_json import (
    ResultJsonError,
    TestImage,
    build_result,
    empty_result,
    encode_mask_to_rle,
    load_test_manifest,
    write_result_json,
)
from submit.solution_commit import (
    CommitError,
    SolutionCommit,
    build_commit,
    hash_zip,
    parse_commit_text,
    verify_against_solution,
    write_commit,
)

IMAGES = [
    TestImage("IMG_1.jpg", 2048, 1152),
    TestImage("IMG_2.jpg", 2048, 1152),
    TestImage("IMG_3.jpg", 2048, 1152),
]


def _instance(category: int = 0, score: float = 0.9, counts: str = "abc") -> dict:
    return {
        "category_id": category,
        "score": score,
        "segmentation": {"size": [1152, 2048], "counts": counts},
    }


# --------------------------------------------------------------- result.json


def test_empty_result_is_valid_and_covers_every_image(base_config) -> None:
    payload = empty_result(IMAGES, base_config)
    assert payload["version"] == "1.0"
    assert len(payload["results"]) == 3
    assert all(record["instances"] == [] for record in payload["results"])
    assert [record["image_id"] for record in payload["results"]] == [i.image_id for i in IMAGES]


def test_records_follow_manifest_order(base_config) -> None:
    payload = build_result(IMAGES, {"IMG_2.jpg": [_instance()]}, base_config)
    assert [r["image_id"] for r in payload["results"]] == ["IMG_1.jpg", "IMG_2.jpg", "IMG_3.jpg"]


def test_missing_image_becomes_empty_not_absent(base_config) -> None:
    payload = build_result(IMAGES, {"IMG_1.jpg": [_instance()]}, base_config)
    by_id = {r["image_id"]: r for r in payload["results"]}
    assert by_id["IMG_2.jpg"]["instances"] == []
    assert len(payload["results"]) == 3


def test_prediction_for_unknown_image_is_rejected(base_config) -> None:
    with pytest.raises(ResultJsonError, match="not test images"):
        build_result(IMAGES, {"IMG_999.jpg": [_instance()]}, base_config)


@pytest.mark.parametrize("category", [2, 3, -1])
def test_category_outside_submit_labels_is_rejected(base_config, category) -> None:
    with pytest.raises(ResultJsonError, match="category_id must be an int in"):
        build_result(IMAGES, {"IMG_1.jpg": [_instance(category=category)]}, base_config)


def test_bool_category_is_rejected(base_config) -> None:
    with pytest.raises(ResultJsonError, match="category_id"):
        build_result(IMAGES, {"IMG_1.jpg": [_instance(category=True)]}, base_config)


@pytest.mark.parametrize("score", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_score_is_rejected(base_config, score) -> None:
    with pytest.raises(ResultJsonError, match="score"):
        build_result(IMAGES, {"IMG_1.jpg": [_instance(score=score)]}, base_config)


def test_bool_score_is_rejected(base_config) -> None:
    with pytest.raises(ResultJsonError, match="score must be a number"):
        build_result(IMAGES, {"IMG_1.jpg": [_instance(score=True)]}, base_config)


def test_extra_instance_field_is_rejected(base_config) -> None:
    bad = _instance()
    bad["bbox"] = [0, 0, 1, 1]
    with pytest.raises(ResultJsonError, match="exactly category_id/score/segmentation"):
        build_result(IMAGES, {"IMG_1.jpg": [bad]}, base_config)


def test_polygon_segmentation_is_rejected(base_config) -> None:
    bad = _instance()
    bad["segmentation"] = [[0.0, 0.0, 1.0, 0.0, 1.0, 1.0]]
    with pytest.raises(ResultJsonError, match="segmentation"):
        build_result(IMAGES, {"IMG_1.jpg": [bad]}, base_config)


def test_rle_size_must_be_two_ints(base_config) -> None:
    bad = _instance()
    bad["segmentation"]["size"] = [1152.0, 2048.0]
    with pytest.raises(ResultJsonError, match="size must be"):
        build_result(IMAGES, {"IMG_1.jpg": [bad]}, base_config)


def test_instances_are_sorted_by_score_and_truncated(base_config) -> None:
    many = [_instance(score=0.1 * index) for index in range(1, 11)]
    payload = build_result(
        IMAGES, {"IMG_1.jpg": many}, base_config, max_instances_per_image=3
    )
    scores = [i["score"] for i in payload["results"][0]["instances"]]
    assert scores == sorted(scores, reverse=True)
    assert len(scores) == 3
    assert scores[0] == pytest.approx(1.0)


def test_drop_invalid_skips_bad_instances(base_config) -> None:
    good, bad = _instance(), _instance(category=9)
    payload = build_result(
        IMAGES, {"IMG_1.jpg": [good, bad]}, base_config, drop_invalid=True
    )
    assert len(payload["results"][0]["instances"]) == 1


def test_encode_mask_to_rle_roundtrip() -> None:
    import numpy as np
    from pycocotools import mask as mask_utils

    mask = np.zeros((64, 48), dtype=np.uint8)
    mask[10:30, 5:25] = 1
    rle = encode_mask_to_rle(mask)
    assert rle["size"] == [64, 48]
    assert isinstance(rle["counts"], str)
    decoded = mask_utils.decode({"size": rle["size"], "counts": rle["counts"].encode()})
    assert np.array_equal(decoded.astype(bool), mask.astype(bool))


def test_encode_empty_mask_is_rejected() -> None:
    import numpy as np

    with pytest.raises(ResultJsonError, match="mask is empty"):
        encode_mask_to_rle(np.zeros((8, 8), dtype=np.uint8))


def test_write_result_json_roundtrip(tmp_path, base_config) -> None:
    path = write_result_json(empty_result(IMAGES, base_config), tmp_path / "result.json")
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert json.loads(raw.decode("utf-8"))["version"] == "1.0"


def test_load_test_manifest_reads_the_official_file(base_config, data_available) -> None:
    if not data_available:
        pytest.skip("competition data not present")
    images = load_test_manifest(base_config.path("data.test_b_manifest"))
    assert len(images) == int(base_config.competition.n_test_b_images)
    assert all(image.image_id.endswith(".jpg") for image in images)
    assert all(image.width == 2048 and image.height == 1152 for image in images)


# ---------------------------------------------------------- solution_commit


def _make_solution(tmp_path: Path, payload: bytes = b"hello") -> Path:
    path = tmp_path / "solution.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("inference.py", payload)
    return path


def test_commit_has_exactly_five_fields_in_order(tmp_path, base_config) -> None:
    commit = build_commit(_make_solution(tmp_path), base_config)
    text = commit.to_text(base_config)
    lines = text.splitlines()
    assert len(lines) == 5
    assert [line.split("=")[0] for line in lines] == list(base_config.submit.commit_fields)
    assert text.endswith("\n") and not text.endswith("\n\n")


def test_commit_hash_and_size_match_the_archive(tmp_path, base_config) -> None:
    solution = _make_solution(tmp_path)
    commit = build_commit(solution, base_config)
    sha, size = hash_zip(solution)
    assert commit.solution_sha256 == sha
    assert commit.solution_size == size
    verify_against_solution(commit, solution)


def test_commit_verification_detects_a_changed_archive(tmp_path, base_config) -> None:
    solution = _make_solution(tmp_path)
    commit = build_commit(solution, base_config)
    _make_solution(tmp_path, payload=b"different bytes now")
    with pytest.raises(CommitError, match="does not match"):
        verify_against_solution(commit, solution)


def test_wrong_archive_name_is_rejected(tmp_path, base_config) -> None:
    path = tmp_path / "not_solution.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("a", b"b")
    with pytest.raises(CommitError, match="must be named"):
        build_commit(path, base_config)


def test_non_zip_is_rejected(tmp_path, base_config) -> None:
    path = tmp_path / "solution.zip"
    path.write_bytes(b"not a zip")
    with pytest.raises(CommitError, match="not a valid zip"):
        build_commit(path, base_config)


def test_written_commit_is_valid_utf8_five_lines(tmp_path, base_config) -> None:
    commit = build_commit(_make_solution(tmp_path), base_config)
    path = write_commit(commit, tmp_path / "solution_commit.txt", base_config)
    text = path.read_text(encoding="utf-8")
    assert len(text.splitlines()) == 5
    parse_commit_text(text, base_config)


def test_commit_parser_rejects_malformed_variants(base_config) -> None:
    good = (
        "solution_name=solution.zip\n"
        "hash_algorithm=SHA-256\n"
        f"solution_sha256={'a' * 64}\n"
        "solution_size=1234\n"
        "b_data_version=B-v1\n"
    )
    parse_commit_text(good, base_config)

    cases = {
        "blank line": good.replace("\n", "\n\n", 1),
        "missing field": good.replace("b_data_version=B-v1\n", ""),
        "unknown field": good + "extra=1\n",
        "duplicate field": good + "solution_size=99\n",
        "bad hash": good.replace("a" * 64, "xyz"),
        "bad size": good.replace("solution_size=1234", "solution_size=01234"),
        "wrong version": good.replace("B-v1", "A-v1"),
        "wrong algorithm": good.replace("SHA-256", "MD5"),
        "padded value": good.replace("solution_size=1234", "solution_size= 1234"),
        "empty value": good.replace("solution_size=1234", "solution_size="),
    }
    for label, text in cases.items():
        with pytest.raises(CommitError):
            parse_commit_text(text, base_config)


def test_commit_field_order_in_config_is_the_official_order(base_config) -> None:
    assert list(base_config.submit.commit_fields) == [
        "solution_name",
        "hash_algorithm",
        "solution_sha256",
        "solution_size",
        "b_data_version",
    ]
