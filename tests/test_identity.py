"""Competition-identity guards.

The workspace contains more than one competition, and two of them are instance
segmentation with a ``result.json`` submission. Uploading a correct archive to
the wrong leaderboard burns one of only three B-board submissions, so the guards
that prevent it are tested like any other safety-critical path.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from submit.identity import (
    IdentityError,
    assert_preflight,
    banner,
    preflight,
    render_identity_record,
    write_identity_record,
)


def test_preflight_passes_on_the_real_data(cfg, data_available) -> None:
    if not data_available:
        pytest.skip("competition data not present")
    checks = preflight(cfg, split="testB")
    failed = [c for c in checks if not c.ok]
    assert not failed, "\n".join(c.render() for c in failed)


def test_preflight_covers_the_identity_signals(cfg, data_available) -> None:
    if not data_available:
        pytest.skip("competition data not present")
    names = {check.name for check in preflight(cfg, split="testB")}
    # The augmented pairing is unique to this competition and is the strongest
    # local signal that we are not looking at a different segmentation task.
    assert "augmented pairing (53 + 53)" in names
    assert "testB image count" in names
    assert "image dimensions" in names
    assert "test images present on disk" in names


def test_preflight_fails_when_the_image_count_is_wrong(cfg, data_available) -> None:
    if not data_available:
        pytest.skip("competition data not present")
    cfg["competition"]["n_test_b_images"] = 999
    checks = preflight(cfg, split="testB")
    count_check = next(c for c in checks if c.name == "testB image count")
    assert not count_check.ok
    assert "expects 999" in count_check.detail


def test_assert_preflight_raises_on_mismatch(cfg, data_available) -> None:
    if not data_available:
        pytest.skip("competition data not present")
    cfg["competition"]["n_test_b_images"] = 999
    with pytest.raises(IdentityError, match="does not look like the configured competition"):
        assert_preflight(cfg, split="testB")


def test_preflight_fails_when_images_are_missing(cfg, data_available) -> None:
    if not data_available:
        pytest.skip("competition data not present")
    cfg["data"]["test_b_images"] = str(cfg.path("data.processed_dir") / "no_such_dir")
    checks = preflight(cfg, split="testB")
    disk_check = next(c for c in checks if "image dir" in c.name or "present on disk" in c.name)
    assert not disk_check.ok


def test_banner_names_the_competition_and_the_ones_it_is_not(cfg) -> None:
    text = banner(cfg, split="testB")
    assert "GrapeFlowerSeg" in text
    assert cfg.competition.name in text
    assert "NOT" in text
    # Every confusable competition must be listed.
    for entry in cfg.competition.not_this_competition:
        assert entry["id"] in text


def test_banner_names_the_required_root_files(cfg) -> None:
    text = banner(cfg, split="testB")
    assert "result.json" in text
    assert "solution_commit.txt" in text


def test_identity_record_binds_archive_to_competition(cfg, tmp_path) -> None:
    archive = tmp_path / "b_submission.zip"
    archive.write_bytes(b"placeholder")
    text = render_identity_record(
        cfg,
        archive=archive,
        split="testB",
        commit_sha256="a" * 64,
        commit_size=1234,
        n_images=106,
        n_instances=7,
        checks=preflight(cfg, split="testB"),
    )
    assert "GrapeFlowerSeg" in text
    assert "a" * 64 in text
    assert "3 submissions" in text
    # It must warn about the specific confusion risks.
    for entry in cfg.competition.not_this_competition:
        assert entry["id"] in text


def test_identity_record_is_written_beside_not_inside_the_archive(cfg, tmp_path) -> None:
    """The platform rejects any extra file inside the archive."""
    archive = tmp_path / "b_submission.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("result.json", b'{"version": "1.0", "results": []}')
        zf.writestr("solution_commit.txt", b"solution_name=solution.zip\n")
    before = archive.read_bytes()

    path = write_identity_record(
        cfg,
        tmp_path,
        archive=archive,
        split="testB",
        commit_sha256="b" * 64,
        commit_size=10,
        n_images=1,
        n_instances=0,
        checks=[],
    )
    assert path.name == "SUBMISSION_IDENTITY.md"
    assert path.parent == tmp_path
    # The archive is byte-for-byte untouched, and gained no member.
    assert archive.read_bytes() == before
    with zipfile.ZipFile(archive) as zf:
        assert sorted(zf.namelist()) == ["result.json", "solution_commit.txt"]


def test_built_archive_root_is_exactly_the_two_required_files(tmp_path, cfg, data_available) -> None:
    """End-to-end: the identity machinery must not leak a file into the archive."""
    if not data_available:
        pytest.skip("competition data not present")
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "build_submission.py"),
            "--empty",
            "--out-dir", str(out_dir),
        ],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        timeout=900,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SUBMISSION IDENTITY" in result.stdout
    archive = out_dir / cfg.submit.b_zip_name
    with zipfile.ZipFile(archive) as zf:
        assert sorted(zf.namelist()) == ["result.json", "solution_commit.txt"]
    assert (out_dir / "SUBMISSION_IDENTITY.md").is_file()
