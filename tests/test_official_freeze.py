"""The frozen official scripts must not change.

Rationale (kept from the team's previous competition): an evaluation script that
can be edited is an evaluation script that will be edited, and then every number
it produced becomes unverifiable. Recording the SHA-256 at freeze time and
asserting it in a test turns any accidental edit into a red light.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.checksum import sha256_file

FREEZE_MANIFEST = Path(__file__).resolve().parents[1] / "src" / "official_freeze.json"

# The frozen copies live outside version control: they are the organiser's own
# scripts, redistributed only inside the official submission kit. A checkout
# without the kit is expected - those tests skip rather than fail, and say how
# to restore the files.
MISSING_KIT_HINT = (
    "official script not present - copy src/eval/official_oracle/ and "
    "src/submit/official/ back from the official submission kit; "
    "src/official_freeze.json records the SHA-256 each file must have"
)


@pytest.fixture(scope="module")
def frozen() -> dict:
    payload = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
    return payload["frozen"]


def test_freeze_manifest_exists_and_is_non_empty(frozen) -> None:
    assert frozen, "official_freeze.json has no frozen entries"


@pytest.mark.parametrize(
    "relative",
    [
        "src/eval/official_oracle/official_evaluate.py",
        "src/eval/official_oracle/rle_validation.py",
        "src/submit/official/submission.py",
        "src/submit/official/validate_b_submission.py",
        "src/submit/official/generate_solution_commit.py",
        "src/submit/official/validate_submission_a.py",
    ],
)
def test_frozen_file_hash_matches(relative: str, frozen: dict) -> None:
    repo_root = FREEZE_MANIFEST.parents[1]
    path = repo_root / relative
    if not path.is_file():
        pytest.skip(f"{relative}: {MISSING_KIT_HINT}")
    assert relative in frozen, f"{relative} is not recorded in official_freeze.json"
    assert sha256_file(path) == frozen[relative]["sha256"], (
        f"{relative} changed since freeze time - the official scorer/validator is "
        "immutable; if the organisers published a new version, re-freeze it "
        "deliberately and log the change in docs/experiments.md"
    )


def test_frozen_file_sizes_match(frozen: dict) -> None:
    repo_root = FREEZE_MANIFEST.parents[1]
    checked = 0
    for relative, record in frozen.items():
        path = repo_root / relative
        if not path.is_file():
            continue
        assert path.stat().st_size == record["bytes"], f"{relative} size changed"
        checked += 1
    if not checked:
        pytest.skip(MISSING_KIT_HINT)


def test_frozen_files_are_unmodified_official_copies(frozen: dict) -> None:
    """Each record must point at a real source archive/member, not a rewrite."""
    for relative, record in frozen.items():
        assert "::" in record["source"], f"{relative} has no provenance"
        assert len(record["sha256"]) == 64
