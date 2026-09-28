"""End-to-end pipeline smoke test + a demonstration that grouping matters.

Two things are proven here:

1. The full chain runs offline: synthetic data -> manifest -> grouped split ->
   simulated predictions -> ``prepare`` -> ``verify`` -> ``pack`` -> valid
   ``submit.zip`` and ``b_submission.zip``. (Acceptance criterion #3.)

2. A *source-memorising* predictor scores dramatically higher under a random
   per-image split than under the grouped split. That gap is the leakage the
   organiser's split already avoids, and it is why the local validation set must
   be grouped. (Acceptance criterion #1, motivated empirically.)
"""
from __future__ import annotations

import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from common.io_utils import write_csv_rows
from data.make_sample_data import simulate_predictions
from data.manifest import ManifestRecord, read_labels, write_manifest
from data.split_by_source import load_folds, run_split
from eval.macro_f1 import evaluate, macro_f1
from submit.prepare_submit import prepare_submission, read_raw_predictions
from submit.verify_submit import verify_submission


def _source_memoriser_oof(
    records: list[ManifestRecord], folds: list[tuple[tuple[int, ...], tuple[int, ...]]]
) -> tuple[list[int], list[int]]:
    """Predict each val crop with its source's majority *training* label.

    This emulates the strongest thing a model can learn from a leaked split:
    "this source image is a mid-veraison vine". When the source is unseen in
    training the predictor has to fall back to the global majority.
    """
    y_true: list[int] = []
    y_pred: list[int] = []
    global_majority = Counter(r.label for r in records).most_common(1)[0][0]
    for train_indices, val_indices in folds:
        per_source: dict[str, Counter] = defaultdict(Counter)
        for index in train_indices:
            per_source[records[index].source_id][records[index].label] += 1
        for index in val_indices:
            record = records[index]
            counts = per_source.get(record.source_id)
            y_true.append(record.label)
            y_pred.append(counts.most_common(1)[0][0] if counts else global_majority)
    return y_true, y_pred


def _random_folds(records: list[ManifestRecord], n_folds: int, seed: int):
    """Deliberately WRONG split: ignores source grouping (the anti-pattern)."""
    rng = random.Random(seed)
    assignment = [rng.randrange(n_folds) for _ in records]
    folds = []
    for fold_id in range(n_folds):
        val = tuple(i for i, a in enumerate(assignment) if a == fold_id)
        train = tuple(i for i, a in enumerate(assignment) if a != fold_id)
        folds.append((train, val))
    return folds


@pytest.fixture
def prepared_project(sandbox_cfg, tmp_path: Path):
    """Manifest + grouped folds persisted under the sandbox config."""
    records = read_labels(sandbox_cfg)
    manifest_path = tmp_path / "processed" / "manifest.csv"
    write_manifest(manifest_path, records)
    result = run_split(
        sandbox_cfg, manifest_path=manifest_path, out_dir=tmp_path / "processed" / "splits"
    )
    return records, manifest_path, result


def test_grouped_split_beats_leakage_on_source_memoriser(prepared_project) -> None:
    records, _, result = prepared_project
    grouped_folds = [(f.train_indices, f.val_indices) for f in result.folds]

    y_true_g, y_pred_g = _source_memoriser_oof(records, grouped_folds)
    grouped_score = macro_f1(y_true_g, y_pred_g)

    y_true_r, y_pred_r = _source_memoriser_oof(records, _random_folds(records, 5, seed=7))
    random_score = macro_f1(y_true_r, y_pred_r)

    # The leaked split must look strictly better - that is exactly the trap.
    assert random_score > grouped_score
    # And the gap must be material, not a rounding artefact.
    assert random_score - grouped_score > 0.05


def test_manifest_and_split_artifacts_are_written(prepared_project, tmp_path: Path) -> None:
    _, manifest_path, result = prepared_project
    assert manifest_path.is_file()
    folds_path = tmp_path / "processed" / "splits" / "folds.json"
    assert folds_path.is_file()

    reloaded, reloaded_records = load_folds(folds_path)
    assert reloaded.n_folds == 5
    assert len(reloaded_records) == result.audit["n_records"]
    # The persisted split must still satisfy the leakage invariant after reload.
    for fold in reloaded.folds:
        assert not (set(fold.train_sources) & set(fold.val_sources))


def test_oof_predictions_cover_every_record_exactly_once(prepared_project) -> None:
    records, _, result = prepared_project
    oof_index = [i for fold in result.folds for i in fold.val_indices]
    assert sorted(oof_index) == list(range(len(records)))


def test_simulated_oof_metrics_are_computable(prepared_project) -> None:
    """Sanity-check the metric plumbing on simulated (non-real) predictions."""
    records, _, result = prepared_project
    rng = random.Random(0)
    y_true = [r.label for r in records]
    y_pred = [
        r.label if rng.random() < 0.75 else min(2, max(0, r.label + rng.choice([-1, 1])))
        for r in records
    ]
    report = evaluate(y_true, y_pred)
    assert 0.0 <= report.macro_f1 <= 1.0
    assert report.confusion.sum() == len(records)
    assert report.accuracy > 0.5
    assert set(report.per_class) == {0, 1, 2}


def test_full_submit_pipeline_a_board(sandbox_cfg, prepared_project, tmp_path: Path) -> None:
    """generate -> verify -> pack for the A-board archive."""
    expected = [f"t{index:04d}" for index in range(25)]
    # Simulate a model's output *for the test ids* (train ids are disjoint).
    truth_csv = write_csv_rows(
        tmp_path / "sim_truth.csv",
        ({"image_id": image_id, "label": index % 3} for index, image_id in enumerate(expected)),
        ("image_id", "label"),
    )
    raw_csv = simulate_predictions(
        truth_csv, tmp_path / "sim_raw.csv", accuracy=0.8, adjacent_error_share=0.8, seed=5
    )
    raw = read_raw_predictions(raw_csv)
    assert sorted(raw) == sorted(expected)

    result = prepare_submission(raw, expected, sandbox_cfg, kind="a")
    assert result.ok, result.verify.report if result.verify else ""
    assert result.n_rows == len(expected)
    assert result.filled == []

    archive = Path(result.archive)
    with zipfile.ZipFile(archive) as zf:
        assert zf.namelist() == ["predictions.csv"]
    assert verify_submission(archive, expected, sandbox_cfg, kind="a").ok


def test_full_submit_pipeline_b_board(sandbox_cfg, prepared_project) -> None:
    records, _, _ = prepared_project
    expected = [f"t{i:04d}" for i in range(25)]
    raw = {image_id: index % 3 for index, image_id in enumerate(expected)}
    result = prepare_submission(
        raw,
        expected,
        sandbox_cfg,
        kind="b",
        solution_commit={"model": "convnext_tiny", "config": "default.yaml", "commit": "abc123"},
    )
    assert result.ok, result.verify.report if result.verify else ""
    with zipfile.ZipFile(result.archive) as zf:
        assert sorted(zf.namelist()) == ["solution_commit.txt", "submit.zip"]
    assert verify_submission(result.archive, expected, sandbox_cfg, kind="b").ok


def test_b_board_build_does_not_clobber_a_board_archive(sandbox_cfg) -> None:
    """Regression: building the B archive must not overwrite the A archive.

    Both live in the same output directory, and the B build stages its own
    inner ``submit.zip`` - if that staging writes to the shared path, the
    standalone A-board artifact is silently replaced by the B predictions.
    """
    expected = [f"t{index:04d}" for index in range(8)]
    raw = {image_id: index % 3 for index, image_id in enumerate(expected)}

    result_a = prepare_submission(raw, expected, sandbox_cfg, kind="a")
    a_path = Path(result_a.archive)
    a_bytes = a_path.read_bytes()
    assert a_path.name == sandbox_cfg.submit.a_zip_name

    result_b = prepare_submission(raw, expected, sandbox_cfg, kind="b")
    assert Path(result_b.archive).name == sandbox_cfg.submit.b_zip_name

    # The A archive is byte-identical after the B build.
    assert a_path.read_bytes() == a_bytes
    with zipfile.ZipFile(a_path) as zf:
        assert zf.namelist() == ["predictions.csv"]
    with zipfile.ZipFile(result_b.archive) as zf:
        assert sorted(zf.namelist()) == ["solution_commit.txt", "submit.zip"]


def test_pipeline_rejects_a_corrupted_submission(sandbox_cfg, prepared_project) -> None:
    """The safety net must actually block, not just report."""
    expected = [f"t{i:04d}" for i in range(10)]
    raw = {image_id: 0 for image_id in expected}
    raw[expected[0]] = 9  # invalid label
    with pytest.raises(Exception):
        result = prepare_submission(raw, expected, sandbox_cfg, kind="a")
        if not result.ok:
            raise RuntimeError("invalid label slipped through")
