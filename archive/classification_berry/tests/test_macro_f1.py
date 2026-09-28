"""Macro-F1 must equal scikit-learn exactly, and the reference must agree.

Acceptance criterion #1: "Macro-F1 与 sklearn 一致性". The HardLane lesson was
that a hand-rolled metric silently drifts from the official one; these tests
make that drift impossible to merge.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score

from eval.macro_f1 import (
    DEFAULT_LABELS,
    accuracy,
    confusion_matrix,
    evaluate,
    macro_f1,
    macro_f1_reference,
    mean_absolute_error,
    per_class_report,
)


def _random_pair(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    y_true = rng.integers(0, 3, size=n)
    # Bias predictions toward truth so the score is not trivially ~0.33.
    flip = rng.random(n) < 0.4
    y_pred = y_true.copy()
    y_pred[flip] = rng.integers(0, 3, size=int(flip.sum()))
    return y_true, y_pred


@pytest.mark.parametrize("seed", range(12))
def test_macro_f1_matches_sklearn_exactly(seed: int) -> None:
    rng = np.random.default_rng(seed)
    y_true, y_pred = _random_pair(rng, 500)
    expected = f1_score(
        y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0
    )
    assert macro_f1(y_true, y_pred) == pytest.approx(expected, abs=1e-15)


@pytest.mark.parametrize("seed", range(12))
def test_reference_matches_sklearn_exactly(seed: int) -> None:
    """The pure-numpy oracle is what catches an accidental metric change."""
    rng = np.random.default_rng(seed + 100)
    y_true, y_pred = _random_pair(rng, 500)
    expected = f1_score(
        y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0
    )
    assert macro_f1_reference(y_true, y_pred) == pytest.approx(expected, abs=1e-15)


def test_perfect_predictions_score_one() -> None:
    y = np.array([0, 1, 2, 0, 1, 2])
    assert macro_f1(y, y) == pytest.approx(1.0)
    assert macro_f1_reference(y, y) == pytest.approx(1.0)


def test_absent_class_uses_zero_division() -> None:
    """A class with no predictions and no truths contributes 0, not NaN."""
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1])
    expected = f1_score(
        y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0
    )
    assert not np.isnan(macro_f1(y_true, y_pred))
    assert macro_f1(y_true, y_pred) == pytest.approx(expected, abs=1e-15)
    assert macro_f1_reference(y_true, y_pred) == pytest.approx(expected, abs=1e-15)


def test_all_predictions_wrong_scores_zero() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([2, 2, 2, 2])
    assert macro_f1(y_true, y_pred) == pytest.approx(0.0)


def test_ordinal_asymmetry_is_visible() -> None:
    """0<->1 and 1<->2 errors are the real failure mode; MAE tracks severity."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    adjacent = np.array([1, 0, 1, 1, 2, 1])
    distant = np.array([2, 2, 0, 2, 0, 0])
    assert mean_absolute_error(y_true, adjacent) < mean_absolute_error(y_true, distant)


def test_confusion_matrix_layout_and_totals() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 1])
    matrix = confusion_matrix(y_true, y_pred)
    assert matrix.shape == (3, 3)
    assert matrix.sum() == len(y_true)
    # rows = truth, columns = prediction
    assert matrix[0, 0] == 1 and matrix[0, 1] == 1
    assert matrix[1, 0] == 0 and matrix[1, 1] == 2
    assert matrix[2, 0] == 0 and matrix[2, 1] == 1 and matrix[2, 2] == 1
    # column sums equal predicted counts
    assert matrix.sum(axis=0).tolist() == [1, 4, 1]


def test_confusion_matrix_rejects_undeclared_labels() -> None:
    with pytest.raises(ValueError, match="outside declared set"):
        confusion_matrix(np.array([0, 3]), np.array([0, 0]))


def test_per_class_report_matches_sklearn_per_class() -> None:
    rng = np.random.default_rng(7)
    y_true, y_pred = _random_pair(rng, 300)
    report = per_class_report(y_true, y_pred)
    expected = f1_score(
        y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0
    )
    for index, label in enumerate(DEFAULT_LABELS):
        assert report[label]["f1"] == pytest.approx(expected[index], abs=1e-15)


def test_evaluate_bundle_is_consistent() -> None:
    rng = np.random.default_rng(11)
    y_true, y_pred = _random_pair(rng, 200)
    report = evaluate(y_true, y_pred)
    assert report.n == 200
    assert report.macro_f1 == pytest.approx(macro_f1(y_true, y_pred), abs=1e-15)
    assert report.accuracy == pytest.approx(accuracy(y_true, y_pred), abs=1e-15)
    payload = report.to_dict()
    assert payload["confusion"][0][0] == int(report.confusion[0, 0])
    assert set(payload["per_class"]) == {"0", "1", "2"}


def test_length_mismatch_and_empty_are_rejected() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        macro_f1([0, 1], [0])
    with pytest.raises(ValueError, match="empty input"):
        macro_f1([], [])
