"""Official Macro-F1 scoring, frozen to the competition definition.

The leaderboard metric is defined as::

    f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)

This module provides three things:

1. :func:`macro_f1` - the authoritative implementation, a thin wrapper over
   scikit-learn so there is exactly one scoring code path.
2. :func:`macro_f1_reference` - an independent pure-numpy implementation used
   as a *differential test oracle*. The HardLane lesson (README "经验教训" #4)
   is that a custom ruler must be cross-checked against the official one, or
   the "custom ruler on custom data" self-justifying loop is unavoidable.
3. :func:`confusion_matrix` / :func:`per_class_report` - diagnostics for the
   adjacent-class confusions (0<->1, 1<->2) that dominate the error budget.

No model code may be imported here: the evaluation layer stays independent of
the training layer (HardLane ARCHITECTURE §3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import f1_score

DEFAULT_LABELS: tuple[int, ...] = (0, 1, 2)


def _as_int_array(values: Iterable[int], name: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.int64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {array.shape}")
    return array


def macro_f1(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    *,
    labels: Sequence[int] = DEFAULT_LABELS,
    zero_division: int = 0,
) -> float:
    """Authoritative Macro-F1 (scikit-learn, the official ranking metric)."""
    y_true = _as_int_array(y_true, "y_true")
    y_pred = _as_int_array(y_pred, "y_pred")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"length mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("empty input: Macro-F1 is undefined")
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=list(labels),
            average="macro",
            zero_division=zero_division,
        )
    )


def macro_f1_reference(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    *,
    labels: Sequence[int] = DEFAULT_LABELS,
    zero_division: int = 0,
) -> float:
    """Independent pure-numpy Macro-F1, used only to differentially test #1.

    For each label ``l``::

        f1_l = tp / (tp + 0.5 * (fp + fn))

    with ``f1_l = 0`` when the denominator is zero (``zero_division=0``).
    The macro score is the unweighted mean over ``labels``.
    """
    y_true = _as_int_array(y_true, "y_true")
    y_pred = _as_int_array(y_pred, "y_pred")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"length mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("empty input: Macro-F1 is undefined")

    scores: list[float] = []
    for label in labels:
        predicted = y_pred == label
        actual = y_true == label
        tp = int(np.count_nonzero(predicted & actual))
        fp = int(np.count_nonzero(predicted & ~actual))
        fn = int(np.count_nonzero(~predicted & actual))
        denominator = tp + 0.5 * (fp + fn)
        if denominator == 0:
            scores.append(float(zero_division))
        else:
            scores.append(tp / denominator)
    return float(np.mean(scores))


def accuracy(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    y_true = _as_int_array(y_true, "y_true")
    y_pred = _as_int_array(y_pred, "y_pred")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"length mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("empty input: accuracy is undefined")
    return float(np.mean(y_true == y_pred))


def mean_absolute_error(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    """Ordinal MAE - displayed only, but a useful ordinal sanity signal."""
    y_true = _as_int_array(y_true, "y_true")
    y_pred = _as_int_array(y_pred, "y_pred")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"length mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("empty input: MAE is undefined")
    return float(np.mean(np.abs(y_true - y_pred)))


def confusion_matrix(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    *,
    labels: Sequence[int] = DEFAULT_LABELS,
) -> np.ndarray:
    """Rows = truth, columns = prediction, ordered by ``labels``."""
    y_true = _as_int_array(y_true, "y_true")
    y_pred = _as_int_array(y_pred, "y_pred")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"length mismatch: {y_true.shape} vs {y_pred.shape}")
    index = {label: position for position, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for truth, prediction in zip(y_true.tolist(), y_pred.tolist()):
        if truth not in index or prediction not in index:
            raise ValueError(
                f"label outside declared set: y_true={truth} y_pred={prediction} "
                f"labels={list(labels)}"
            )
        matrix[index[truth], index[prediction]] += 1
    return matrix


@dataclass
class EvalReport:
    """Full diagnostic bundle for one prediction set."""

    macro_f1: float
    accuracy: float
    mae: float
    labels: tuple[int, ...]
    confusion: np.ndarray
    per_class: dict[int, dict[str, float]] = field(default_factory=dict)
    n: int = 0

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "macro_f1": self.macro_f1,
            "accuracy": self.accuracy,
            "mae": self.mae,
            "labels": list(self.labels),
            "confusion": self.confusion.tolist(),
            "per_class": {str(k): v for k, v in self.per_class.items()},
        }


def per_class_report(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    *,
    labels: Sequence[int] = DEFAULT_LABELS,
) -> dict[int, dict[str, float]]:
    """Precision / recall / F1 / support per class (zero_division=0)."""
    y_true = _as_int_array(y_true, "y_true")
    y_pred = _as_int_array(y_pred, "y_pred")
    report: dict[int, dict[str, float]] = {}
    for label in labels:
        predicted = y_pred == label
        actual = y_true == label
        tp = int(np.count_nonzero(predicted & actual))
        fp = int(np.count_nonzero(predicted & ~actual))
        fn = int(np.count_nonzero(~predicted & actual))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        denominator = tp + 0.5 * (fp + fn)
        report[int(label)] = {
            "precision": precision,
            "recall": recall,
            "f1": tp / denominator if denominator else 0.0,
            "support": float(tp + fn),
        }
    return report


def evaluate(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    *,
    labels: Sequence[int] = DEFAULT_LABELS,
    zero_division: int = 0,
) -> EvalReport:
    """One-call bundle of every metric we report for a model."""
    y_true = _as_int_array(y_true, "y_true")
    y_pred = _as_int_array(y_pred, "y_pred")
    return EvalReport(
        macro_f1=macro_f1(y_true, y_pred, labels=labels, zero_division=zero_division),
        accuracy=accuracy(y_true, y_pred),
        mae=mean_absolute_error(y_true, y_pred),
        labels=tuple(labels),
        confusion=confusion_matrix(y_true, y_pred, labels=labels),
        per_class=per_class_report(y_true, y_pred, labels=labels),
        n=int(y_true.size),
    )


__all__ = [
    "DEFAULT_LABELS",
    "macro_f1",
    "macro_f1_reference",
    "accuracy",
    "mean_absolute_error",
    "confusion_matrix",
    "per_class_report",
    "EvalReport",
    "evaluate",
]
