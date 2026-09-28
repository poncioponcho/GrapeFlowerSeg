"""Assemble out-of-fold predictions into the single trustworthy local number.

Out-of-fold (OOF) Macro-F1 on a *grouped* split is the only local evidence that
may be used to choose a model (HardLane lesson #1: an offline conclusion does
not transfer for free, but a leaky offline number is worthless outright).

This module also quantifies the failure mode that matters most here: the
adjacent-class confusions 0<->1 and 1<->2. The label is ordinal, so an error
that moves one step is far more likely - and far less costly - than a 0<->2
jump. Reporting them separately turns "Macro-F1 = 0.8" into an actionable
diagnosis.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from eval.macro_f1 import EvalReport, evaluate

PathLike = str | Path


@dataclass
class OofReport:
    """Grouped-OOF metrics plus the diagnostics that drive the next experiment."""

    overall: EvalReport
    per_fold: list[dict]
    adjacent: dict
    n_folds: int

    def to_dict(self) -> dict:
        return {
            "n_folds": self.n_folds,
            "overall": self.overall.to_dict(),
            "per_fold": self.per_fold,
            "adjacent_confusion": self.adjacent,
        }

    def to_markdown(self, *, synthetic: bool = False) -> str:
        banner = (
            "> ⚠️ **SYNTHETIC DATA** - these numbers come from generated data used\n"
            "> to self-test the pipeline. They are **not** competition results.\n\n"
            if synthetic
            else ""
        )
        overall = self.overall
        lines = [
            "# OOF report (grouped K-fold by source image)",
            "",
            banner.rstrip(),
            "",
            f"- folds: **{self.n_folds}**",
            f"- n: **{overall.n}**",
            f"- **Macro-F1 (OOF): `{overall.macro_f1:.5f}`**",
            f"- accuracy: {overall.accuracy:.5f}",
            f"- MAE (ordinal): {overall.mae:.5f}",
            "",
            "## Confusion matrix (rows = truth, cols = prediction)",
            "",
            "| truth \\ pred | 0 | 1 | 2 | support |",
            "|---|---|---|---|---|",
        ]
        for index, label in enumerate(overall.labels):
            row = overall.confusion[index]
            support = int(row.sum())
            lines.append(
                f"| **{label}** | {row[0]} | {row[1]} | {row[2]} | {support} |"
            )
        lines += [
            "",
            "## Adjacent-class confusion (the real error budget)",
            "",
            f"- 0 -> 1: {self.adjacent['0->1']}   |  1 -> 0: {self.adjacent['1->0']}",
            f"- 1 -> 2: {self.adjacent['1->2']}   |  2 -> 1: {self.adjacent['2->1']}",
            f"- distant (0 <-> 2): {self.adjacent['0<->2']}",
            f"- total errors: {self.adjacent['total_errors']} of {overall.n}",
            f"- **share of errors that are adjacent: "
            f"{self.adjacent['adjacent_share']:.3f}**",
            "",
            "## Per class",
            "",
            "| label | precision | recall | f1 | support |",
            "|---|---|---|---|---|",
        ]
        for label in overall.labels:
            metrics = overall.per_class[int(label)]
            lines.append(
                f"| {label} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                f"{metrics['f1']:.4f} | {int(metrics['support'])} |"
            )
        lines += ["", "## Per fold", "", "| fold | n | Macro-F1 | accuracy |", "|---|---|---|---|"]
        for entry in self.per_fold:
            lines.append(
                f"| {entry['fold']} | {entry['n']} | {entry['macro_f1']:.5f} | "
                f"{entry['accuracy']:.5f} |"
            )
        lines.append("")
        return "\n".join(lines)


def assemble_oof(
    labels: Sequence[int],
    fold_indices: Sequence[Sequence[int]],
    fold_predictions: Sequence[Sequence[int]],
) -> tuple[list[int], list[int]]:
    """Stitch per-fold predictions back into record order.

    ``fold_indices[k]`` are the validation indices of fold ``k`` and
    ``fold_predictions[k]`` the predictions for exactly those indices. Every
    record must be covered exactly once; anything else is a bug in the training
    loop and is reported as such.
    """
    if len(fold_indices) != len(fold_predictions):
        raise ValueError("fold_indices and fold_predictions must have equal length")
    y_true: list[int | None] = [None] * len(labels)
    y_pred: list[int | None] = [None] * len(labels)
    for indices, predictions in zip(fold_indices, fold_predictions):
        if len(indices) != len(predictions):
            raise ValueError(
                f"fold length mismatch: {len(indices)} indices vs {len(predictions)} predictions"
            )
        for index, prediction in zip(indices, predictions):
            if y_pred[index] is not None:
                raise ValueError(f"record {index} predicted more than once (OOF leak)")
            y_true[index] = labels[index]
            y_pred[index] = int(prediction)
    missing = [i for i, value in enumerate(y_pred) if value is None]
    if missing:
        raise ValueError(f"{len(missing)} records have no OOF prediction (e.g. {missing[:5]})")
    return [int(v) for v in y_true], [int(v) for v in y_pred]  # type: ignore[arg-type]


def adjacent_confusion_analysis(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    """Split errors into adjacent (0<->1, 1<->2) and distant (0<->2) buckets."""
    counts = Counter(
        (int(truth), int(prediction))
        for truth, prediction in zip(y_true, y_pred)
        if truth != prediction
    )
    adjacent = (
        counts[(0, 1)] + counts[(1, 0)] + counts[(1, 2)] + counts[(2, 1)]
    )
    distant = counts[(0, 2)] + counts[(2, 0)]
    total_errors = adjacent + distant
    return {
        "0->1": counts[(0, 1)],
        "1->0": counts[(1, 0)],
        "1->2": counts[(1, 2)],
        "2->1": counts[(2, 1)],
        "0<->2": distant,
        "total_errors": total_errors,
        "adjacent_share": (adjacent / total_errors) if total_errors else 0.0,
    }


def build_oof_report(
    labels: Sequence[int],
    fold_indices: Sequence[Sequence[int]],
    fold_predictions: Sequence[Sequence[int]],
    *,
    labels_set: Sequence[int] = (0, 1, 2),
) -> OofReport:
    """Full OOF report: overall metrics, per-fold breakdown, error anatomy."""
    y_true, y_pred = assemble_oof(labels, fold_indices, fold_predictions)
    overall = evaluate(y_true, y_pred, labels=labels_set)
    per_fold = []
    for fold_id, (indices, predictions) in enumerate(zip(fold_indices, fold_predictions)):
        if not indices:
            continue
        truth = [labels[i] for i in indices]
        fold_report = evaluate(truth, predictions, labels=labels_set)
        per_fold.append(
            {
                "fold": fold_id,
                "n": fold_report.n,
                "macro_f1": fold_report.macro_f1,
                "accuracy": fold_report.accuracy,
            }
        )
    return OofReport(
        overall=overall,
        per_fold=per_fold,
        adjacent=adjacent_confusion_analysis(y_true, y_pred),
        n_folds=len(fold_indices),
    )


def write_oof_report(report: OofReport, path: PathLike, *, synthetic: bool = False) -> Path:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    else:
        path.write_text(report.to_markdown(synthetic=synthetic), encoding="utf-8")
    return path


__all__ = [
    "OofReport",
    "assemble_oof",
    "adjacent_confusion_analysis",
    "build_oof_report",
    "write_oof_report",
]
