#!/usr/bin/env python
"""Build the OOF report from a training run's ``oof_predictions.json``.

Reporting is deliberately separated from training: a report can be regenerated
(or re-scored after a metric fix) without touching the GPU or the checkpoints.

Usage
-----
    python scripts/make_oof_report.py --oof outputs/train/oof_predictions.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from common.config import load_config  # noqa: E402
from eval.oof_report import build_oof_report, write_oof_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, default=None)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "outputs" / "reports" / "oof_report"
    )
    args = parser.parse_args()

    cfg = load_config()
    oof_path = args.oof or (cfg.path("train.output_dir") / "oof_predictions.json")
    if not oof_path.is_file():
        raise SystemExit(
            f"{oof_path} not found - train a model first (scripts/train_baseline.py)"
        )

    payload = json.loads(oof_path.read_text(encoding="utf-8"))
    records = payload["records"]
    if "fold_indices" not in payload:
        raise SystemExit(
            "oof_predictions.json has no 'fold_indices'. Re-run "
            "scripts/train_baseline.py (newer versions record them)."
        )

    labels = [int(record["label"]) for record in records]
    fold_items = sorted(payload["fold_indices"].items(), key=lambda kv: int(kv[0]))
    fold_indices = [list(indices) for _, indices in fold_items]
    oof = {int(k): int(v) for k, v in payload["oof"].items()}
    if len(oof) != len(records):
        raise SystemExit(
            f"OOF covers {len(oof)} of {len(records)} records - "
            "every fold must be trained for a valid OOF score."
        )
    fold_predictions = [[oof[i] for i in indices] for indices in fold_indices]

    report = build_oof_report(
        labels, fold_indices, fold_predictions, labels_set=list(cfg.eval.labels)
    )
    md_path = write_oof_report(report, args.out.with_suffix(".md"))
    json_path = write_oof_report(report, args.out.with_suffix(".json"))

    print(f"backbone          : {payload.get('backbone', 'unknown')}")
    print(f"OOF Macro-F1      : {report.overall.macro_f1:.5f}")
    print(f"OOF accuracy      : {report.overall.accuracy:.5f}")
    print(f"OOF MAE (ordinal) : {report.overall.mae:.5f}")
    print(f"adjacent-error share: {report.adjacent['adjacent_share']:.3f}")
    print(f"written: {md_path.relative_to(REPO_ROOT)}, {json_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
