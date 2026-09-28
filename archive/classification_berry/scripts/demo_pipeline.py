#!/usr/bin/env python
"""P0 end-to-end demo: synthetic data -> manifest -> split -> OOF -> submission.

Runs the entire engineering skeleton offline on a CPU-only machine, producing
the artifacts that acceptance criteria #2 and #3 ask for:

* ``outputs/reports/oof_report.md`` / ``.json`` - grouped-OOF Macro-F1,
  confusion matrix, adjacent-class error anatomy;
* ``outputs/submissions/submit.zip`` and ``b_submission.zip`` - real,
  contract-valid archives produced by ``prepare -> pack -> verify``.

⚠️ The data is SYNTHETIC. The numbers below are a pipeline self-test, not
competition results. Re-run with the real dataset (see README) to get real ones.

Usage
-----
    python scripts/demo_pipeline.py
    python scripts/demo_pipeline.py --skip-b-board
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from common.config import Config, load_config  # noqa: E402
from data.make_sample_data import generate  # noqa: E402
from data.manifest import read_labels, source_grouping_report, write_manifest  # noqa: E402
from data.split_by_source import load_folds, run_split  # noqa: E402
from eval.macro_f1 import macro_f1  # noqa: E402
from eval.oof_report import build_oof_report, write_oof_report  # noqa: E402
from submit.prepare_submit import prepare_submission  # noqa: E402
from submit.verify_submit import read_expected_ids  # noqa: E402


def sandbox_config(base: Config, sample_dir: Path, work_dir: Path) -> Config:
    """Point every config path at the synthetic dataset and a scratch work dir."""
    cfg = Config(copy.deepcopy(dict(base)))
    cfg["data"]["labels_csv"] = str(sample_dir / "raw" / "labels.csv")
    cfg["data"]["image_dir"] = str(sample_dir / "raw" / "images")
    cfg["data"]["manifest_path"] = str(work_dir / "manifest.csv")
    cfg["data"]["processed_dir"] = str(work_dir / "processed")
    cfg["split"]["output_dir"] = str(work_dir / "splits")
    cfg["submit"]["output_dir"] = str(REPO_ROOT / "outputs" / "submissions")
    cfg["submit"]["report_dir"] = str(REPO_ROOT / "outputs" / "reports")
    return cfg


def simulate_fold_predictions(
    records,
    fold_indices,
    *,
    accuracy: float,
    adjacent_share: float,
    seed: int,
) -> list[list[int]]:
    """A stand-in "model": mostly right, and wrong mostly by one ordinal step."""
    rng = random.Random(seed)
    out: list[list[int]] = []
    for indices in fold_indices:
        fold_preds = []
        for index in indices:
            truth = records[index].label
            if rng.random() < accuracy:
                fold_preds.append(truth)
                continue
            if rng.random() < adjacent_share:
                step = -1 if truth == 2 else (1 if truth == 0 else rng.choice([-1, 1]))
            else:
                step = rng.choice([-2, -1, 1, 2])
            fold_preds.append(min(2, max(0, truth + step)))
        out.append(fold_preds)
    return out


def source_memoriser_oof(records, fold_indices) -> float:
    """Macro-F1 of a predictor that memorises "this source image = this class".

    This emulates the strongest thing a *leaked* split teaches. Under the
    grouped split a validation source is unseen, so the predictor must fall
    back to the global majority and scores poorly; under a random per-image
    split the same predictor has seen the source in training and scores well.
    The gap is the inflation a non-grouped local split would buy you.
    """
    global_majority = Counter(r.label for r in records).most_common(1)[0][0]
    y_true: list[int] = []
    y_pred: list[int] = []
    for indices in fold_indices:
        val_set = set(indices)
        per_source: dict[str, Counter] = defaultdict(Counter)
        for index, record in enumerate(records):
            if index not in val_set:
                per_source[record.source_id][record.label] += 1
        for index in indices:
            counts = per_source.get(records[index].source_id)
            y_true.append(records[index].label)
            y_pred.append(counts.most_common(1)[0][0] if counts else global_majority)
    return macro_f1(y_true, y_pred)


def random_fold_indices(records, *, n_folds: int, seed: int) -> list[list[int]]:
    """The anti-pattern: per-image random split that ignores source grouping."""
    rng = random.Random(seed)
    assignment = [rng.randrange(n_folds) for _ in records]
    return [
        [i for i, a in enumerate(assignment) if a == fold_id] for fold_id in range(n_folds)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, default=REPO_ROOT / "data" / "sample")
    parser.add_argument("--work-dir", type=Path, default=REPO_ROOT / "outputs" / "demo")
    parser.add_argument("--skip-b-board", action="store_true")
    args = parser.parse_args()

    base = load_config(REPO_ROOT / "configs" / "default.yaml")
    sample_dir = args.sample_dir
    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("P0 demo - SYNTHETIC data (pipeline self-test, NOT competition results)")
    print("=" * 72)

    # ---- 1. synthetic data -------------------------------------------------
    summary = generate(
        sample_dir,
        n_sources=int(base.competition.n_train_sources),
        n_test_a=int(base.competition.n_test_a),
        n_test_b=int(base.competition.n_test_b),
        seed=int(base.project.seed),
        with_images=False,
        image_size=32,
    )
    print(f"[1/6] synthetic dataset: {summary['n_train_images']} crops "
          f"from {summary['n_sources']} source images")

    cfg = sandbox_config(base, sample_dir, work_dir)

    # ---- 2. manifest -------------------------------------------------------
    records = read_labels(cfg)
    manifest_path = Path(cfg.data.manifest_path)
    write_manifest(manifest_path, records)
    grouping = source_grouping_report(records)
    print(f"[2/6] manifest: {len(records)} records, {grouping['n_sources']} sources "
          f"(mean {grouping['images_per_source_mean']} crops/source)")
    if grouping["grouping_looks_broken"]:
        print("      ⚠️  grouping looks broken (one group per image)")

    # ---- 3. grouped split --------------------------------------------------
    result = run_split(cfg, manifest_path=manifest_path)
    folds_path = Path(cfg.split.output_dir) / "folds.json"
    reloaded, records = load_folds(folds_path)
    sizes = [len(fold.val_indices) for fold in reloaded.folds]
    print(f"[3/6] grouped {reloaded.n_folds}-fold split: val sizes {sizes}; "
          f"leakage check PASSED (no source spans folds)")

    # ---- 4. OOF report -----------------------------------------------------
    fold_indices = [list(fold.val_indices) for fold in reloaded.folds]
    fold_predictions = simulate_fold_predictions(
        records, fold_indices, accuracy=0.78, adjacent_share=0.85, seed=7
    )
    report = build_oof_report(
        [r.label for r in records], fold_indices, fold_predictions,
        labels_set=list(base.eval.labels),
    )
    oof_md = write_oof_report(
        report, REPO_ROOT / "outputs" / "reports" / "oof_report.md", synthetic=True
    )
    write_oof_report(
        report, REPO_ROOT / "outputs" / "reports" / "oof_report.json", synthetic=True
    )
    print(f"[4/6] OOF Macro-F1 = {report.overall.macro_f1:.5f}  "
          f"(accuracy {report.overall.accuracy:.5f}, MAE {report.overall.mae:.5f})")
    print(f"      adjacent-error share = {report.adjacent['adjacent_share']:.3f} "
          f"-> {oof_md.relative_to(REPO_ROOT)}")

    leaky = source_memoriser_oof(
        records, random_fold_indices(records, n_folds=reloaded.n_folds, seed=7)
    )
    honest = source_memoriser_oof(records, fold_indices)
    print(f"      leakage demo: a source-memorising predictor scores {honest:.5f} under "
          f"the grouped split vs {leaky:.5f} under a random per-image split "
          f"(inflation {leaky - honest:+.5f})")

    # ---- 5. A-board submission --------------------------------------------
    test_a = read_expected_ids(sample_dir / "testA" / "sample_submission.csv")
    rng = random.Random(11)
    raw_a = {
        image_id: min(2, max(0, rng.choice([0, 1, 2])))
        for image_id in test_a
    }
    result_a = prepare_submission(raw_a, test_a, cfg, kind="a")
    print(f"[5/6] A-board: {Path(result_a.archive).name} "
          f"({result_a.n_rows} rows) verify={'PASS' if result_a.ok else 'FAIL'}")
    if not result_a.ok and result_a.verify:
        print(result_a.verify.report)

    # ---- 6. B-board submission --------------------------------------------
    if not args.skip_b_board:
        test_b = read_expected_ids(sample_dir / "testB" / "sample_submission.csv")
        raw_b = {image_id: min(2, max(0, rng.choice([0, 1, 2]))) for image_id in test_b}
        result_b = prepare_submission(
            raw_b,
            test_b,
            cfg,
            kind="b",
            solution_commit={
                "model": "SYNTHETIC-demo (no model trained)",
                "config": "configs/default.yaml",
                "commit": "demo",
                "data": "synthetic - not competition data",
            },
        )
        print(f"[6/6] B-board: {Path(result_b.archive).name} "
              f"({result_b.n_rows} rows) verify={'PASS' if result_b.ok else 'FAIL'}")
        if not result_b.ok and result_b.verify:
            print(result_b.verify.report)
    else:
        print("[6/6] B-board skipped")

    print("-" * 72)
    print("P0 chain complete. Artifacts:")
    print(f"  OOF report   : outputs/reports/oof_report.md")
    print(f"  A archive    : outputs/submissions/{base.submit.a_zip_name}")
    if not args.skip_b_board:
        print(f"  B archive    : outputs/submissions/{base.submit.b_zip_name}")
    print("NOTE: numbers are from SYNTHETIC data - replace with the real dataset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
