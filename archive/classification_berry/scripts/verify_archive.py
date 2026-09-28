#!/usr/bin/env python
"""Standalone verifier for a submission archive (A or B board).

Thin wrapper so the safety net can be run without setting ``PYTHONPATH``:

    python scripts/verify_archive.py outputs/submissions/submit.zip \
        --expected data/raw/testA/sample_submission.csv --kind a

Exits 0 on PASS, 1 on FAIL, so it drops straight into CI or a shell ``&&`` chain.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from common.config import load_config  # noqa: E402
from submit.verify_submit import verify_submission  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--expected", required=True, type=Path,
                        help="sample_submission.csv or a one-id-per-line file")
    parser.add_argument("--kind", choices=["a", "b"], default="a")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    result = verify_submission(
        args.archive, args.expected, load_config(),
        kind=args.kind, report_path=args.report,
    )
    print(result.report)
    print(json.dumps({"ok": result.ok, "errors": len(result.errors), "rows": result.n_rows}))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
