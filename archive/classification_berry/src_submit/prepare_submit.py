"""Stage 1 of the submission safety net: canonicalize raw predictions.

Ported from HardLane's ``src/submit/prepare_submit.py``. That repo rewrote a
tree of ``.lines.txt`` files at fixed decimal precision; the analogue here is
rewriting arbitrary model output into the single canonical ``predictions.csv``:
UTF-8, LF, header ``image_id,label``, every test id exactly once, labels drawn
from ``{0,1,2}``.

The ordering rule matters for reproducibility: rows are emitted in the order of
the authoritative expected-id list, not in whatever order the model happened to
produce. Two runs of the same model therefore yield byte-identical CSVs, and a
diff of two submissions shows only real prediction changes.

This module never trains or loads a model. It consumes a CSV/JSON of
``image_id -> label`` so the whole stage runs offline on a CPU-only machine.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Union

from common.config import Config, load_config
from common.io_utils import ensure_dir
from submit.pack_submit import pack_a_submission, pack_b_submission, write_solution_commit
from submit.verify_submit import VerifyResult, read_expected_ids, verify_submission

PathLike = Union[str, Path]


class PrepareError(ValueError):
    """Raised when raw predictions cannot be canonicalized into a valid CSV."""


@dataclass
class PrepareResult:
    ok: bool
    predictions_csv: str
    archive: str
    kind: str
    n_rows: int
    filled: list[str] = field(default_factory=list)
    verify: VerifyResult | None = None

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "predictions_csv": self.predictions_csv,
            "archive": self.archive,
            "kind": self.kind,
            "n_rows": self.n_rows,
            "n_filled": len(self.filled),
            "filled_examples": self.filled[:5],
            "verify_errors": [] if self.verify is None else self.verify.errors,
        }


def read_raw_predictions(source: PathLike) -> dict[str, int]:
    """Read ``image_id,label[,score...]`` into a dict.

    Extra columns (e.g. a confidence score) are ignored, so an inference script
    can dump its full frame without a separate export step.
    """
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"raw predictions not found: {path}")

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return {str(k): int(v) for k, v in payload.items()}
        return {str(row["image_id"]): int(row["label"]) for row in payload}

    text = path.read_text(encoding="utf-8-sig", errors="strict")
    reader = csv.reader(io.StringIO(text))
    rows = [row for row in reader if row and any(cell.strip() for cell in row)]
    if not rows:
        raise PrepareError(f"{path} is empty")
    header = [cell.strip().lstrip("\ufeff") for cell in rows[0]]
    if "image_id" not in header or "label" not in header:
        raise PrepareError(
            f"{path}: header must contain 'image_id' and 'label', got {header}"
        )
    id_col, label_col = header.index("image_id"), header.index("label")

    out: dict[str, int] = {}
    for line_number, row in enumerate(rows[1:], start=2):
        if len(row) <= max(id_col, label_col):
            raise PrepareError(f"{path}:L{line_number}: too few fields: {row}")
        image_id = row[id_col].strip()
        if not image_id:
            raise PrepareError(f"{path}:L{line_number}: empty image_id")
        try:
            label = int(row[label_col].strip())
        except ValueError as exc:
            raise PrepareError(
                f"{path}:L{line_number}: label {row[label_col]!r} is not an integer"
            ) from exc
        if image_id in out:
            raise PrepareError(f"{path}:L{line_number}: duplicate image_id {image_id!r}")
        out[image_id] = label
    return out


def canonicalize_predictions(
    raw: Mapping[str, int],
    expected_ids: Sequence[str],
    cfg: Config,
    *,
    fill_missing: int | None = None,
) -> tuple[list[tuple[str, int]], list[str]]:
    """Order ``raw`` by ``expected_ids`` and enforce the exact-id-set contract.

    Returns ``(rows, filled)`` where ``filled`` lists ids that had no prediction
    and were substituted with ``fill_missing``. Extra ids (not in the test set)
    are always an error: silently dropping them would hide an inference bug.
    """
    valid_labels = {int(value) for value in cfg.submit.valid_labels}
    expected_set = set(expected_ids)
    if len(expected_set) != len(expected_ids):
        raise PrepareError("expected_ids contains duplicates")

    extra = sorted(set(raw) - expected_set)
    if extra:
        raise PrepareError(
            f"{len(extra)} prediction id(s) are not in the test set: {extra[:5]}"
        )

    for image_id, label in raw.items():
        if not isinstance(label, int) or label not in valid_labels:
            raise PrepareError(
                f"{image_id}: label {label!r} outside {sorted(valid_labels)}"
            )

    rows: list[tuple[str, int]] = []
    filled: list[str] = []
    for image_id in expected_ids:
        if image_id in raw:
            rows.append((image_id, int(raw[image_id])))
        elif fill_missing is not None:
            rows.append((image_id, int(fill_missing)))
            filled.append(image_id)
        else:
            raise PrepareError(
                f"test id {image_id!r} has no prediction and fill_missing is not set "
                "(every test id must appear exactly once)"
            )
    return rows, filled


def write_predictions_csv(rows: Iterable[tuple[str, int]], path: PathLike, cfg: Config) -> Path:
    """Write the canonical CSV: UTF-8, no BOM, LF, header ``image_id,label``."""
    path = Path(path)
    ensure_dir(path.parent)
    header = [str(value) for value in cfg.submit.predictions_header]
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for image_id, label in rows:
        writer.writerow([image_id, int(label)])
    path.write_bytes(buffer.getvalue().encode("utf-8"))
    return path


def prepare_submission(
    raw_predictions: PathLike | Mapping[str, int],
    expected_ids: Sequence[str] | PathLike,
    cfg: Config | None = None,
    *,
    kind: str = "a",
    out_dir: PathLike | None = None,
    archive_name: str | None = None,
    fill_missing: int | None = None,
    solution_commit: Mapping[str, str] | None = None,
    report_path: PathLike | None = None,
) -> PrepareResult:
    """Run ``canonicalize -> write csv -> pack -> verify`` end to end.

    ``kind='a'`` produces ``submit.zip``; ``kind='b'`` produces
    ``b_submission.zip`` wrapping a freshly built ``submit.zip`` plus
    ``solution_commit.txt`` (fields from ``solution_commit``).
    """
    cfg = cfg or load_config()
    expected_ids = (
        read_expected_ids(expected_ids)
        if isinstance(expected_ids, (str, Path))
        else list(expected_ids)
    )
    raw = (
        read_raw_predictions(raw_predictions)
        if isinstance(raw_predictions, (str, Path))
        else {str(k): int(v) for k, v in raw_predictions.items()}
    )

    rows, filled = canonicalize_predictions(raw, expected_ids, cfg, fill_missing=fill_missing)

    out_dir = Path(out_dir) if out_dir else cfg.path("submit.output_dir")
    ensure_dir(out_dir)
    csv_path = write_predictions_csv(rows, out_dir / str(cfg.submit.predictions_file), cfg)

    if kind == "a":
        archive = pack_a_submission(csv_path, out_dir / str(cfg.submit.a_zip_name), cfg)
    else:
        # Stage the inner A archive under a dot-prefixed name and delete it in a
        # finally block: building the B archive must not clobber a standalone
        # submit.zip already in out_dir, and explicit cleanup is more reliable
        # than TemporaryDirectory on sandboxed / brokered filesystems.
        inner_path = out_dir / f".staging_{cfg.submit.a_zip_name}"
        try:
            inner = pack_a_submission(csv_path, inner_path, cfg)
            commit_path = write_solution_commit(
                out_dir / "solution_commit.txt",
                model=(solution_commit or {}).get("model", "unspecified"),
                config=(solution_commit or {}).get("config", str(cfg.project.name)),
                commit=(solution_commit or {}).get("commit", "uncommitted"),
                extra={
                    k: v
                    for k, v in (solution_commit or {}).items()
                    if k not in {"model", "config", "commit"}
                },
            )
            archive = pack_b_submission(
                inner, commit_path, out_dir / str(cfg.submit.b_zip_name), cfg
            )
        finally:
            inner_path.unlink(missing_ok=True)

    if archive_name:
        archive = Path(archive).rename(Path(out_dir) / archive_name)

    verify = verify_submission(archive, expected_ids, cfg, kind=kind)
    if report_path is not None:
        verify.write_report(report_path)

    return PrepareResult(
        ok=verify.ok,
        predictions_csv=str(csv_path),
        archive=str(archive),
        kind=kind,
        n_rows=len(rows),
        filled=filled,
        verify=verify,
    )


def main() -> None:  # pragma: no cover - CLI glue
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path,
                        help="raw predictions: image_id,label[,score...] csv or json")
    parser.add_argument("--expected", required=True, type=Path,
                        help="sample_submission.csv or one-id-per-line file")
    parser.add_argument("--kind", choices=["a", "b"], default="a")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--fill-missing", type=int, default=None)
    parser.add_argument("--model", default="unspecified")
    parser.add_argument("--commit", default="uncommitted")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    result = prepare_submission(
        args.raw, args.expected, load_config(),
        kind=args.kind, out_dir=args.out_dir, fill_missing=args.fill_missing,
        solution_commit={"model": args.model, "commit": args.commit},
        report_path=args.report,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    if result.verify is not None:
        print(result.verify.report)
    raise SystemExit(0 if result.ok else 1)


if __name__ == "__main__":  # pragma: no cover
    main()
