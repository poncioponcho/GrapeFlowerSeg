"""Stage 3 of the submission safety net: verify a candidate archive offline.

Ported from the HardLane repo's ``src/submit/verify_submit.py`` (three-stage
``prepare -> pack -> verify`` pattern), retargeted from ``.lines.txt`` trees to
a single ``predictions.csv``.

The competition contract this gate enforces (see ``configs/default.yaml``):

* A-board ``submit.zip``: the archive root contains **exactly one** file,
  ``predictions.csv``. No wrapper directory, no extra files.
* B-board ``b_submission.zip``: the root contains exactly ``submit.zip`` and
  ``solution_commit.txt``; the embedded ``submit.zip`` must independently pass
  the A-board checks.

``predictions.csv`` must be UTF-8 with header ``image_id,label``; every test
``image_id`` appears exactly once; ``label`` is an integer in ``{0,1,2}``.
A missing, duplicated, or extra id invalidates the submission outright.

Five interception classes are covered explicitly and each has a dedicated unit
test in ``tests/test_submit_verify.py``:

1. not valid UTF-8
2. missing header row / wrong column names
3. ``image_id`` missing, duplicated, or not in the test set
4. ``label`` not an integer, or outside ``{0,1,2}``
5. archive root is not a flat, exact file set (multi-level directory / extra file)

Runs with the standard library + PyYAML only: no GPU, no torch, no network.
"""
from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence, Union

from common.config import Config, load_config

PathLike = Union[str, Path]
MAX_REPORTED_ERRORS = 25


@dataclass
class VerifyResult:
    """Outcome of one verification run."""

    ok: bool
    errors: list[str] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)
    n_rows: int = 0
    archive: str = ""
    kind: str = "a"

    @property
    def report(self) -> str:
        head = f"verify {'PASS' if self.ok else 'FAIL'}  {self.archive}  (kind={self.kind})"
        lines = [head, f"  rows: {self.n_rows}", f"  checks: {len(self.checks_run)}"]
        if self.errors:
            lines.append(f"  errors: {len(self.errors)}")
            lines.extend(f"    - {err}" for err in self.errors[:MAX_REPORTED_ERRORS])
            if len(self.errors) > MAX_REPORTED_ERRORS:
                lines.append(f"    ... and {len(self.errors) - MAX_REPORTED_ERRORS} more")
        else:
            lines.append("  all checks passed")
        return "\n".join(lines)

    def write_report(self, path: PathLike) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.report + "\n", encoding="utf-8")
        return path


def read_expected_ids(source: PathLike | Sequence[str]) -> list[str]:
    """Load the authoritative test ``image_id`` list.

    Accepts a ``sample_submission.csv`` (header ``image_id,label`` - only the id
    column is used), a plain one-id-per-line text file, or an in-memory
    sequence. Order is preserved; duplicates raise.
    """
    if not isinstance(source, (str, Path)):
        ids = [str(value).strip() for value in source]
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"expected-id source not found: {path}")
        raw = path.read_text(encoding="utf-8-sig", errors="replace")
        if path.suffix.lower() == ".csv":
            reader = csv.reader(io.StringIO(raw))
            rows = [row for row in reader if row and any(cell.strip() for cell in row)]
            if not rows:
                raise ValueError(f"{path} is empty")
            header = [cell.strip().lstrip("\ufeff") for cell in rows[0]]
            if "image_id" not in header:
                raise ValueError(
                    f"{path}: expected a header containing 'image_id', got {header}"
                )
            column = header.index("image_id")
            ids = [row[column].strip() for row in rows[1:] if len(row) > column]
        else:
            ids = [line.strip() for line in raw.splitlines() if line.strip()]

    if not ids:
        raise ValueError("expected-id list is empty")
    if len(ids) != len(set(ids)):
        raise ValueError("expected-id list itself contains duplicates")
    return ids


def verify_predictions_bytes(data: bytes, expected_ids: Sequence[str], cfg: Config) -> VerifyResult:
    """Validate ``predictions.csv`` content against the official contract."""
    errors: list[str] = []
    checks: list[str] = []
    submit_cfg = cfg.submit
    header_wanted = [str(value) for value in submit_cfg.predictions_header]
    valid_labels = {int(value) for value in submit_cfg.valid_labels}

    # ---- check 1: UTF-8 -------------------------------------------------
    checks.append("utf8")
    if data.startswith(b"\xef\xbb\xbf"):
        errors.append("file starts with a UTF-8 BOM (expected plain UTF-8, no BOM)")
        data = data[3:]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        errors.append(f"not valid UTF-8: {exc}")
        return VerifyResult(False, errors, checks, 0, "", "csv")

    if not text.strip():
        errors.append("file is empty")
        return VerifyResult(False, errors, checks, 0, "", "csv")

    lines = text.splitlines()
    # ---- check 2: header ------------------------------------------------
    checks.append("header")
    header = [cell.strip() for cell in lines[0].split(",")]
    if header != header_wanted:
        errors.append(
            f"header mismatch: expected {header_wanted}, got {header} "
            "(first line must be exactly 'image_id,label')"
        )
        return VerifyResult(False, errors, checks, 0, "", "csv")

    # ---- checks 3 & 4: rows --------------------------------------------
    checks.extend(["image_id_set", "image_id_unique", "label_domain", "row_shape"])
    expected_set = set(expected_ids)
    seen: set[str] = set()
    duplicates: list[str] = []
    unknown: list[str] = []
    bad_labels: list[str] = []
    bad_shapes: list[str] = []
    n_rows = 0

    reader = csv.reader(lines[1:])
    for row_number, row in enumerate(reader, start=2):
        if not row or not any(cell.strip() for cell in row):
            continue
        n_rows += 1
        if len(row) != len(header_wanted):
            bad_shapes.append(f"L{row_number}: expected {len(header_wanted)} fields, got {len(row)}")
            continue
        image_id = row[0].strip()
        raw_label = row[1].strip()

        if not image_id:
            bad_shapes.append(f"L{row_number}: empty image_id")
            continue
        if image_id not in expected_set:
            unknown.append(f"L{row_number}: {image_id!r}")
        elif image_id in seen:
            duplicates.append(f"L{row_number}: {image_id!r}")
        else:
            seen.add(image_id)

        try:
            label = int(raw_label)
        except ValueError:
            bad_labels.append(f"L{row_number}: {raw_label!r} is not an integer")
            continue
        if label not in valid_labels:
            bad_labels.append(f"L{row_number}: {label} outside {sorted(valid_labels)}")

    missing = sorted(expected_set - seen)

    if bad_shapes:
        errors.append(f"{len(bad_shapes)} malformed row(s): {bad_shapes[:3]}")
    if unknown:
        errors.append(
            f"{len(unknown)} image_id not in the test set: {unknown[:3]}"
        )
    if duplicates:
        errors.append(f"{len(duplicates)} duplicated image_id: {duplicates[:3]}")
    if bad_labels:
        errors.append(f"{len(bad_labels)} invalid label(s): {bad_labels[:3]}")
    if missing:
        errors.append(
            f"{len(missing)} test image_id missing from predictions: {missing[:3]}"
        )

    return VerifyResult(not errors, errors, checks, n_rows, "", "csv")


def _check_flat_root(names: Sequence[str], expected_files: Sequence[str]) -> list[str]:
    """Archive root must be exactly ``expected_files``: flat, no directories."""
    errors: list[str] = []
    expected = set(expected_files)
    entries = [name for name in names if not name.endswith("/")]
    directories = [name for name in names if name.endswith("/")]

    nested = [name for name in entries if "/" in name.rstrip("/")]
    if nested:
        errors.append(
            f"archive is not flat: {len(nested)} entry(ies) inside a directory, "
            f"e.g. {nested[:3]} (root must contain only {sorted(expected)})"
        )
    if directories:
        errors.append(f"archive contains directory entries: {directories[:3]}")

    actual = set(entries)
    extra = sorted(actual - expected)
    missing = sorted(expected - actual)
    if missing:
        errors.append(f"missing required root file(s): {missing}")
    if extra:
        errors.append(f"unexpected extra root file(s): {extra}")
    return errors


def verify_submission(
    archive: PathLike,
    expected_ids: Sequence[str] | PathLike,
    cfg: Config | None = None,
    *,
    kind: str = "a",
    report_path: PathLike | None = None,
) -> VerifyResult:
    """Verify ``submit.zip`` (``kind='a'``) or ``b_submission.zip`` (``kind='b'``)."""
    cfg = cfg or load_config()
    archive = Path(archive)
    if not archive.is_file():
        raise FileNotFoundError(f"archive not found: {archive}")

    expected_ids = (
        read_expected_ids(expected_ids)
        if isinstance(expected_ids, (str, Path))
        else list(expected_ids)
    )
    submit_cfg = cfg.submit
    root_files = (
        list(submit_cfg.a_root_files) if kind == "a" else list(submit_cfg.b_root_files)
    )

    errors: list[str] = []
    checks = ["archive_readable", "flat_root", "exact_file_set"]
    n_rows = 0

    try:
        with zipfile.ZipFile(archive, "r") as zf:
            names = zf.namelist()
            bad_zip = zf.testzip()
            if bad_zip is not None:
                errors.append(f"corrupt entry in archive: {bad_zip}")

            errors.extend(_check_flat_root(names, root_files))

            if kind == "a":
                if submit_cfg.predictions_file in names:
                    checks.append("predictions_csv")
                    inner = verify_predictions_bytes(
                        zf.read(submit_cfg.predictions_file), expected_ids, cfg
                    )
                    errors.extend(inner.errors)
                    n_rows = inner.n_rows
            else:
                checks.append("nested_submit_zip")
                inner_name = submit_cfg.a_zip_name
                if inner_name in names:
                    try:
                        with zipfile.ZipFile(io.BytesIO(zf.read(inner_name))) as inner_zip:
                            inner_names = inner_zip.namelist()
                            errors.extend(_check_flat_root(inner_names, submit_cfg.a_root_files))
                            if submit_cfg.predictions_file in inner_names:
                                checks.append("nested_predictions_csv")
                                nested = verify_predictions_bytes(
                                    inner_zip.read(submit_cfg.predictions_file), expected_ids, cfg
                                )
                                errors.extend(f"[{inner_name}] {e}" for e in nested.errors)
                                n_rows = nested.n_rows
                    except zipfile.BadZipFile as exc:
                        errors.append(f"{inner_name} is not a readable zip: {exc}")
                commit_name = "solution_commit.txt"
                if commit_name in names:
                    checks.append("solution_commit_nonempty")
                    if not zf.read(commit_name).strip():
                        errors.append(f"{commit_name} is empty")
    except zipfile.BadZipFile as exc:
        return VerifyResult(False, [f"not a readable zip: {exc}"], checks, 0, str(archive), kind)

    result = VerifyResult(not errors, errors, checks, n_rows, str(archive), kind)
    if report_path is not None:
        result.write_report(report_path)
    return result


def main() -> None:  # pragma: no cover - CLI glue
    import argparse
    import json

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
    raise SystemExit(0 if result.ok else 1)


if __name__ == "__main__":  # pragma: no cover
    main()
