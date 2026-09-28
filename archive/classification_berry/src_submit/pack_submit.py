"""Stage 2 of the submission safety net: pack the archive (flat root).

Ported from HardLane's ``src/submit/pack_submit.py``, with one deliberate
difference: that competition required a ``submit/`` wrapper directory, while
this one requires the archive root to contain the file(s) directly.

* A-board: ``submit.zip``  -> root = ``predictions.csv`` only.
* B-board: ``b_submission.zip`` -> root = ``submit.zip`` + ``solution_commit.txt``.

Arcnames are supplied explicitly by the caller and never derived from a
directory scan, which is what makes "no missing, no extra" verifiable rather
than hopeful. Any arcname containing a path separator is rejected up front so a
multi-level archive can never be produced in the first place.
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Mapping, Union

PathLike = Union[str, Path]


class PackError(ValueError):
    """Raised when an archive cannot satisfy the flat-root contract."""


def pack_flat(files: Mapping[str, PathLike], out_zip: PathLike) -> Path:
    """Write ``{arcname: source_path}`` into a zip whose root is flat.

    All members must be plain file names (no ``/``), which is exactly the
    "根目录仅含 ..." requirement. Members are stored in sorted arcname order so
    the archive bytes are reproducible for identical inputs.
    """
    if not files:
        raise PackError("nothing to pack")

    out = Path(out_zip)
    out.parent.mkdir(parents=True, exist_ok=True)

    for arcname, source in files.items():
        if "/" in arcname or "\\" in arcname:
            raise PackError(
                f"arcname {arcname!r} contains a path separator; "
                "the archive root must be flat"
            )
        if arcname in ("", ".", ".."):
            raise PackError(f"invalid arcname {arcname!r}")
        if not Path(source).is_file():
            raise FileNotFoundError(f"missing source file for {arcname!r}: {source}")

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname in sorted(files):
            zf.write(Path(files[arcname]), arcname=arcname)
    return out


def pack_a_submission(predictions_csv: PathLike, out_zip: PathLike, cfg) -> Path:
    """Build ``submit.zip`` (A-board) containing exactly ``predictions.csv``."""
    return pack_flat({str(cfg.submit.predictions_file): predictions_csv}, out_zip)


def pack_b_submission(
    submit_zip: PathLike,
    solution_commit: PathLike,
    out_zip: PathLike,
    cfg,
) -> Path:
    """Build ``b_submission.zip`` containing ``submit.zip`` + ``solution_commit.txt``."""
    return pack_flat(
        {
            str(cfg.submit.a_zip_name): submit_zip,
            "solution_commit.txt": solution_commit,
        },
        out_zip,
    )


def write_solution_commit(
    path: PathLike,
    *,
    model: str,
    config: str,
    commit: str,
    extra: Mapping[str, str] | None = None,
) -> Path:
    """Write the B-board provenance file bound to the solution version."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BerryColorGrade B-board solution commit",
        f"model: {model}",
        f"config: {config}",
        f"commit: {commit}",
    ]
    for key, value in sorted((extra or {}).items()):
        lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


__all__ = [
    "PackError",
    "pack_flat",
    "pack_a_submission",
    "pack_b_submission",
    "write_solution_commit",
]
