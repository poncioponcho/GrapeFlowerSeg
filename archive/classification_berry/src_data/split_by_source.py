"""Grouped K-fold split keyed on the original source image (果穗图).

Adapted from the HardLane repo's ``src/data/split_by_clip.py``: the original
splits by *video clip* because frames from one clip are near-duplicates; here
we split by *source image* because berries cropped from one 果穗图 share
lighting, device colour cast, vine, and often a very similar colour state.

The organiser already did exactly this for train/A/B, so a random per-image
split locally would produce a validation score that is systematically too high
and would pick the wrong model. The grouping is therefore enforced by a hard
assertion, not by convention.

Determinism: the assignment depends only on ``(source_ids, n_folds, seed)``.
Same inputs -> byte-identical fold file, so any reported OOF number is
reproducible.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from common.checksum import sha256_file
from common.config import Config, load_config
from common.io_utils import write_json
from data.manifest import ManifestRecord, read_manifest


class SplitError(ValueError):
    """Raised when a split violates the grouping or coverage contract."""


@dataclass(frozen=True)
class Fold:
    """One train/val partition; indices refer to the input record order."""

    fold: int
    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]
    train_sources: tuple[str, ...]
    val_sources: tuple[str, ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["train_indices"] = list(self.train_indices)
        payload["val_indices"] = list(self.val_indices)
        payload["train_sources"] = list(self.train_sources)
        payload["val_sources"] = list(self.val_sources)
        return payload


@dataclass
class SplitResult:
    n_folds: int
    seed: int
    folds: list[Fold]
    audit: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "strategy": "grouped_kfold_by_source",
            "n_folds": self.n_folds,
            "seed": self.seed,
            "folds": [fold.to_dict() for fold in self.folds],
            "audit": self.audit,
        }


def assign_sources_to_folds(
    source_ids: Sequence[str],
    *,
    n_folds: int,
    seed: int,
    weights: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """Deterministically map each source image to a fold.

    Longest-processing-time-first balancing: sources are visited largest-first
    and dropped into the currently lightest fold, with a seeded shuffle
    breaking ties so the result is reproducible but not correlated with
    alphabetical source ordering.

    ``weights`` defaults to a uniform weight of 1 per source, i.e. folds are
    balanced by *number of source images*. Pass ``{source: n_crops}`` to
    balance by number of berry crops instead, which is usually the better
    objective since a fold's validation size is what drives metric variance.
    """
    unique_sources = sorted(set(source_ids))
    if n_folds < 2:
        raise SplitError(f"n_folds must be >= 2, got {n_folds}")
    if len(unique_sources) < n_folds:
        raise SplitError(
            f"need at least {n_folds} distinct source images, got {len(unique_sources)}"
        )

    weight_of = {s: int((weights or {}).get(s, 1)) for s in unique_sources}
    if any(w <= 0 for w in weight_of.values()):
        raise SplitError("all source weights must be positive")

    rng = random.Random(seed)
    # Seeded shuffle first, then a stable size-descending sort: ties within
    # equal sizes keep the shuffled order, so the result varies with `seed`.
    shuffled = unique_sources[:]
    rng.shuffle(shuffled)
    ordered = sorted(shuffled, key=lambda s: -weight_of[s])

    loads = [0] * n_folds
    assignment: dict[str, int] = {}
    for source in ordered:
        target = min(range(n_folds), key=lambda f: (loads[f], f))
        assignment[source] = target
        loads[target] += weight_of[source]
    return assignment


def assert_no_source_leakage(folds: Iterable[Fold]) -> None:
    """Hard gate: no source image may appear in both halves of a fold."""
    for fold in folds:
        overlap = set(fold.train_sources) & set(fold.val_sources)
        if overlap:
            raise SplitError(
                f"source leakage in fold {fold.fold}: {sorted(overlap)[:5]} "
                f"({len(overlap)} sources)"
            )


def assert_full_coverage(folds: Iterable[Fold], records: Sequence[ManifestRecord]) -> None:
    """Every record is validated exactly once across all folds (OOF contract)."""
    folds = list(folds)
    counter: Counter[int] = Counter()
    for fold in folds:
        counter.update(fold.val_indices)
    expected = set(range(len(records)))
    seen = set(counter)
    if seen != expected:
        raise SplitError(
            f"validation coverage mismatch: missing={sorted(expected - seen)[:5]} "
            f"extra={sorted(seen - expected)[:5]}"
        )
    duplicated = sorted(index for index, count in counter.items() if count != 1)
    if duplicated:
        raise SplitError(
            f"{len(duplicated)} records appear in more than one validation fold "
            f"(e.g. {duplicated[:5]})"
        )


def assert_train_has_all_labels(folds: Iterable[Fold], records: Sequence[ManifestRecord]) -> None:
    """Every training fold must see all classes, else Macro-F1 is ill-defined."""
    for fold in folds:
        labels = {records[i].label for i in fold.train_indices}
        if len(labels) < 2:
            raise SplitError(
                f"fold {fold.fold}: training split has only label(s) {sorted(labels)}"
            )


def build_folds(
    records: Sequence[ManifestRecord],
    *,
    n_folds: int,
    seed: int,
) -> SplitResult:
    """Grouped K-fold split with all leakage/coverage assertions applied."""
    records = list(records)
    if not records:
        raise SplitError("no records to split")

    source_ids = [record.source_id for record in records]
    weights = Counter(source_ids)
    assignment = assign_sources_to_folds(
        source_ids, n_folds=n_folds, seed=seed, weights=weights
    )

    by_fold: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_fold[assignment[record.source_id]].append(index)

    folds: list[Fold] = []
    for fold_id in range(n_folds):
        val_indices = tuple(sorted(by_fold.get(fold_id, ())))
        val_set = set(val_indices)
        train_indices = tuple(i for i in range(len(records)) if i not in val_set)
        if not val_indices:
            raise SplitError(f"fold {fold_id} is empty; reduce n_folds or add sources")
        folds.append(
            Fold(
                fold=fold_id,
                train_indices=train_indices,
                val_indices=val_indices,
                train_sources=tuple(sorted({records[i].source_id for i in train_indices})),
                val_sources=tuple(sorted({records[i].source_id for i in val_indices})),
            )
        )

    assert_no_source_leakage(folds)
    assert_full_coverage(folds, records)
    assert_train_has_all_labels(folds, records)

    audit = {
        "n_records": len(records),
        "n_sources": len(set(source_ids)),
        "fold_val_sizes": [len(fold.val_indices) for fold in folds],
        "fold_val_sources": [len(fold.val_sources) for fold in folds],
        "label_histogram": dict(sorted(Counter(r.label for r in records).items())),
        "balance_objective": "n_crops_per_source",
    }
    return SplitResult(n_folds=n_folds, seed=seed, folds=folds, audit=audit)


def fold_summary(result: SplitResult, records: Sequence[ManifestRecord]) -> str:
    """Human-readable fold table for the console / report."""
    lines = [
        f"grouped {result.n_folds}-fold by source image  (seed={result.seed})",
        f"{'fold':>4}  {'#train':>7}  {'#val':>6}  {'#src_tr':>7}  {'#src_val':>8}  val label counts",
    ]
    for fold in result.folds:
        counts = Counter(records[i].label for i in fold.val_indices)
        label_str = " ".join(f"{label}:{counts.get(label, 0)}" for label in sorted(counts))
        lines.append(
            f"{fold.fold:>4}  {len(fold.train_indices):>7}  {len(fold.val_indices):>6}  "
            f"{len(fold.train_sources):>7}  {len(fold.val_sources):>8}  {label_str}"
        )
    return "\n".join(lines)


def run_split(
    cfg: Config | None = None,
    *,
    manifest_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    n_folds: int | None = None,
    seed: int | None = None,
) -> SplitResult:
    """Load the manifest, build folds, persist ``folds.json`` + a summary."""
    cfg = cfg or load_config()
    manifest_path = Path(manifest_path) if manifest_path else cfg.path("data.manifest_path")
    out_dir = Path(out_dir) if out_dir else cfg.path("split.output_dir")
    n_folds = n_folds if n_folds is not None else int(cfg.split.n_folds)
    seed = seed if seed is not None else int(cfg.split.seed)

    records = read_manifest(manifest_path)
    result = build_folds(records, n_folds=n_folds, seed=seed)
    result.audit["manifest_sha256"] = sha256_file(manifest_path)
    result.audit["manifest_path"] = str(manifest_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    payload["records"] = [record.to_row() for record in records]
    write_json(out_dir / "folds.json", payload)
    (out_dir / "summary.txt").write_text(fold_summary(result, records) + "\n", encoding="utf-8")
    return result


def load_folds(path: str | Path) -> tuple[SplitResult, list[ManifestRecord]]:
    """Read back a persisted split (used by the training/OOF scripts)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    folds = [
        Fold(
            fold=entry["fold"],
            train_indices=tuple(entry["train_indices"]),
            val_indices=tuple(entry["val_indices"]),
            train_sources=tuple(entry["train_sources"]),
            val_sources=tuple(entry["val_sources"]),
        )
        for entry in raw["folds"]
    ]
    records = [
        ManifestRecord(
            image_id=row["image_id"],
            file_name=row["file_name"],
            label=int(row["label"]),
            source_id=row["source_id"],
            split=row.get("split", "train"),
            fold=int(row.get("fold", -1)),
        )
        for row in raw["records"]
    ]
    result = SplitResult(
        n_folds=raw["n_folds"], seed=raw["seed"], folds=folds, audit=raw.get("audit", {})
    )
    return result, records


def main() -> None:  # pragma: no cover - CLI glue
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--n-folds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    result = run_split(
        cfg,
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        n_folds=args.n_folds,
        seed=args.seed,
    )
    records = read_manifest(args.manifest or cfg.path("data.manifest_path"))
    print(fold_summary(result, records))
    print(json.dumps(result.audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
