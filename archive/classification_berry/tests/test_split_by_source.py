"""Grouped K-fold: the leakage gate is the point of this file.

Acceptance criterion #1: "分组划分无泄漏断言". If a source image can straddle a
fold, every local model-selection decision is suspect, so these tests assert
the invariant directly rather than trusting the implementation.
"""
from __future__ import annotations

from collections import Counter

import pytest

from data.manifest import ManifestRecord, derive_source_id, read_labels, source_grouping_report
from data.split_by_source import (
    SplitError,
    assert_full_coverage,
    assert_no_source_leakage,
    build_folds,
    fold_summary,
    load_folds,
    run_split,
)


def _records(spec: dict[str, list[int]]) -> list[ManifestRecord]:
    """Build records from {source_id: [labels]} with deterministic ids."""
    records: list[ManifestRecord] = []
    counter = 0
    for source_id, labels in spec.items():
        for label in labels:
            records.append(
                ManifestRecord(
                    image_id=f"{counter:05d}",
                    file_name=f"{source_id}_{counter:03d}.jpg",
                    label=label,
                    source_id=source_id,
                )
            )
            counter += 1
    return records


@pytest.fixture
def records() -> list[ManifestRecord]:
    spec = {}
    for index in range(30):
        source = f"src{index:03d}"
        # 3-6 crops per source, labels correlated within a source.
        base = index % 3
        labels = [base] * (3 + index % 4)
        if index % 5 == 0:
            labels.append((base + 1) % 3)
        spec[source] = labels
    return _records(spec)


def test_no_source_spans_train_and_val(records: list[ManifestRecord]) -> None:
    result = build_folds(records, n_folds=5, seed=42)
    for fold in result.folds:
        assert not (set(fold.train_sources) & set(fold.val_sources))
    # and the module-level guard agrees
    assert_no_source_leakage(result.folds)


def test_every_record_validated_exactly_once(records: list[ManifestRecord]) -> None:
    result = build_folds(records, n_folds=5, seed=42)
    assert_full_coverage(result.folds, records)
    validated = [index for fold in result.folds for index in fold.val_indices]
    assert sorted(validated) == list(range(len(records)))


def test_train_and_val_indices_partition_the_records(records: list[ManifestRecord]) -> None:
    result = build_folds(records, n_folds=5, seed=42)
    for fold in result.folds:
        assert set(fold.train_indices) & set(fold.val_indices) == set()
        assert len(fold.train_indices) + len(fold.val_indices) == len(records)


def test_every_training_fold_sees_all_classes(records: list[ManifestRecord]) -> None:
    result = build_folds(records, n_folds=5, seed=42)
    for fold in result.folds:
        assert {records[i].label for i in fold.train_indices} == {0, 1, 2}


def test_split_is_deterministic_for_a_seed(records: list[ManifestRecord]) -> None:
    first = build_folds(records, n_folds=5, seed=42)
    second = build_folds(records, n_folds=5, seed=42)
    assert [f.val_sources for f in first.folds] == [f.val_sources for f in second.folds]


def test_different_seeds_generally_differ(records: list[ManifestRecord]) -> None:
    a = build_folds(records, n_folds=5, seed=42)
    b = build_folds(records, n_folds=5, seed=43)
    assert [f.val_sources for f in a.folds] != [f.val_sources for f in b.folds]


def test_source_disjointness_survives_many_seeds(records: list[ManifestRecord]) -> None:
    """The invariant must hold for every seed, not just the default one."""
    for seed in range(25):
        result = build_folds(records, n_folds=5, seed=seed)
        assert_no_source_leakage(result.folds)
        assert_full_coverage(result.folds, records)


def test_too_few_sources_is_rejected() -> None:
    few = _records({"srcA": [0, 1], "srcB": [2]})
    with pytest.raises(SplitError, match="at least 5 distinct source images"):
        build_folds(few, n_folds=5, seed=42)


def test_folds_are_balanced_by_crop_count(records: list[ManifestRecord]) -> None:
    result = build_folds(records, n_folds=5, seed=42)
    sizes = [len(fold.val_indices) for fold in result.folds]
    # LPT balancing over 30 sources: no fold may be wildly off the mean.
    assert max(sizes) - min(sizes) <= max(6, 0.5 * (sum(sizes) / len(sizes)))


def test_run_split_persists_and_roundtrips(
    sandbox_cfg, records, tmp_path
) -> None:
    from data.manifest import read_labels as _read_labels
    from data.manifest import write_manifest

    manifest_path = tmp_path / "manifest.csv"
    parsed = read_labels(sandbox_cfg)
    write_manifest(manifest_path, parsed)

    result = run_split(sandbox_cfg, manifest_path=manifest_path, out_dir=tmp_path / "splits")
    assert (tmp_path / "splits" / "folds.json").is_file()
    assert (tmp_path / "splits" / "summary.txt").is_file()

    reloaded, reloaded_records = load_folds(tmp_path / "splits" / "folds.json")
    assert [f.val_indices for f in reloaded.folds] == [f.val_indices for f in result.folds]
    assert [r.image_id for r in reloaded_records] == [r.image_id for r in parsed]
    assert "grouped 5-fold" in fold_summary(result, parsed)


def test_source_grouping_report_flags_broken_grouping() -> None:
    good = _records({"srcA": [0, 1], "srcB": [2, 0]})
    report = source_grouping_report(good)
    assert report["n_sources"] == 2
    assert report["grouping_looks_broken"] is False
    assert report["sources_with_multiple_labels"] == 2

    # One crop per "source" means the derivation failed to group anything,
    # which is precisely the state that would leak.
    broken = [
        ManifestRecord(f"i{i}", f"unique{i}.jpg", i % 3, f"unique{i}") for i in range(4)
    ]
    broken_report = source_grouping_report(broken)
    assert broken_report["n_sources"] == broken_report["n_images"]
    assert broken_report["grouping_looks_broken"] is True


def test_derive_source_id_prefers_directory() -> None:
    from common.config import load_config

    cfg = load_config()
    # 1. directory component wins
    assert derive_source_id("src_017/berry_003.jpg", cfg) == "src_017"
    # 2. fallback splits on the FIRST delimiter (documented, lossy for
    #    source tokens that themselves contain a delimiter)
    assert derive_source_id("src0007_berry_003.jpg", cfg) == "src0007"
    assert derive_source_id("plain.jpg", cfg) == "plain"


def test_derive_source_id_regex_mode_handles_underscored_sources() -> None:
    """The regex escape hatch for names like ``src_017_berry_003.jpg``."""
    import copy

    from common.config import Config, load_config

    cfg = Config(copy.deepcopy(dict(load_config())))
    cfg["data"]["source_key"]["regex"] = r"^(?P<source>src_\d+)_"
    assert derive_source_id("src_017_berry_003.jpg", cfg) == "src_017"
    assert derive_source_id("src_1234_berry_056.jpg", cfg) == "src_1234"


def test_derive_source_id_regex_mode_errors_when_unmatched() -> None:
    import copy

    from common.config import Config, load_config

    cfg = Config(copy.deepcopy(dict(load_config())))
    cfg["data"]["source_key"]["regex"] = r"^(?P<source>src_\d+)_"
    with pytest.raises(ValueError, match="did not match a named 'source' group"):
        derive_source_id("totally_different.jpg", cfg)


def test_label_histogram_is_reported(records: list[ManifestRecord]) -> None:
    result = build_folds(records, n_folds=5, seed=42)
    histogram = result.audit["label_histogram"]
    assert set(histogram) == {0, 1, 2}
    assert sum(histogram.values()) == len(records)
    assert dict(Counter(r.label for r in records)) == {int(k): v for k, v in histogram.items()}
