"""Grouped K-fold: the leakage gate.

If a source image can straddle a fold, every local Mask mAP is inflated and
model selection is meaningless. These tests assert the invariant directly.
"""
from __future__ import annotations

from collections import Counter

import pytest

from data.coco import PanicleImage
from data.split_by_source import (
    SplitError,
    assert_full_coverage,
    assert_no_source_leakage,
    build_folds,
    fold_summary,
    load_folds,
    run_split,
)


def _images(spec: dict[str, int]) -> list[PanicleImage]:
    """Build images from {source_id: n_crops}, mirroring the real 2-per-source shape."""
    images: list[PanicleImage] = []
    image_id = 1
    for source_id, count in spec.items():
        for index in range(count):
            suffix = "" if index == 0 else f"_aug{index}"
            images.append(
                PanicleImage(
                    image_id=image_id,
                    file_name=f"{source_id}{suffix}.jpg",
                    width=2048,
                    height=1152,
                    source_id=source_id,
                    n_instances=1 + (image_id % 3),
                )
            )
            image_id += 1
    return images


@pytest.fixture
def images() -> list[PanicleImage]:
    return _images({f"IMG_{index}": 2 for index in range(1, 41)})


def test_no_source_spans_train_and_val(images) -> None:
    result = build_folds(images, n_folds=5, seed=42)
    for fold in result.folds:
        assert not (set(fold.train_sources) & set(fold.val_sources))
    assert_no_source_leakage(result.folds)


def test_every_image_validated_exactly_once(images) -> None:
    result = build_folds(images, n_folds=5, seed=42)
    assert_full_coverage(result.folds, images)
    validated = [i for fold in result.folds for i in fold.val_image_ids]
    assert sorted(validated) == sorted(image.image_id for image in images)


def test_indices_partition_the_dataset(images) -> None:
    result = build_folds(images, n_folds=5, seed=42)
    for fold in result.folds:
        assert set(fold.train_image_ids) & set(fold.val_image_ids) == set()
        assert len(fold.train_image_ids) + len(fold.val_image_ids) == len(images)


def test_augmented_copies_never_split_across_folds(images) -> None:
    """The pair IMG_x.jpg / IMG_x_aug1.jpg must land in the same fold."""
    result = build_folds(images, n_folds=5, seed=42)
    by_source: dict[str, set[int]] = {}
    for fold in result.folds:
        for image_id in fold.val_image_ids:
            source = next(i.source_id for i in images if i.image_id == image_id)
            by_source.setdefault(source, set()).add(fold.fold)
    assert all(len(folds) == 1 for folds in by_source.values()), "source split across folds"


def test_split_is_deterministic(images) -> None:
    a = build_folds(images, n_folds=5, seed=42)
    b = build_folds(images, n_folds=5, seed=42)
    assert [f.val_image_ids for f in a.folds] == [f.val_image_ids for f in b.folds]


def test_different_seeds_differ(images) -> None:
    a = build_folds(images, n_folds=5, seed=42)
    b = build_folds(images, n_folds=5, seed=43)
    assert [f.val_image_ids for f in a.folds] != [f.val_image_ids for f in b.folds]


def test_invariant_holds_for_many_seeds(images) -> None:
    for seed in range(20):
        result = build_folds(images, n_folds=5, seed=seed)
        assert_no_source_leakage(result.folds)
        assert_full_coverage(result.folds, images)


def test_too_few_sources_is_rejected() -> None:
    few = _images({"A": 2, "B": 2})
    with pytest.raises(SplitError, match="at least 5 sources"):
        build_folds(few, n_folds=5, seed=42)


def test_folds_are_balanced(images) -> None:
    result = build_folds(images, n_folds=5, seed=42)
    sizes = [len(fold.val_image_ids) for fold in result.folds]
    assert max(sizes) - min(sizes) <= 2  # 40 sources over 5 folds


def test_run_split_persists_and_roundtrips(cfg, data_available) -> None:
    if not data_available:
        pytest.skip("competition data not present")
    from data.coco import load_train_images

    # run_split always reads the real annotations via the config, so the
    # expected image list must come from the same source.
    expected = load_train_images(cfg)
    result = run_split(cfg)
    assert (cfg.path("split.output_dir") / "folds.json").is_file()
    assert (cfg.path("split.output_dir") / "summary.txt").is_file()

    reloaded, reloaded_images = load_folds(cfg.path("split.output_dir") / "folds.json")
    assert [f.val_image_ids for f in reloaded.folds] == [f.val_image_ids for f in result.folds]
    assert [i.image_id for i in reloaded_images] == [i.image_id for i in expected]
    assert "grouped 5-fold" in fold_summary(result)


def test_real_split_matches_the_official_source_count(cfg, data_available) -> None:
    """On the real data the split must produce the official source count."""
    if not data_available:
        pytest.skip("competition data not present")
    from common.config import load_config

    base = load_config()
    result = run_split(cfg)
    assert result.audit["n_sources"] == int(base.competition.n_train_sources)
    assert result.audit["n_images"] == int(base.competition.n_train_images)
    for fold in result.folds:
        assert not (set(fold.train_sources) & set(fold.val_sources))
