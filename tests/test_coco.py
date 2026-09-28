"""Data layer: COCO parsing, category renumbering, and source grouping.

The source grouping is the highest-risk assumption in the project: if
``IMG_x.jpg`` and ``IMG_x_aug1.jpg`` are not recognised as one source, local
validation leaks and every model decision is made on an inflated number.
"""
from __future__ import annotations

import pytest

from data.coco import (
    CocoError,
    dataset_stats,
    derive_source_id,
    gt_category_to_submit,
    load_ignore_regions,
    load_train_annotations,
    load_train_images,
)


def test_source_id_strips_augmentation_suffix(base_config) -> None:
    assert derive_source_id("IMG_332.jpg", base_config) == "IMG_332"
    assert derive_source_id("IMG_332_aug1.jpg", base_config) == "IMG_332"
    assert derive_source_id("IMG_1_aug1.JPG", base_config) == "IMG_1"


def test_augmented_pair_shares_one_source(base_config, data_available) -> None:
    if not data_available:
        pytest.skip("competition data not present")
    images = load_train_images(base_config)
    by_name = {image.file_name: image for image in images}
    assert by_name["IMG_1.jpg"].source_id == by_name["IMG_1_aug1.jpg"].source_id


def test_every_source_has_exactly_the_original_and_the_augmentation(
    base_config, data_available
) -> None:
    if not data_available:
        pytest.skip("competition data not present")
    images = load_train_images(base_config)
    per_source: dict[str, int] = {}
    for image in images:
        per_source[image.source_id] = per_source.get(image.source_id, 0) + 1
    assert set(per_source.values()) == {2}, "expected exactly 2 images per source"


def test_category_renumbering_is_mandatory_and_correct(base_config, data_available) -> None:
    """Training ids 1/2 must become submission ids 0/1, never identity-mapped."""
    if not data_available:
        pytest.skip("competition data not present")
    assert gt_category_to_submit(1, base_config) == 0   # 完整花穗
    assert gt_category_to_submit(2, base_config) == 1   # 不完整花穗
    with pytest.raises(CocoError, match="unexpected GT category_id"):
        gt_category_to_submit(0, base_config)


def test_annotations_are_renumbered_to_submit_ids(base_config, data_available) -> None:
    if not data_available:
        pytest.skip("competition data not present")
    annotations = load_train_annotations(base_config)
    assert {a["category_id"] for a in annotations} == {0, 1}


def test_dataset_stats_are_self_consistent(base_config, data_available) -> None:
    if not data_available:
        pytest.skip("competition data not present")
    stats = dataset_stats(base_config)
    assert stats["grouping_looks_broken"] is False
    assert stats["n_sources"] * 2 == stats["n_images"]
    assert stats["n_annotations"] == sum(stats["submit_category_histogram"].values())
    assert stats["n_ignore_regions"] > 0
    assert stats["instances_per_image_max"] >= stats["instances_per_image_min"]


def test_ignore_regions_are_not_a_third_class(base_config, data_available) -> None:
    """Ignore regions carry no category and must not be treated as predictions."""
    if not data_available:
        pytest.skip("competition data not present")
    regions = load_ignore_regions(base_config)
    assert regions
    assert all("category_id" not in region for region in regions)
    assert all(region["segmentation"] for region in regions)


def test_missing_coco_file_raises(tmp_path, cfg) -> None:
    from data.coco import load_coco

    with pytest.raises(FileNotFoundError):
        load_coco(tmp_path / "nope.json")


def test_duplicate_json_keys_are_rejected(tmp_path) -> None:
    from data.coco import load_coco

    path = tmp_path / "dup.json"
    path.write_text('{"a": 1, "a": 2}', encoding="utf-8")
    with pytest.raises(CocoError, match="duplicate JSON key"):
        load_coco(path)
