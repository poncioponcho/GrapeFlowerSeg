"""Regression guard for the detector's target contract.

The bug this file exists for: ``targets["masks"]`` was emitted as
``[N, 1, H, W]`` instead of torchvision's required ``UInt8Tensor[N, H, W]``.
Training then ran to completion with a healthy-looking total loss while the
mask head collapsed to ~0 everywhere - the box head learned normally, so
nothing in the loss curve hinted at a problem. It was only visible by
inspecting a predicted mask (max 0.0096) against its box score (0.89).

The dataset lives in a script rather than a package module, so it is imported
by path.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common.config import load_config  # noqa: E402
from data.coco import load_train_annotations, load_train_images  # noqa: E402


def _load_train_module():
    path = REPO_ROOT / "scripts" / "train_segmentation.py"
    spec = importlib.util.spec_from_file_location("_train_segmentation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train_module():
    return _load_train_module()


@pytest.fixture(scope="module")
def dataset(base_config, data_available):
    if not data_available:
        pytest.skip("competition data not present")
    module = _load_train_module()
    images = load_train_images(base_config)
    annotations = load_train_annotations(base_config)
    return module.build_dataset(
        images[:4],
        annotations,
        base_config.path("data.train_images"),
        384,
        False,
    )


def test_mask_target_is_three_dimensional(dataset) -> None:
    """torchvision requires [N, H, W]; [N, 1, H, W] silently breaks the mask head."""
    _, target = dataset[0]
    assert target["masks"].dim() == 3, (
        f"targets['masks'] must be [N, H, W], got {tuple(target['masks'].shape)}"
    )


def test_mask_target_dtype_is_uint8(dataset) -> None:
    import torch

    _, target = dataset[0]
    assert target["masks"].dtype == torch.uint8


def test_mask_targets_match_the_image_resolution(dataset) -> None:
    image, target = dataset[0]
    _, height, width = image.shape
    assert target["masks"].shape[1:] == (height, width)


def test_instance_counts_agree_across_fields(dataset) -> None:
    _, target = dataset[0]
    n_masks = target["masks"].shape[0]
    assert n_masks == target["boxes"].shape[0]
    assert n_masks == target["labels"].shape[0]
    assert n_masks == target["area"].shape[0]
    assert n_masks == target["iscrowd"].shape[0]


def test_mask_targets_are_not_all_zero(dataset) -> None:
    """An all-zero mask target is the symptom of the original bug."""
    _, target = dataset[0]
    assert target["masks"].shape[0] > 0, "no instances in the probe image"
    assert bool(target["masks"].any()), "mask target is entirely zero"


def test_labels_are_one_based_never_zero(dataset, base_config) -> None:
    """Labels must be 1..num_classes-1; label 0 is torchvision's background.

    This test previously asserted the opposite (labels in {0, 1}) and so locked
    the bug in place. Feeding 0-based submission ids makes the model train
    class 0 as background, after which it predicts only the other class - the
    loss curve looks healthy and only the per-class AP reveals it.
    """
    num_classes = len(base_config.competition.submit_labels) + 1
    for index in range(len(dataset)):
        _, target = dataset[index]
        labels = sorted(set(target["labels"].tolist()))
        assert labels, f"no labels at index {index}"
        assert min(labels) >= 1, (
            f"label 0 is background and must never be a target (index {index}): {labels}"
        )
        assert max(labels) < num_classes, (
            f"labels must be < num_classes={num_classes} (index {index}): {labels}"
        )


def test_both_classes_are_present_across_the_dataset(dataset) -> None:
    """Both classes must appear, mapped to distinct one-based labels."""
    seen: set[int] = set()
    for index in range(len(dataset)):
        _, target = dataset[index]
        seen.update(target["labels"].tolist())
    assert seen == {1, 2}, f"expected labels {{1, 2}}, saw {sorted(seen)}"


def test_submission_id_maps_to_label_by_adding_one(dataset, base_config) -> None:
    """The +1 shift must be the *only* difference between the two numberings."""
    import numpy as np

    from data.coco import load_train_annotations

    annotations = load_train_annotations(base_config)
    by_image: dict[int, list[int]] = {}
    for annotation in annotations:
        by_image.setdefault(annotation["image_id"], []).append(annotation["category_id"])

    _, target = dataset[0]
    image_id = int(target["image_id"].item())
    expected = sorted(value + 1 for value in by_image.get(image_id, []))
    # Instance order may differ, so compare as multisets.
    assert sorted(target["labels"].tolist()) == expected


def test_boxes_are_within_the_resized_image(dataset) -> None:
    image, target = dataset[0]
    _, height, width = image.shape
    for x0, y0, x1, y1 in target["boxes"].tolist():
        assert 0 <= x0 < x1 <= width, f"bad x range {(x0, x1)} for width {width}"
        assert 0 <= y0 < y1 <= height, f"bad y range {(y0, y1)} for height {height}"
