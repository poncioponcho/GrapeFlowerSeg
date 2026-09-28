"""Shared fixtures: a throwaway project sandbox backed by synthetic data.

Every test that needs "a dataset" gets one generated fresh inside ``tmp_path``,
so tests never read or write the real ``data/raw`` and can run offline on a
CPU-only machine.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # belt-and-braces alongside pytest.ini pythonpath
    sys.path.insert(0, str(SRC))

from common.config import Config, load_config  # noqa: E402
from data.make_sample_data import generate  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def base_config() -> Config:
    return load_config(REPO_ROOT / "configs" / "default.yaml")


@pytest.fixture(scope="session")
def sample_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """One synthetic dataset per test session (generation is not free)."""
    out_dir = tmp_path_factory.mktemp("sample_data")
    return generate(
        out_dir,
        n_sources=40,
        n_test_a=60,
        n_test_b=50,
        seed=1234,
        with_images=False,
        image_size=16,
    )


@pytest.fixture
def sandbox_cfg(base_config: Config, sample_dataset: dict, tmp_path: Path) -> Config:
    """A config whose every path points inside this test's tmp dir."""
    cfg = Config(copy.deepcopy(dict(base_config)))
    cfg["data"]["labels_csv"] = sample_dataset["labels_csv"]
    cfg["data"]["image_dir"] = str(Path(sample_dataset["labels_csv"]).parent / "images")
    cfg["data"]["manifest_path"] = str(tmp_path / "processed" / "manifest.csv")
    cfg["data"]["processed_dir"] = str(tmp_path / "processed")
    cfg["split"]["output_dir"] = str(tmp_path / "processed" / "splits")
    cfg["submit"]["output_dir"] = str(tmp_path / "submissions")
    cfg["submit"]["report_dir"] = str(tmp_path / "reports")
    return cfg


@pytest.fixture
def sample_submission_a(sample_dataset: dict) -> Path:
    return Path(sample_dataset["sample_submission_a"])


@pytest.fixture
def sample_submission_b(sample_dataset: dict) -> Path:
    return Path(sample_dataset["sample_submission_b"])
