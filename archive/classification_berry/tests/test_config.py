"""Config discipline: one source of truth, and no hardcoded constants.

HardLane's governance rule (DECISIONS §14) is that decision constants live only
in ``configs/default.yaml``. This file tests that the rule is actually
enforceable: the distinctive competition numbers must not appear in source.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from common.config import REPO_ROOT, load_config

# Numbers that are decision constants, so they belong in config only.
FORBIDDEN_LITERALS = ("2209", "1769", "168", "130", "218")


def test_config_loads_and_exposes_required_sections() -> None:
    cfg = load_config()
    for section in ("project", "competition", "data", "split", "eval", "submit", "env_pins"):
        assert section in cfg, f"missing config section: {section}"


def test_class_definitions_are_ordinal_and_complete() -> None:
    cfg = load_config()
    assert list(cfg.competition.labels) == [0, 1, 2]
    assert set(cfg.competition.classes) == {0, 1, 2}
    assert cfg.competition.classes[0] == "early_veraison"
    assert cfg.competition.classes[2] == "full_coloration"
    assert cfg.competition.adjacent_pairs == [[0, 1], [1, 2]]


def test_metric_matches_the_official_definition() -> None:
    cfg = load_config()
    assert cfg.competition.metric == "macro_f1"
    assert cfg.eval.average == "macro"
    assert cfg.eval.zero_division == 0
    assert list(cfg.eval.labels) == [0, 1, 2]


def test_split_is_grouped_by_source_not_random() -> None:
    cfg = load_config()
    assert cfg.split.strategy == "grouped_kfold_by_source"
    assert cfg.split.n_folds == 5
    assert cfg.data.source_key.mode in ("derive_from_filename", "column")


def test_submission_contract_fields() -> None:
    cfg = load_config()
    assert list(cfg.submit.predictions_header) == ["image_id", "label"]
    assert cfg.submit.a_root_files == ["predictions.csv"]
    assert sorted(cfg.submit.b_root_files) == ["solution_commit.txt", "submit.zip"]
    assert list(cfg.submit.valid_labels) == [0, 1, 2]


def test_env_pins_are_recorded() -> None:
    cfg = load_config()
    pins = cfg.env_pins
    for key in ("python", "numpy", "scikit-learn"):
        assert pins.get(key), f"env pin missing: {key}"


def test_installed_versions_match_the_pins() -> None:
    """Drift between the pinned and installed versions invalidates every number."""
    import numpy
    import sklearn

    cfg = load_config()
    assert numpy.__version__ == str(cfg.env_pins.numpy)
    assert sklearn.__version__ == str(cfg.env_pins["scikit-learn"])


def test_paths_resolve_under_repo_root() -> None:
    cfg = load_config()
    for key in ("data.labels_csv", "data.manifest_path", "split.output_dir"):
        resolved = cfg.path(key)
        assert resolved.is_absolute()
        assert str(resolved).startswith(str(REPO_ROOT))


def test_no_hardcoded_competition_constants_in_source() -> None:
    """Scan src/ for the distinctive constants that must live in config only."""
    offenders: list[str] = []
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for literal in FORBIDDEN_LITERALS:
            if literal in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {literal}")
    assert not offenders, "hardcoded decision constants found: " + "; ".join(offenders)


def test_config_has_no_duplicate_top_level_keys() -> None:
    """PyYAML silently keeps the last duplicate key; catch that class of typo."""
    import yaml

    text = (REPO_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")

    class StrictLoader(yaml.SafeLoader):
        pass

    seen: list[str] = []

    def construct_mapping(loader, node, deep=False):  # type: ignore[no-untyped-def]
        keys = [loader.construct_object(k, deep=deep) for k, _ in node.value]
        duplicates = {k for k in keys if keys.count(k) > 1}
        if duplicates:
            seen.extend(sorted(map(str, duplicates)))
        return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)

    StrictLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    yaml.load(text, Loader=StrictLoader)
    assert not seen, f"duplicate YAML keys: {seen}"


@pytest.mark.parametrize("key", ["a_zip_name", "b_zip_name", "predictions_file"])
def test_submit_file_names_are_configured(key: str) -> None:
    cfg = load_config()
    assert str(cfg.submit[key]).strip()
