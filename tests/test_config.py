"""Config loading and validation.

The single most important property: training and serving read the *same* file. Every
guard against train/serve skew depends on there being exactly one source of truth, so
these tests pin the values the rest of the project assumes.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from nasa_space_ml.config import (
    DEFAULT_CONFIG_PATH,
    Config,
    config_candidates,
    resolve_default_config_path,
)
from nasa_space_ml.errors import ConfigError


def test_default_config_loads(config: Config) -> None:
    assert config.exoplanets.table == "Q1_Q17_DR25_KOI"
    assert config.asteroids.expected_api_version == "1.5"


def test_labels_are_ordered_negative_first(config: Config) -> None:
    """`labels[0]` is the negative class everywhere in the codebase; the convention is load-bearing."""
    assert config.exoplanets.labels[0] == "CONFIRMED"
    assert config.asteroids.labels[0] == "NOT_PHA"
    assert config.asteroids.labels[1] == "PHA"


def test_pha_thresholds_match_nasas_published_definition(config: Config) -> None:
    assert config.asteroids.pha_distance_au == 0.05
    assert config.asteroids.pha_absolute_magnitude == 22.0


def test_split_fractions_leave_room_for_train(config: Config) -> None:
    assert config.split.test_size + config.split.val_size < 1.0
    config.split.validate()


def test_split_rejects_a_broken_ratio(config: Config) -> None:
    bad = replace(config.split, test_size=0.7, val_size=0.4)
    with pytest.raises(ConfigError, match=r"must be < 1\.0"):
        bad.validate()


def test_split_rejects_one_fold(config: Config) -> None:
    with pytest.raises(ConfigError, match="at least 2"):
        replace(config.split, n_folds=1).validate()


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        Config.load(tmp_path / "nope.yaml")


def test_missing_required_key_is_named(tmp_path: Path) -> None:
    """A half-written config must say which key it choked on, not just fail."""
    path = tmp_path / "settings.yaml"
    path.write_text("random_seed: 1\nhttp:\n  timeout_seconds: 5\n", encoding="utf-8")
    with pytest.raises(ConfigError) as excinfo:
        Config.load(path)
    assert str(excinfo.value)


def test_group_columns_are_the_star_and_the_object_not_the_row(config: Config) -> None:
    """Grouping by host star and by designation is the whole leakage defence; if someone
    'simplifies' these back to a per-row id the tests here should fail loudly."""
    assert config.exoplanets.group_column == "kepid"
    assert config.asteroids.group_column == "des"


def test_seed_is_stable_across_loads(config: Config) -> None:
    assert Config.load().seed == config.seed == 20261111


class TestConfigFileDiscovery:
    """Where `settings.yaml` comes from when nobody says.

    The first CI run of this repository installed the package with `pip install .` rather
    than `pip install -e .`, which put it in site-packages - where the package-relative
    default is not a real file - and then `nasa-space-ml --offline status` failed for a
    reason that had nothing to do with the code. Searching upward from the working directory
    fixes that, because the checkout you are standing in is the repository whose thresholds
    you meant.
    """

    def test_environment_variable_is_the_first_candidate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NSM_CONFIG", str(tmp_path / "container.yaml"))
        assert config_candidates(tmp_path)[0] == tmp_path / "container.yaml"

    def test_a_settings_file_any_parent_of_the_cwd_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("NSM_CONFIG", raising=False)
        clone = tmp_path / "clone"
        (clone / "configs").mkdir(parents=True)
        deep = clone / "src" / "nasa_space_ml"
        deep.mkdir(parents=True)
        found = clone / "configs" / "settings.yaml"
        found.write_text("random_seed: 7\n", encoding="utf-8")

        assert resolve_default_config_path(deep) == found

    def test_the_repository_itself_is_found_from_a_subdirectory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`Config.load()` from anywhere inside this checkout must read this checkout."""
        monkeypatch.delenv("NSM_CONFIG", raising=False)
        from_repo_root = Config.load()
        from_a_subdirectory = Config.load(resolve_default_config_path(Path("src").resolve()))
        assert from_a_subdirectory.seed == from_repo_root.seed == 20261111

    def test_nothing_on_disk_falls_back_to_the_named_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The fallback is not a guess that succeeds; it is the path the error message names."""
        monkeypatch.delenv("NSM_CONFIG", raising=False)
        assert resolve_default_config_path(tmp_path) == DEFAULT_CONFIG_PATH
        with pytest.raises(ConfigError, match="config file not found"):
            Config.load(tmp_path / "absent.yaml")
