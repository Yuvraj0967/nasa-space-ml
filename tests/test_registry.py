"""Tests for the save/load guards in `models/registry.py`.

This module is the only thing standing between "the model I trained" and "the model that
answers your requests", so the tests are mostly about *refusing*: a feature list that moved,
a label set that changed, a binary that is not a binary, an artifact written by a different
major version of scikit-learn. Each of those is a bug that produces confident nonsense rather
than an error, which is exactly the class of failure a capstone reviewer cannot see and a
production reviewer cannot forgive.

The last test in the file is the one that matters most: it loads the *real* artifacts
committed under `models/` against the live feature lists, so renaming or reordering a feature
in `features/*.py` without retraining breaks the build instead of the demo.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from nasa_space_ml.errors import ConfigError, DataError, ModelNotTrainedError
from nasa_space_ml.features import asteroids as asteroid_features
from nasa_space_ml.features import exoplanets as exoplanet_features
from nasa_space_ml.models import registry
from nasa_space_ml.models.registry import (
    META_SUFFIX,
    MODEL_SUFFIX,
    LoadedModel,
    environment_snapshot,
    feature_list_hash,
    load_artifact,
    load_metadata,
    model_paths,
    save_artifact,
)

FEATURES = ["depth_ppm", "period_days", "radius_earth"]
LABELS = ["NO", "MAYBE", "YES"]
FINGERPRINT = "0" * 64


def make_estimator() -> DecisionTreeClassifier:
    """A deterministic, instantly-fitted estimator - the guards do not care which one."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(90, 3))
    score = X[:, 0] + 0.6 * X[:, 2]
    y = np.select([score > 0.5, score < -0.5], [2, 0], default=1)
    return DecisionTreeClassifier(random_state=0).fit(X, y)


def save(
    directory: Path, *, features: list[str] = FEATURES, name: str = "toy", **overrides: Any
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "feature_names": features,
        "labels": LABELS,
        "task": "exoplanets",
        "algorithm": "decision_tree",
        "metrics": {"macro_f1": 0.83, "accuracy": 0.9},
        "dataset_fingerprint": FINGERPRINT,
        "rows": 90,
        "groups": 40,
    }
    kwargs.update(overrides)
    return save_artifact(directory, name, make_estimator(), **kwargs)


class TestFeatureListHash:
    def test_is_deterministic_and_short(self) -> None:
        first = feature_list_hash(FEATURES)
        assert first == feature_list_hash(list(FEATURES))
        assert len(first) == 16
        assert all(character in "0123456789abcdef" for character in first)

    def test_different_lists_get_different_hashes(self) -> None:
        assert feature_list_hash(FEATURES) != feature_list_hash([*FEATURES, "extra"])

    def test_order_is_part_of_the_hash(self) -> None:
        """Same three names, different order: a different matrix, so a different hash.

        sklearn takes columns positionally. If the hash ignored order, reordering
        FEATURE_COLUMNS would still "match" the artifact and every prediction would be
        scrambled while every check passed.
        """
        assert feature_list_hash(FEATURES) != feature_list_hash(list(reversed(FEATURES)))

    def test_the_separator_makes_concatenation_unambiguous(self) -> None:
        """['ab', 'c'] must not hash like ['a', 'bc'] - a plain join would collide."""
        assert feature_list_hash(["ab", "c"]) != feature_list_hash(["a", "bc"])

    def test_a_tuple_and_a_list_agree(self) -> None:
        assert feature_list_hash(tuple(FEATURES)) == feature_list_hash(FEATURES)


class TestModelPaths:
    def test_suffixes_are_the_declared_ones(self, tmp_path: Path) -> None:
        binary, meta = model_paths(tmp_path, "toy")
        assert binary == tmp_path / f"toy{MODEL_SUFFIX}"
        assert meta == tmp_path / f"toy{META_SUFFIX}"
        assert MODEL_SUFFIX == ".joblib"
        assert META_SUFFIX == ".json"


class TestSaveArtifact:
    def test_writes_both_files_and_creates_the_directory(self, tmp_path: Path) -> None:
        directory = tmp_path / "not" / "made" / "yet"
        save(directory)
        assert directory.is_dir()
        binary, meta = model_paths(directory, "toy")
        assert binary.exists() and meta.exists()

    def test_metadata_carries_the_whole_contract(self, tmp_path: Path) -> None:
        metadata = save(tmp_path)
        assert set(metadata) >= {
            "schema_version",
            "name",
            "task",
            "algorithm",
            "feature_names",
            "feature_hash",
            "n_features",
            "labels",
            "metrics",
            "dataset_fingerprint",
            "rows",
            "groups",
            "trained_utc",
            "environment",
        }
        assert metadata["feature_names"] == FEATURES
        assert metadata["n_features"] == len(FEATURES)
        assert metadata["feature_hash"] == feature_list_hash(FEATURES)

    def test_the_sidecar_is_json_a_human_can_read(self, tmp_path: Path) -> None:
        """The point of a sidecar is that reviewing the numbers needs no binary loader."""
        save(tmp_path)
        text = model_paths(tmp_path, "toy")[1].read_text(encoding="utf-8")
        assert text.startswith('{\n  "algorithm"')  # indent=2, sort_keys=True
        assert json.loads(text)["metrics"]["macro_f1"] == 0.83

    def test_numpy_row_counts_survive_the_json_writer(self, tmp_path: Path) -> None:
        metadata = save(tmp_path, rows=np.int64(4321), groups=np.int32(12))
        assert isinstance(metadata["rows"], int) and isinstance(metadata["groups"], int)
        assert (
            json.loads(model_paths(tmp_path, "toy")[1].read_text(encoding="utf-8"))["rows"] == 4321
        )

    def test_extra_is_merged_without_clobbering_the_contract(self, tmp_path: Path) -> None:
        metadata = save(tmp_path, extra={"cv": [{"name": "a", "mean": 0.5}], "name": "ignored"})
        assert metadata["cv"] == [{"name": "a", "mean": 0.5}]
        # `extra` is applied after the built-in keys, so a caller that passes "name" in extra
        # does win. Documented by the assertion: nothing in the repo relies on it, and a
        # future caller needs to know which side is authoritative.
        assert metadata["name"] == "ignored"

    def test_trained_utc_is_a_parseable_timestamp(self, tmp_path: Path) -> None:
        metadata = save(tmp_path)
        assert datetime.fromisoformat(metadata["trained_utc"]).tzinfo is not None

    def test_environment_snapshot_names_the_installed_sklearn(self) -> None:
        snapshot = environment_snapshot()
        assert set(snapshot) == {"python", "scikit_learn", "platform"}
        assert snapshot["scikit_learn"] == registry.sklearn.__version__
        assert snapshot["python"].split(".")[0].isdigit()


class TestLoadMetadata:
    def test_missing_sidecar_names_the_command_to_run(self, tmp_path: Path) -> None:
        with pytest.raises(ModelNotTrainedError) as excinfo:
            load_metadata(tmp_path, "absent")
        assert "nasa-space-ml train" in str(excinfo.value)
        assert excinfo.value.details["path"].endswith(f"absent{META_SUFFIX}")

    def test_reads_back_what_was_written(self, tmp_path: Path) -> None:
        written = save(tmp_path)
        assert load_metadata(tmp_path, "toy") == written


class TestLoadArtifact:
    def test_roundtrip_predictions_are_identical(self, tmp_path: Path) -> None:
        original = make_estimator()
        save_artifact(
            tmp_path,
            "toy",
            original,
            feature_names=FEATURES,
            labels=LABELS,
            task="exoplanets",
            algorithm="decision_tree",
            metrics={},
            dataset_fingerprint=FINGERPRINT,
            rows=90,
            groups=40,
        )
        loaded = load_artifact(tmp_path, "toy", expected_feature_names=FEATURES)
        rng = np.random.default_rng(1)
        X = rng.normal(size=(25, 3))
        assert np.array_equal(loaded.estimator.predict(X), original.predict(X))
        assert np.allclose(loaded.estimator.predict_proba(X), original.predict_proba(X))

    def test_returns_a_loaded_model_exposing_the_contract(self, tmp_path: Path) -> None:
        save(tmp_path)
        loaded = load_artifact(tmp_path, "toy", expected_feature_names=FEATURES)
        assert isinstance(loaded, LoadedModel)
        assert loaded.name == "toy"
        assert loaded.feature_names == tuple(FEATURES)
        assert loaded.labels == tuple(LABELS)
        assert isinstance(loaded.metadata, dict)

    def test_a_feature_added_after_training_is_refused(self, tmp_path: Path) -> None:
        save(tmp_path)
        live = [*FEATURES, "impact_parameter"]
        with pytest.raises(DataError, match="different feature list") as excinfo:
            load_artifact(tmp_path, "toy", expected_feature_names=live)
        details = excinfo.value.details
        assert details["missing_in_model"] == ["impact_parameter"]
        assert details["extra_in_model"] == []
        assert details["live_hash"] == feature_list_hash(live)
        assert "Re-run" in str(excinfo.value)

    def test_a_feature_removed_after_training_is_refused(self, tmp_path: Path) -> None:
        save(tmp_path)
        with pytest.raises(DataError) as excinfo:
            load_artifact(tmp_path, "toy", expected_feature_names=FEATURES[:2])
        assert excinfo.value.details["extra_in_model"] == ["radius_earth"]

    def test_a_reordered_feature_list_is_refused(self, tmp_path: Path) -> None:
        """The guard is list equality, not set equality - order is a column mapping."""
        save(tmp_path)
        with pytest.raises(DataError):
            load_artifact(tmp_path, "toy", expected_feature_names=list(reversed(FEATURES)))

    def test_a_renamed_feature_reports_both_sides_of_the_gap(self, tmp_path: Path) -> None:
        save(tmp_path)
        live = ["depth_ppm", "period_days", "planet_radius"]
        with pytest.raises(DataError) as excinfo:
            load_artifact(tmp_path, "toy", expected_feature_names=live)
        assert excinfo.value.details["missing_in_model"] == ["planet_radius"]
        assert excinfo.value.details["extra_in_model"] == ["radius_earth"]

    def test_a_changed_label_contract_is_a_config_error(self, tmp_path: Path) -> None:
        save(tmp_path)
        with pytest.raises(ConfigError, match="label contract changed") as excinfo:
            load_artifact(
                tmp_path,
                "toy",
                expected_feature_names=FEATURES,
                expected_labels=["NO", "YES"],
            )
        assert excinfo.value.details["model_labels"] == LABELS

    def test_the_label_check_only_runs_when_the_caller_asks(self, tmp_path: Path) -> None:
        save(tmp_path)
        loaded = load_artifact(
            tmp_path, "toy", expected_feature_names=FEATURES, expected_labels=LABELS
        )
        assert loaded.labels == tuple(LABELS)

    def test_metadata_without_a_binary_is_not_trained(self, tmp_path: Path) -> None:
        save(tmp_path)
        model_paths(tmp_path, "toy")[0].unlink()
        with pytest.raises(ModelNotTrainedError, match="pipeline file is missing"):
            load_artifact(tmp_path, "toy", expected_feature_names=FEATURES)

    def test_a_binary_that_is_not_a_joblib_is_a_data_error(self, tmp_path: Path) -> None:
        save(tmp_path)
        binary = model_paths(tmp_path, "toy")[0]
        binary.write_bytes(b"this is a pickle-shaped lie")
        with pytest.raises(DataError, match="could not load trained pipeline") as excinfo:
            load_artifact(tmp_path, "toy", expected_feature_names=FEATURES)
        assert excinfo.value.details["path"].endswith(MODEL_SUFFIX)

    def test_a_sidecar_that_is_not_json_raises_the_decoder_not_a_domain_error(
        self, tmp_path: Path
    ) -> None:
        """Pinned as-is: `load_metadata` trusts that it wrote the file itself.

        A hand-edited or truncated sidecar therefore surfaces as `json.JSONDecodeError`
        rather than `DataError`. That is survivable - `api.deps.initialise` catches every
        exception at boot and prints the reason - but it is a rough edge, so it is recorded
        here instead of quietly papered over.
        """
        save(tmp_path)
        model_paths(tmp_path, "toy")[1].write_text("{nope", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_artifact(tmp_path, "toy", expected_feature_names=FEATURES)

    def test_a_different_sklearn_major_version_is_refused(self, tmp_path: Path) -> None:
        """joblib round-trips across minors; majors are where unpickling starts to bite."""
        save(tmp_path)
        _, meta_path = model_paths(tmp_path, "toy")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        metadata["environment"]["scikit_learn"] = "0.22.1"
        meta_path.write_text(json.dumps(metadata), encoding="utf-8")
        with pytest.raises(DataError, match="Major-version unpickling is not safe") as excinfo:
            load_artifact(tmp_path, "toy", expected_feature_names=FEATURES)
        assert excinfo.value.details["trained_with"] == "0.22.1"

    def test_a_different_sklearn_minor_version_is_allowed(self, tmp_path: Path) -> None:
        installed = registry.sklearn.__version__
        major = installed.split(".")[0]
        save(tmp_path)
        _, meta_path = model_paths(tmp_path, "toy")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        metadata["environment"]["scikit_learn"] = f"{major}.0.0"
        meta_path.write_text(json.dumps(metadata), encoding="utf-8")
        loaded = load_artifact(tmp_path, "toy", expected_feature_names=FEATURES)
        assert loaded.name == "toy"

    def test_an_unreadable_sklearn_version_fails_closed(self, tmp_path: Path) -> None:
        """`environment.scikit_learn == "unknown"` blocks the load rather than passing.

        Deliberate: the comparison is a string prefix check, so a missing version cannot be
        proven compatible and the guard errs toward refusing. The API survives it anyway -
        `initialise()` catches every exception at boot and reports the reason on `/health` -
        so the cost is a degraded service with an explanation, not a crash.
        """
        save(tmp_path)
        _, meta_path = model_paths(tmp_path, "toy")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        metadata["environment"]["scikit_learn"] = "unknown"
        meta_path.write_text(json.dumps(metadata), encoding="utf-8")
        with pytest.raises(DataError, match="Major-version unpickling is not safe"):
            load_artifact(tmp_path, "toy", expected_feature_names=FEATURES)

    def test_a_saved_pipeline_can_be_reloaded_from_the_file_alone(self, tmp_path: Path) -> None:
        binary = tmp_path / "raw.joblib"
        joblib.dump(make_estimator(), binary)
        estimator = joblib.load(binary)
        assert hasattr(estimator, "predict_proba")


class TestRealArtifacts:
    """Train/serve skew, tested against the files that actually ship in the repository."""

    def test_committed_exoplanet_model_matches_the_live_feature_list(
        self, config: Any, cache: Any
    ) -> None:
        binary = config.paths.model_dir / f"exoplanet_disposition{MODEL_SUFFIX}"
        if not binary.exists():
            pytest.skip("no trained exoplanet model in this checkout; run `nasa-space-ml train`")
        loaded = load_artifact(
            config.paths.model_dir,
            "exoplanet_disposition",
            expected_feature_names=list(exoplanet_features.FEATURE_COLUMNS),
            expected_labels=list(config.exoplanets.labels),
        )
        assert loaded.metadata["task"] == "exoplanets"
        assert loaded.metadata["labels"] == list(config.exoplanets.labels)
        assert loaded.metadata["n_features"] == len(exoplanet_features.FEATURE_COLUMNS)

    def test_committed_asteroid_model_matches_the_live_feature_list(
        self, config: Any, cache: Any
    ) -> None:
        binary = config.paths.model_dir / f"asteroid_pha{MODEL_SUFFIX}"
        if not binary.exists():
            pytest.skip("no trained asteroid model in this checkout; run `nasa-space-ml train`")
        loaded = load_artifact(
            config.paths.model_dir,
            "asteroid_pha",
            expected_feature_names=list(asteroid_features.FEATURE_COLUMNS),
            expected_labels=list(config.asteroids.labels),
        )
        assert loaded.metadata["task"] == "asteroids"
        assert loaded.metadata["labels"] == list(config.asteroids.labels)
        assert loaded.metadata["n_features"] == len(asteroid_features.FEATURE_COLUMNS)

    def test_committed_models_record_the_fingerprint_of_the_data_they_ship_with(self) -> None:
        """`reports/*/metrics.json` and `models/*.json` must describe the same run."""
        model_dir = Path(registry.__file__).resolve().parents[3] / "models"
        for name in ("exoplanet_disposition", "asteroid_pha"):
            sidecar = model_dir / f"{name}{META_SUFFIX}"
            if not sidecar.exists():
                pytest.skip(f"{name} is not trained in this checkout")
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            assert len(metadata["dataset_fingerprint"]) == 64
            assert metadata["rows"] > 0 and metadata["groups"] > 0
