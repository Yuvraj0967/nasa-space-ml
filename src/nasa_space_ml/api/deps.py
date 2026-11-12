"""Application state, model loading, and the shared predict path.

Two properties this file is trying to hold:

  * **Boot must not fail because a model is missing.** The service comes up, `/api/v1/meta`
    reports exactly which model is absent and why, and a prediction request returns a 503
    telling the operator to run `nasa-space-ml train`. A service that refuses to start gives
    nobody information.
  * **Serving must use the training code, not a copy of it.** The feature vector is built by
    the same `derive()` the pipeline used, in the same column order, and then handed to
    `registry.load_artifact`, which refuses any model whose recorded feature list differs.
    That single guard is what makes train/serve skew a loud startup error instead of a quiet
    stream of wrong answers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from ..data import asteroids as asteroid_data
from ..data.cache import Cache
from ..errors import InvalidFeaturesError, ModelNotTrainedError
from ..features import asteroids as asteroid_features
from ..features import exoplanets as exoplanet_features
from ..models import registry

EXOPLANET_MODEL = "exoplanet_disposition"
ASTEROID_MODEL = "asteroid_pha"


@dataclass
class AppState:
    config: Config
    cache: Cache
    started_utc: str
    models: dict[str, Any] = field(default_factory=dict)
    model_errors: dict[str, str] = field(default_factory=dict)
    #: (dataset_fingerprint, events, per-object aggregation). See `asteroid_tables`.
    _asteroid_tables: tuple[str, pd.DataFrame, pd.DataFrame] | None = field(
        default=None, repr=False
    )

    def asteroid_tables(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Close-approach events and the one-row-per-object aggregation, cached per dataset.

        The aggregation is the expensive part: 32,390 events grouped into 22,581 objects,
        each summarised in Python, measured at ~13 s. Recomputing it for every `/sample`
        click made the front end's main button take fifteen seconds to answer, so the result
        is memoised against `dataset_fingerprint` - the same content hash the training run is
        stamped with. A new fetch changes the hash, which is the only invalidation rule that
        can be right here: the tables are a pure function of (cached bytes, config), and the
        config cannot change without a restart.

        Both frames are handed out read-only; callers filter with boolean masks, which
        produces copies rather than views, so two concurrent requests cannot corrupt it.
        """
        fingerprint = self.cache.dataset_fingerprint()
        if self._asteroid_tables is None or self._asteroid_tables[0] != fingerprint:
            events = asteroid_data.load_events(self.config, self.cache)
            objects = asteroid_features.aggregate_events(events, self.config)
            self._asteroid_tables = (fingerprint, events, objects)
        return self._asteroid_tables[1], self._asteroid_tables[2]

    def describe_model(self, name: str) -> dict[str, Any]:
        loaded = self.models.get(name)
        if loaded is None:
            return {
                "name": name,
                "loaded": False,
                "reason": self.model_errors.get(name, "not loaded"),
            }
        metadata = loaded.metadata
        return {
            "name": name,
            "loaded": True,
            "task": metadata["task"],
            "algorithm": metadata["algorithm"],
            "trained_utc": metadata["trained_utc"],
            "labels": metadata["labels"],
            "selection_metric": metadata.get("metrics", {}).get(
                "macro_f1" if metadata["task"] == "exoplanets" else "average_precision"
            ),
            "n_features": metadata["n_features"],
            "feature_hash": metadata["feature_hash"],
            "dataset_fingerprint": metadata["dataset_fingerprint"],
            "rows": metadata["rows"],
            "groups": metadata["groups"],
            "scikit_learn": (metadata.get("environment") or {}).get("scikit_learn"),
        }


def expected_features(task: str, config: Config) -> list[str]:
    """The live feature contract, read from the same modules training reads."""
    if task == "exoplanets":
        return list(exoplanet_features.FEATURE_COLUMNS)
    if task == "asteroids":
        return list(asteroid_features.FEATURE_COLUMNS)
    raise InvalidFeaturesError(f"unknown task {task!r}")


def initialise(config: Config, cache: Cache) -> AppState:
    """Load both models, keeping going if either is missing."""
    state = AppState(
        config=config,
        cache=cache,
        started_utc=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    pairs = {
        EXOPLANET_MODEL: ("exoplanets", list(config.exoplanets.labels)),
        ASTEROID_MODEL: ("asteroids", list(config.asteroids.labels)),
    }
    for name, (task, labels) in pairs.items():
        try:
            state.models[name] = registry.load_artifact(
                config.paths.model_dir,
                name,
                expected_feature_names=expected_features(task, config),
                expected_labels=labels,
            )
        except Exception as exc:
            state.model_errors[name] = f"{type(exc).__name__}: {exc}"
    return state


def require_model(state: AppState, name: str) -> registry.LoadedModel:
    loaded = state.models.get(name)
    if loaded is None:
        raise ModelNotTrainedError(
            f"model {name!r} is not loaded ({state.model_errors.get(name, 'unknown reason')}); "
            "run `nasa-space-ml train` and restart the service",
            details={"model": name},
        )
    return loaded


def _clean_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _features_dict(matrix: pd.DataFrame) -> dict[str, float | None]:
    row = matrix.iloc[0]
    return {column: _clean_float(row[column]) for column in matrix.columns}


def outside_training_range(
    matrix: pd.DataFrame,
    ranges: dict[str, Any] | None,
) -> list[str]:
    """Feature names whose submitted value falls outside the training min/max.

    Reported, never blocked. Extrapolation is the legitimate reason to call a model at all;
    the service's job is to make it visible, not to forbid it.
    """
    if not ranges:
        return []
    offenders: list[str] = []
    row = matrix.iloc[0]
    for column in matrix.columns:
        bounds = ranges.get(column)
        if not bounds:
            continue
        value = _clean_float(row[column])
        if value is None:
            continue
        low, high = float(bounds[0]), float(bounds[1])
        if value < low or value > high:
            offenders.append(column)
    return offenders


def _ordered_matrix(built: pd.DataFrame, feature_names: tuple[str, ...]) -> pd.DataFrame:
    """Reindex the built row onto the model's recorded column order.

    Wrong order is the quietest possible failure mode: the shapes match, nothing raises, and
    every prediction is nonsense. Reindexing against the artifact's own list makes the model
    file the authority, and a genuinely missing column becomes a KeyError we can see.
    """
    missing = [name for name in feature_names if name not in built.columns]
    if missing:
        raise InvalidFeaturesError(
            "the feature builder did not produce columns the model expects",
            details={"missing": missing},
        )
    return built[list(feature_names)]


def predict_proba(task: str, payload: dict[str, Any], state: AppState) -> dict[str, Any]:
    """Score one request. Returns probabilities plus everything needed to reproduce it."""
    model_name = EXOPLANET_MODEL if task == "exoplanets" else ASTEROID_MODEL
    loaded = require_model(state, model_name)
    config = state.config

    if task == "exoplanets":
        raw = exoplanet_features.frame_from_payload(payload)
        derived = exoplanet_features.derive(raw)
        matrix = _ordered_matrix(derived, loaded.feature_names)
        raw_echo = {
            column: _clean_float(raw.iloc[0][column])
            for column in exoplanet_features.RAW_INPUT_COLUMNS
        }
    else:
        frame = asteroid_features.frame_from_payload(payload, config)
        derived = asteroid_features.derive(frame)
        matrix = _ordered_matrix(derived, loaded.feature_names)
        raw_echo = {
            column: _clean_float(frame.iloc[0][column])
            for column in asteroid_features.BASE_INPUT_COLUMNS
        }

    probabilities = np.asarray(loaded.estimator.predict_proba(matrix))[0]
    classes = np.asarray(loaded.estimator.classes_)
    order = int(np.argmax(probabilities))

    scored = _features_dict(matrix)
    explanation = build_explanation(task, classes, probabilities, order, loaded.metadata)

    return {
        "task": task,
        "model": loaded.name,
        "prediction": str(classes[order]),
        "probabilities": [
            {"label": str(label), "probability": float(value)}
            for label, value in zip(classes, probabilities, strict=True)
        ],
        "features_used": raw_echo,
        "derived_features": scored,
        "model_trained_utc": loaded.metadata["trained_utc"],
        "dataset_fingerprint": loaded.metadata["dataset_fingerprint"],
        "scikit_learn": (loaded.metadata.get("environment") or {}).get("scikit_learn"),
        "outside_training_range": outside_training_range(
            matrix, loaded.metadata.get("feature_ranges")
        ),
        "feature_ranges_available": bool(loaded.metadata.get("feature_ranges")),
        "classes_order": [str(label) for label in classes],
        "explanation": explanation,
    }


def build_explanation(
    task: str,
    classes: np.ndarray,
    probabilities: np.ndarray,
    order: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Say, in words, what the numbers mean - and what this model is not.

    A probability of 0.94 for CONFIRMED on a Kepler candidate is not a 94% chance the planet
    exists: it is the model's calibrated score under a grouped split on one catalogue. The
    front end renders this block verbatim so the disclaimer cannot be separated from the
    answer.
    """
    top = float(probabilities[order])
    runner_up = float(sorted(probabilities)[-2]) if len(probabilities) > 1 else 0.0
    if top - runner_up < 0.05:
        confidence = "marginal: the two leading classes are within 5 percentage points"
    elif top < 0.5 and len(probabilities) > 2:
        confidence = "weak: no class reaches 50%"
    else:
        confidence = "clear"

    notes = [
        f"selected model: {metadata.get('algorithm')}",
        f"headline {task} metric: {_headline(task, metadata)}",
        "probabilities are the model's own output, not measurement error bars",
        "splits were grouped by host star / designation, so this is not a random-row score",
    ]
    if task == "asteroids":
        notes.append(
            "this is a classifier over JPL's published PHA definition; it is not a hazard "
            "assessment and must not be used for conjunction screening"
        )
    return {
        "confidence": confidence,
        "margin": top - runner_up,
        "notes": notes,
    }


def _headline(task: str, metadata: dict[str, Any]) -> str:
    metrics = metadata.get("metrics") or {}
    key = "macro_f1" if task == "exoplanets" else "average_precision"
    value = metrics.get(key)
    if value is None:
        return "n/a"
    return f"{key}={value:.4f}"
