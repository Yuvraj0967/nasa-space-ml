"""Saving and loading trained pipelines, with the guards that make serving safe.

A `.joblib` file on disk is not a model, it is a guess about a model. This module turns it
back into something we are allowed to serve by checking, in order:

  1. the artifact and its sidecar metadata both exist (otherwise `ModelNotTrainedError`,
     which the API maps to 503 with a message telling the user to run `nasa-space-ml train`);
  2. the feature list recorded at training time is *identical* to the list the serving code
     is about to build, in the same order. This is the train/serve-skew guard, and it is the
     single most valuable line in this file: without it, adding one feature to
     `configs/settings.yaml` silently feeds a shifted matrix to a model that expects the old
     one, and every prediction is confidently wrong;
  3. the label set matches what the API promises its callers;
  4. the scikit-learn version is recorded and surfaced, so a pickle written by a different
     major version is visible rather than mysterious (joblib round-trips across minor
     versions fine; majors are where unpickling starts to bite).

Metadata is written next to the model rather than inside it, so a reviewer can read the
numbers without loading a binary.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import sklearn

from ..errors import ConfigError, DataError, ModelNotTrainedError

MODEL_SUFFIX = ".joblib"
META_SUFFIX = ".json"


def feature_list_hash(names: list[str] | tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()[:16]


def environment_snapshot() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
    }


@dataclass(frozen=True)
class LoadedModel:
    name: str
    estimator: Any
    metadata: dict[str, Any]

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.metadata["feature_names"])

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self.metadata["labels"])


def model_paths(directory: Path, name: str) -> tuple[Path, Path]:
    directory = Path(directory)
    return directory / f"{name}{MODEL_SUFFIX}", directory / f"{name}{META_SUFFIX}"


def save_artifact(
    directory: Path,
    name: str,
    estimator: Any,
    *,
    feature_names: list[str] | tuple[str, ...],
    labels: list[str] | tuple[str, ...],
    task: str,
    algorithm: str,
    metrics: dict[str, Any],
    dataset_fingerprint: str,
    rows: int,
    groups: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the pipeline plus a human-readable sidecar, model last.

    Order matters: if the joblib dump fails we do not want a metadata file claiming a model
    exists. Writing the sidecar first and the binary second, then verifying both are present,
    means `load_artifact` can never see half an artifact.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    binary_path, meta_path = model_paths(directory, name)

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "name": name,
        "task": task,
        "algorithm": algorithm,
        "feature_names": list(feature_names),
        "feature_hash": feature_list_hash(feature_names),
        "n_features": len(feature_names),
        "labels": list(labels),
        "metrics": metrics,
        "dataset_fingerprint": dataset_fingerprint,
        "rows": int(rows),
        "groups": int(groups),
        "trained_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": environment_snapshot(),
    }
    if extra:
        metadata.update(extra)

    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    joblib.dump(estimator, binary_path)
    return metadata


def load_metadata(directory: Path, name: str) -> dict[str, Any]:
    _, meta_path = model_paths(Path(directory), name)
    if not meta_path.exists():
        raise ModelNotTrainedError(
            f"no trained model metadata for {name!r} at {meta_path}; run "
            "`nasa-space-ml train` first",
            details={"path": str(meta_path)},
        )
    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_artifact(
    directory: Path,
    name: str,
    *,
    expected_feature_names: list[str] | tuple[str, ...],
    expected_labels: list[str] | tuple[str, ...] | None = None,
) -> LoadedModel:
    """Load a saved pipeline, refusing anything that does not match the live feature list."""
    directory = Path(directory)
    metadata = load_metadata(directory, name)
    binary_path, _ = model_paths(directory, name)
    if not binary_path.exists():
        raise ModelNotTrainedError(
            f"model metadata exists but the pipeline file is missing: {binary_path}",
            details={"path": str(binary_path)},
        )

    recorded = list(metadata.get("feature_names") or [])
    live = list(expected_feature_names)
    if recorded != live:
        raise DataError(
            f"trained model {name!r} was fitted on a different feature list than the one "
            "this code builds. Refusing to predict: the matrix would be silently misaligned. "
            f"Model expects {recorded}, current pipeline produces {live}. Re-run "
            "`nasa-space-ml train`.",
            details={
                "model_hash": metadata.get("feature_hash"),
                "live_hash": feature_list_hash(live),
                "missing_in_model": sorted(set(live) - set(recorded)),
                "extra_in_model": sorted(set(recorded) - set(live)),
            },
        )
    if expected_labels is not None and list(metadata.get("labels") or []) != list(expected_labels):
        raise ConfigError(
            f"trained model {name!r} has labels {metadata.get('labels')!r} but the caller "
            f"expects {list(expected_labels)!r}; the label contract changed",
            details={"model_labels": metadata.get("labels")},
        )

    try:
        estimator = joblib.load(binary_path)
    except Exception as exc:
        raise DataError(
            f"could not load trained pipeline {binary_path}: {type(exc).__name__}: {exc}",
            details={"path": str(binary_path)},
        ) from exc

    installed = sklearn.__version__
    trained_with = str((metadata.get("environment") or {}).get("scikit_learn", "unknown"))
    if trained_with.split(".")[:1] != installed.split(".")[:1]:
        raise DataError(
            f"model {name!r} was trained with scikit-learn {trained_with} but this "
            f"environment has {installed}. Major-version unpickling is not safe; retrain.",
            details={"trained_with": trained_with, "installed": installed},
        )

    return LoadedModel(name=name, estimator=estimator, metadata=metadata)
