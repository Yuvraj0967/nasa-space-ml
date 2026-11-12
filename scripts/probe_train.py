"""Smoke-check the training stack on a subsample before committing to a full run.

Verifies, on the real cached data:
  * HistGradientBoostingClassifier really accepts class_weight in the installed sklearn
  * LogisticRegression does NOT get the removed `multi_class` kwarg
  * StratifiedGroupKFold runs on both label sets without a "group appears in both folds"
  * the rule baselines satisfy the sklearn protocol enough to go through the same metric code
  * binary_metrics / multiclass_metrics return every key the report renderer expects
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from nasa_space_ml.config import Config
from nasa_space_ml.data.cache import Cache
from nasa_space_ml.evaluation.metrics import selection_metric
from nasa_space_ml.models.splits import assert_no_group_leakage, grouped_holdout
from nasa_space_ml.models.train import (
    ASTEROID_TASK,
    EXOPLANET_TASK,
    _score_predictions,
    asteroid_candidates,
    cross_validate_candidates,
    exoplanet_candidates,
    load_asteroid_matrices,
    load_exoplanet_matrices,
)


def subsample(X, y, groups, n=1200):
    step = max(1, len(X) // n)
    keep = np.arange(0, len(X), step)[:n]
    return (
        X.iloc[keep].reset_index(drop=True),
        y.iloc[keep].reset_index(drop=True),
        groups.iloc[keep].reset_index(drop=True),
    )


def run(task: str, config: Config, cache: Cache) -> None:
    if task == EXOPLANET_TASK:
        X, y, groups = load_exoplanet_matrices(config, cache)
        labels = list(config.exoplanets.labels)
        candidates = exoplanet_candidates(config)
    else:
        X, y, groups = load_asteroid_matrices(config, cache)
        labels = list(config.asteroids.labels)
        candidates = asteroid_candidates(config)

    X, y, groups = subsample(X, y, groups)
    print(
        f"[{task}] subsample rows={len(X)} groups={groups.nunique()} classes={y.value_counts().to_dict()}"
    )

    holdout = grouped_holdout(y, groups, config)
    assert_no_group_leakage(
        ("train", groups.iloc[holdout.train_index]),
        ("val", groups.iloc[holdout.val_index]),
        ("test", groups.iloc[holdout.test_index]),
    )
    print(f"[{task}] holdout sizes={holdout.sizes()}")

    results = cross_validate_candidates(
        X.iloc[holdout.train_index],
        y.iloc[holdout.train_index],
        groups.iloc[holdout.train_index],
        candidates,
        config,
        task,
        labels,
    )
    for item in results:
        print(
            f"[{task}]   {item.name:26s} {item.kind:8s} cv={item.cv_mean:.4f} +-{item.cv_std:.4f} {item.fold_scores}"
        )

    name = max(results, key=lambda item: item.cv_mean).name
    estimator = candidates[name][1]
    estimator.fit(X, y)
    metrics = _score_predictions(
        task,
        y.to_numpy(),
        np.asarray(estimator.predict_proba(X)),
        np.asarray(estimator.predict(X)),
        labels,
        class_order=getattr(estimator, "classes_", None),
    )
    print(f"[{task}] winner-in-sample {name}: keys={sorted(metrics)}")
    print(f"[{task}] selection {selection_metric(metrics, task):.4f}")
    if task == ASTEROID_TASK:
        print(f"[{task}] operating points: {metrics['operating_points']}")
    print(f"[{task}] sklearn classes_ = {list(getattr(estimator, 'classes_', []))}")


def main() -> int:
    config = Config.load()
    cache = Cache(config.paths.raw_dir)
    run(EXOPLANET_TASK, config, cache)
    run(ASTEROID_TASK, config, cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
