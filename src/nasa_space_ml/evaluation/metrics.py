"""Metric computation, in one place, used identically by rules and by learned models.

Why a module instead of inline `f1_score(...)` calls: the whole credibility of the
rule-vs-ML comparison rests on both sides being scored by the same function. A baseline
that gets its own metric code is a baseline that can lose by arithmetic.

Two selection metrics, chosen for the problem each task actually has:

  * Exoplanets - **macro-F1**. Three classes with a 2.9:1 majority, and all three matter:
    a model that only ever says FALSE POSITIVE is accurate and useless. Macro-F1 weights
    the minority CANDIDATE class exactly as heavily as the majority.
  * Asteroids - **average precision (PR-AUC)** on the PHA class. With 5.3% positives,
    ROC-AUC is flattering: a model that ranks 10% of the harmless catalogue above half the
    hazards still looks decent, because 95% of the population is genuinely boring. For an
    imbalance this strong, the precision-recall plane is where the honest signal lives.

`brier` and the calibration curve are reported but never used for selection: a model can be
perfectly calibrated and useless at ranking, and we do not want that trade-off hidden.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _one_hot(y_true: np.ndarray, classes: np.ndarray) -> np.ndarray:
    index = {label: position for position, label in enumerate(classes)}
    matrix = np.zeros((len(y_true), len(classes)), dtype=float)
    for row, label in enumerate(y_true):
        matrix[row, index[label]] = 1.0
    return matrix


def multiclass_brier(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> float:
    """Mean squared error between the predicted probability vector and the one-hot truth."""
    return float(np.mean((proba - _one_hot(np.asarray(y_true), classes)) ** 2))


def per_class_block(
    y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray
) -> dict[str, dict[str, float]]:
    """Per-class precision / recall / F1 / support, from one vectorised call each.

    `average=None` is not decoration: asking for `labels=[one_class]` with the default
    `average="binary"` makes scikit-learn raise as soon as the problem has three classes.
    `zero_division=0` is the documented answer to "precision of a class nothing predicted",
    which happens in small folds and must not abort a training run.
    """
    label_list = [str(label) for label in classes]
    precision = precision_score(y_true, y_pred, labels=label_list, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=label_list, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=label_list, average=None, zero_division=0)
    truth = np.asarray(y_true)
    return {
        label: {
            "precision": float(precision[position]),
            "recall": float(recall[position]),
            "f1": float(f1[position]),
            "support": int(np.sum(truth == label)),
        }
        for position, label in enumerate(label_list)
    }


def confusion_block(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, y_pred, labels=classes)
    return {
        "labels": [str(label) for label in classes],
        "matrix": [[int(value) for value in row] for row in matrix],
    }


def multiclass_metrics(
    y_true: pd.Series | np.ndarray,
    proba: np.ndarray,
    y_pred: pd.Series | np.ndarray,
    classes: np.ndarray,
) -> dict[str, Any]:
    """Everything we report for the 3-class disposition task."""
    truth = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    probabilities = np.asarray(proba, dtype=float)

    metrics: dict[str, Any] = {
        "macro_f1": float(f1_score(truth, predicted, average="macro", labels=list(classes))),
        "weighted_f1": float(f1_score(truth, predicted, average="weighted", labels=list(classes))),
        "accuracy": float(np.mean(predicted == truth)),
        "log_loss": float(log_loss(truth, probabilities, labels=list(classes))),
        "brier": multiclass_brier(truth, probabilities, classes),
        "per_class": per_class_block(truth, predicted, classes),
        "confusion": confusion_block(truth, predicted, classes),
    }
    if len(classes) > 2:
        try:
            metrics["roc_auc_ovr"] = float(roc_auc_score(truth, probabilities, multi_class="ovr"))
        except (ValueError, TypeError):  # pragma: no cover - degenerate fold
            metrics["roc_auc_ovr"] = None
        try:
            indicator = _one_hot(truth, classes)
            per_label = [
                roc_auc_score(indicator[:, column], probabilities[:, column])
                for column in range(len(classes))
                if 0 < indicator[:, column].sum() < len(indicator)
            ]
            metrics["roc_auc_macro"] = float(np.mean(per_label)) if per_label else None
        except ValueError:  # pragma: no cover - a class absent from the fold
            metrics["roc_auc_macro"] = None
    return metrics


def _binary_scores(truth: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """Accept a probability matrix or a 1-D positive-class score and return the latter."""
    array = np.asarray(proba, dtype=float)
    if array.ndim == 2:
        return array[:, -1]
    return array


def binary_metrics(
    y_true: pd.Series | np.ndarray,
    proba: np.ndarray,
    y_pred: pd.Series | np.ndarray,
    *,
    positive: str,
    negative: str,
) -> dict[str, Any]:
    """Everything we report for the PHA task. `average_precision` is the selection metric."""
    truth_binary = (np.asarray(y_true) == positive).astype(int)
    predicted_binary = (np.asarray(y_pred) == positive).astype(int)
    score = _binary_scores(truth_binary, proba)

    metrics: dict[str, Any] = {
        "average_precision": float(average_precision_score(truth_binary, score)),
        "roc_auc": float(roc_auc_score(truth_binary, score)),
        "brier": float(np.mean((score - truth_binary) ** 2)),
        "macro_f1": float(f1_score(truth_binary, predicted_binary, average="macro")),
        "precision_positive": float(
            precision_score(truth_binary, predicted_binary, zero_division=0)
        ),
        "recall_positive": float(recall_score(truth_binary, predicted_binary, zero_division=0)),
        "f1_positive": float(f1_score(truth_binary, predicted_binary, zero_division=0)),
        "prevalence": float(truth_binary.mean()),
        "per_class": per_class_block(
            np.where(truth_binary == 1, positive, negative),
            np.where(predicted_binary == 1, positive, negative),
            np.array([negative, positive]),
        ),
        "confusion": confusion_block(
            np.where(truth_binary == 1, positive, negative),
            np.where(predicted_binary == 1, positive, negative),
            np.array([negative, positive]),
        ),
    }
    metrics["operating_points"] = operating_points(truth_binary, score)
    return metrics


def operating_points(
    truth_binary: np.ndarray,
    score: np.ndarray,
    *,
    target_precisions: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75),
) -> list[dict[str, float]]:
    """Threshold, precision and recall at each requested precision level.

    This is the table an actual reviewer asks for: "at what cut-off do I get 50% of my
    alerts being real, and how many hazards does that miss?" A single 0.5 threshold cannot
    answer that, and pretending otherwise is how dashboards end up ignoring real alerts.
    """
    precision, recall, thresholds = precision_recall_curve(truth_binary, score)
    rows: list[dict[str, float]] = []
    for target in target_precisions:
        candidates = [
            (index, precision[index], recall[index])
            for index in range(len(precision))
            if precision[index] >= target
        ]
        if not candidates:
            rows.append(
                {"target_precision": target, "threshold": None, "precision": None, "recall": None}
            )
            continue
        # Highest recall among the points that clear the requested precision.
        index, precision_value, recall_value = max(candidates, key=lambda item: item[2])
        threshold = float(thousands_safe_threshold(thresholds, index))
        rows.append(
            {
                "target_precision": target,
                "threshold": threshold,
                "precision": float(precision_value),
                "recall": float(recall_value),
            }
        )
    return rows


def thousands_safe_threshold(thresholds: np.ndarray, index: int) -> float:
    """Map a precision-recall curve index onto its threshold, clamped at the array edges.

    `precision_recall_curve` returns arrays one element longer than `thresholds`, because
    the final operating point is "predict positive for everything".
    """
    if len(thresholds) == 0:
        return 1.0
    position = min(index, len(thresholds) - 1)
    return float(thresholds[position])


def calibration_block(
    truth_binary: np.ndarray,
    score: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> dict[str, Any]:
    """Reliability data, straight from sklearn's calibration_curve."""
    from sklearn.calibration import calibration_curve

    fraction_positive, mean_predicted = calibration_curve(
        truth_binary, score, n_bins=n_bins, strategy=strategy
    )
    return {
        "n_bins": n_bins,
        "strategy": strategy,
        "mean_predicted": [float(value) for value in mean_predicted],
        "fraction_positive": [float(value) for value in fraction_positive],
    }


def build_scorer(scorer_name: str, *, pos_label: str | None = None) -> Any:
    """Turn a metric name into an sklearn scorer, stating the positive label out loud.

    This exists because `get_scorer("average_precision")` is a trap. It builds the scorer
    with the default `pos_label=1`, and our labels are the strings "PHA" and "NOT_PHA", so
    sklearn raises `ValueError: pos_label=1 is not a valid label` before it has permuted a
    single column. Any metric that ranks one class against another has to be told which
    class it is ranking; with integer labels 1 happens to be right and the bug hides, with
    string labels it surfaces. That is a good argument for string labels.
    """
    from sklearn.metrics import get_scorer, make_scorer

    if scorer_name in {"average_precision", "roc_auc"}:
        if pos_label is None:
            raise ValueError(
                f"{scorer_name!r} ranks one class against the other, so it needs an explicit "
                "pos_label; pass the positive class name from the config"
            )
        metric = average_precision_score if scorer_name == "average_precision" else roc_auc_score
        return make_scorer(metric, pos_label=pos_label, response_method="predict_proba")
    return get_scorer(scorer_name)


def permutation_importance_block(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    scorer_name: str,
    *,
    pos_label: str | None = None,
    n_repeats: int = 8,
    seed: int = 0,
) -> dict[str, Any]:
    """Feature importance by permutation - the honest version of the tree-importance numbers.

    Tree impurity importances are biased toward high-cardinality and continuous features
    and are measured on the training set. Permutation importance asks the question we
    actually care about: if I scramble this column on held-out data, how much worse do we
    get? Negative values are real and are reported as-is.
    """
    from sklearn.inspection import permutation_importance

    scoring = build_scorer(scorer_name, pos_label=pos_label)
    result = permutation_importance(
        model, X, y, scoring=scoring, n_repeats=n_repeats, random_state=seed, n_jobs=1
    )
    rows = [
        {
            "feature": name,
            "mean_decrease": float(mean),
            "std_decrease": float(std),
            "null_importance": bool(_all_nan(X[name])),
        }
        for name, mean, std in zip(
            X.columns, result["importances_mean"], result["importances_std"], strict=True
        )
    ]
    rows.sort(key=lambda row: row["mean_decrease"], reverse=True)
    return {
        "scorer": scorer_name,
        "pos_label": pos_label,
        "n_repeats": n_repeats,
        "features": rows,
    }


def _all_nan(series: pd.Series) -> bool:
    return bool(pd.isna(series).all())


def selection_metric(metrics: dict[str, Any], task: str) -> float:
    """Pull the one number a candidate is ranked on, per task."""
    key = {"exoplanets": "macro_f1", "asteroids": "average_precision"}[task]
    value = metrics.get(key)
    if value is None:
        raise ValueError(f"metrics for {task!r} have no {key!r}")
    return float(value)


def fold_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Small helper so training can print a tidy per-fold summary without pandas gymnastics."""
    return pd.DataFrame(rows)
