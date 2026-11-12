"""Report generation: figures, machine-readable metrics, and a markdown summary.

Every figure here exists to answer a question a reviewer would otherwise ask in writing:

  * confusion matrix - which disposition does the model confuse with which?
  * PR curve vs ROC curve - why we reported PR-AUC and not the prettier number
  * reliability curve - can the probability be *used*, not just ranked by
  * permutation importance - what is actually carrying the signal, including the features
    that carry nothing (drawn, not omitted)
  * candidate comparison with error bars - how big is the model-selection margin compared to
    the fold-to-fold noise

Figures go to `reports/<task>/`, metrics to `reports/<task>/metrics.json`, and the markdown
next to them. Nothing in here calls the network, and the module is import-safe without a
display because the Agg backend is set before pyplot is touched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from ..models.train import EXOPLANET_TASK, TrainingResult

FIGURE_DPI = 140

DARK = "#10151c"
PANEL = "#1b2430"
INK = "#e6edf3"
ACCENT = "#4fd1c5"
WARN = "#f6ad55"
MUTED = "#8b98a8"


def _style(axis: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    axis.set_facecolor(PANEL)
    axis.set_title(title, color=INK, fontsize=11, pad=10)
    axis.set_xlabel(xlabel, color=MUTED, fontsize=9)
    axis.set_ylabel(ylabel, color=MUTED, fontsize=9)
    axis.tick_params(colors=MUTED, labelsize=8)
    for spine in axis.spines.values():
        spine.set_color("#2b3644")
    axis.grid(True, color="#2b3644", linewidth=0.6, alpha=0.7)


def _figure(task_dir: Path, name: str) -> tuple[plt.Figure, Path]:
    figure = plt.figure(figsize=(6.4, 4.6), dpi=FIGURE_DPI)
    figure.patch.set_facecolor(DARK)
    return figure, task_dir / name


def _save(figure: plt.Figure, path: Path) -> str:
    figure.savefig(path, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)
    return path.name


def plot_confusion_matrix(result: TrainingResult, task_dir: Path) -> str | None:
    matrix = np.asarray(result.test_metrics["confusion"]["matrix"], dtype=float)
    labels = result.test_metrics["confusion"]["labels"]
    figure, path = _figure(task_dir, "confusion_matrix.png")
    axis = figure.add_subplot(111)
    axis.imshow(matrix, cmap="viridis")
    axis.set_xticks(range(len(labels)), labels, color=INK, fontsize=8, rotation=15, ha="right")
    axis.set_yticks(range(len(labels)), labels, color=INK, fontsize=8)
    limit = matrix.max() or 1.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{int(matrix[row, column])}",
                ha="center",
                va="center",
                fontsize=9,
                color=INK if matrix[row, column] < 0.55 * limit else DARK,
            )
    _style(axis, "Disposition confusion on the held-out test fold", "predicted", "actual")
    axis.grid(False)
    return _save(figure, path)


def plot_candidate_comparison(result: TrainingResult, task_dir: Path) -> str | None:
    rows = [candidate.as_row() for candidate in result.candidates]
    names = [row["candidate"] for row in rows]
    means = [row["cv_mean"] for row in rows]
    stds = [row["cv_std"] for row in rows]
    figure, path = _figure(task_dir, "candidate_comparison.png")
    axis = figure.add_subplot(111)
    colours = [WARN if row["kind"] == "rule" else ACCENT for row in rows]
    axis.bar(
        range(len(names)),
        means,
        yerr=stds,
        color=colours,
        capsize=4,
        error_kw={"ecolor": MUTED, "elinewidth": 1.2, "capthick": 1.2},
    )
    axis.set_xticks(
        range(len(names)), [name.replace("_", "\n") for name in names], color=INK, fontsize=8
    )
    _style(
        axis,
        f"Cross-validated selection metric: {result.selection_metric_name}",
        "",
        result.selection_metric_name,
    )
    axis.set_ylim(0, min(1.0, max(means + stds) * 1.25))
    axis.text(
        0.0,
        -0.32,
        "amber = rule baseline, teal = learned model; bars are fold standard deviation",
        transform=axis.transAxes,
        color=MUTED,
        fontsize=7.5,
    )
    return _save(figure, path)


def plot_pr_vs_roc(
    result: TrainingResult, task_dir: Path, probabilities: np.ndarray, truth: np.ndarray
) -> str | None:
    score = np.asarray(probabilities, dtype=float)[:, -1]
    binary = (np.asarray(truth) == result.labels[1]).astype(int)
    precision, recall, _ = precision_recall_curve(binary, score)
    fpr, tpr, _ = roc_curve(binary, score)
    ap = average_precision_score(binary, score)
    auc = roc_auc_score(binary, score)

    figure, path = _figure(task_dir, "pr_vs_roc.png")
    left = figure.add_subplot(121)
    right = figure.add_subplot(122)
    left.plot(recall, precision, color=ACCENT, linewidth=1.8, label=f"PR-AUC = {ap:.3f}")
    left.fill_between(recall, 0, precision, color=ACCENT, alpha=0.15)
    left.axhline(
        binary.mean(),
        color=WARN,
        linestyle="--",
        linewidth=1.0,
        label=f"prevalence = {binary.mean():.3f}",
    )
    _style(left, "Precision-recall (reported)", "recall of hazardous objects", "precision")
    left.legend(frameon=False, fontsize=7, labelcolor=INK)

    right.plot(fpr, tpr, color=MUTED, linewidth=1.8, label=f"ROC-AUC = {auc:.3f}")
    right.plot([0, 1], [0, 1], color="#3a4757", linestyle=":", linewidth=1.0)
    _style(right, "ROC (shown for contrast only)", "false positive rate", "true positive rate")
    right.legend(frameon=False, fontsize=7, labelcolor=INK)
    return _save(figure, path)


def plot_calibration(
    result: TrainingResult, task_dir: Path, calibration: dict[str, Any]
) -> str | None:
    figure, path = _figure(task_dir, "calibration.png")
    axis = figure.add_subplot(111)
    predicted = np.asarray(calibration["mean_predicted"], dtype=float)
    observed = np.asarray(calibration["fraction_positive"], dtype=float)
    axis.plot([0, 1], [0, 1], color="#3a4757", linestyle=":", linewidth=1.0, label="perfect")
    axis.plot(
        predicted, observed, color=ACCENT, marker="o", markersize=4, linewidth=1.6, label="model"
    )
    _style(
        axis,
        "Reliability of the predicted probability",
        "mean predicted probability",
        "observed fraction hazardous",
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, max(1.0, observed.max() * 1.3 if len(observed) else 1.0))
    axis.legend(frameon=False, fontsize=7, labelcolor=INK)
    return _save(figure, path)


def plot_importance(result: TrainingResult, task_dir: Path) -> str | None:
    features = result.importance.get("features") or []
    if not features:
        return None
    top = features[:14][::-1]
    names = [row["feature"] for row in top]
    values = [row["mean_decrease"] for row in top]
    errors = [row["std_decrease"] for row in top]
    figure, path = _figure(task_dir, "permutation_importance.png")
    axis = figure.add_subplot(111)
    colours = [MUTED if row["null_importance"] else ACCENT for row in top]
    axis.barh(range(len(names)), values, xerr=errors, color=colours, height=0.66)
    axis.set_yticks(range(len(names)), names, color=INK, fontsize=7.5)
    _style(
        axis,
        f"Permutation importance on the test fold ({result.importance.get('scorer', 'n/a')})",
        "decrease in metric when the column is scrambled",
        "",
    )
    axis.axvline(0, color="#3a4757", linewidth=0.8)
    axis.text(
        0.0,
        -0.22,
        "grey = column is entirely missing in this data; negative = the column "
        "hurts when permuted into noise",
        transform=axis.transAxes,
        color=MUTED,
        fontsize=7.0,
    )
    return _save(figure, path)


def write_metrics_json(
    result: TrainingResult, task_dir: Path, figures: dict[str, str], extra: dict[str, Any]
) -> Path:
    payload = {
        "task": result.task,
        "winner": result.winner,
        "algorithm": result.algorithm,
        "selection_metric": result.selection_metric_name,
        "rows": result.rows,
        "groups": result.groups,
        "labels": result.labels,
        "feature_names": result.feature_names,
        "dataset_fingerprint": result.dataset_fingerprint,
        "holdout": result.holdout,
        "folds": result.folds,
        "candidates": [candidate.as_row() for candidate in result.candidates],
        "test_metrics": result.test_metrics,
        "train_metrics": result.train_metrics,
        "baseline_name": result.selected_baseline_name,
        "baseline_test_metrics": result.baseline_test_metrics,
        "importance": result.importance,
        "figures": figures,
        **extra,
    }
    path = task_dir / "metrics.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_jsonable), encoding="utf-8"
    )
    return path


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def build_reports(
    result: TrainingResult,
    report_root: Path,
    *,
    probabilities: np.ndarray | None = None,
    truth: np.ndarray | None = None,
    calibration: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write every figure and metrics.json for one task; return the index of what exists."""
    task_dir = Path(report_root) / result.task
    task_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, str] = {}

    if result.task == EXOPLANET_TASK:
        figures["confusion_matrix"] = plot_confusion_matrix(result, task_dir) or ""
    else:
        if probabilities is not None and truth is not None:
            figures["pr_vs_roc"] = plot_pr_vs_roc(result, task_dir, probabilities, truth) or ""
        if calibration is not None:
            figures["calibration"] = plot_calibration(result, task_dir, calibration) or ""
            figures["calibration_data"] = json.dumps(calibration)

    figures["candidate_comparison"] = plot_candidate_comparison(result, task_dir) or ""
    figures["permutation_importance"] = plot_importance(result, task_dir) or ""

    metrics_path = write_metrics_json(result, task_dir, figures, extra or {})
    return {
        "task_dir": str(task_dir),
        "metrics_path": str(metrics_path),
        "figures": figures,
        "summary": result.summary(),
    }


def summary_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    """The one table that goes in the README: model vs the rule it had to beat."""
    rows = []
    for item in results:
        summary = item["summary"]
        rows.append(
            {
                "task": summary["task"],
                "winner": summary["winner"],
                "metric": summary["selection_metric"],
                "model": summary["test_selection_score"],
                "baseline": summary["baseline_selection_score"],
                "delta": summary["delta_vs_baseline"],
                "rows": summary["rows"],
                "groups": summary["groups"],
            }
        )
    return pd.DataFrame(rows)
