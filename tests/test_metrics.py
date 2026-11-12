"""Metric layer, and the two scikit-learn traps that actually bit this project.

Both are recorded as tests rather than comments, because both produce *plausible looking*
numbers instead of errors: a mislabelled per-class table, or a scorer that quietly ranks
the wrong class. The third entry is the crash that started this file - `get_scorer` with
string labels - which at least had the decency to raise.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss

from nasa_space_ml.evaluation.metrics import (
    binary_metrics,
    build_scorer,
    calibration_block,
    multiclass_brier,
    multiclass_metrics,
    operating_points,
    per_class_block,
    permutation_importance_block,
    selection_metric,
)

NEGATIVE, POSITIVE = "NOT_PHA", "PHA"


def _binary_problem(seed: int = 3, n: int = 120):
    """A genuinely separable-but-noisy binary problem with string labels, so a metric bug
    shows up as a wrong number rather than an unparseable input."""
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.15
    score = np.clip(rng.normal(0.0, 0.35, n) + 0.85 * y, 0.0, 1.0)
    labels = np.where(y, POSITIVE, NEGATIVE)
    proba = np.column_stack([1.0 - score, score])
    predicted = np.where(score > 0.5, POSITIVE, NEGATIVE)
    return labels, proba, score, predicted


def _multiclass_problem(seed: int = 5, n: int = 180):
    rng = np.random.default_rng(seed)
    classes = np.array(["CANDIDATE", "CONFIRMED", "FALSE POSITIVE"])
    indices = rng.integers(0, 3, size=n)
    proba = rng.dirichlet(np.ones(3) * 2.5, size=n)
    proba[np.arange(n), indices] += 1.5
    proba = proba / proba.sum(axis=1, keepdims=True)
    truth = classes[indices]
    predicted = classes[np.argmax(proba, axis=1)]
    return truth, proba, predicted, classes


# ------------------------------------------------------------------- per-class tables


def test_per_class_block_accepts_three_classes() -> None:
    """The bug this guards: asking for `precision_score(labels=[one_class])` defaults to
    `average="binary"`, and scikit-learn refuses to do binary arithmetic on a three-class
    problem. `average=None` is not a style choice."""
    truth, _proba, predicted, classes = _multiclass_problem()
    block = per_class_block(truth, predicted, classes)
    assert set(block) == set(classes.tolist())
    for row in block.values():
        assert {"precision", "recall", "f1", "support"} <= set(row)


def test_per_class_support_counts_the_truth_not_the_predictions() -> None:
    truth, _proba, _predicted, classes = _multiclass_problem()
    block = per_class_block(truth, np.array(["FALSE POSITIVE"] * len(truth)), classes)
    for label, row in block.items():
        assert row["support"] == int((truth == label).sum())


def test_a_class_nobody_predicted_scores_zero_instead_of_crashing() -> None:
    """Small folds regularly contain a class that the model never predicts. `zero_division`
    has to be explicit or the run dies forty thousand rows in."""
    truth, _proba, _predicted, classes = _multiclass_problem()
    block = per_class_block(truth, np.array(["CONFIRMED"] * len(truth)), classes)
    assert block["CANDIDATE"]["precision"] == 0.0
    assert block["CONFIRMED"]["recall"] == 1.0


def test_f1_is_the_harmonic_mean_of_the_block() -> None:
    truth, _proba, predicted, classes = _multiclass_problem()
    for row in per_class_block(truth, predicted, classes).values():
        expected = (
            0.0
            if row["precision"] + row["recall"] == 0
            else 2 * row["precision"] * row["recall"] / (row["precision"] + row["recall"])
        )
        assert row["f1"] == pytest.approx(expected, abs=1e-9)


# --------------------------------------------------------------------- multiclass block


def test_multiclass_metrics_reports_the_documented_keys() -> None:
    truth, proba, predicted, classes = _multiclass_problem()
    metrics = multiclass_metrics(truth, proba, predicted, classes)
    assert {
        "macro_f1",
        "weighted_f1",
        "accuracy",
        "log_loss",
        "brier",
        "per_class",
        "confusion",
    } <= set(metrics)
    assert {"roc_auc_ovr", "roc_auc_macro"} <= set(metrics)


def test_macro_f1_weights_the_minority_class_equally() -> None:
    """The reason macro-F1 is the selection metric: a model that always says the majority
    class scores 0.22 here, not 0.5, despite 50%-ish accuracy."""
    truth, proba, _predicted, classes = _multiclass_problem()
    always_majority = np.array(["FALSE POSITIVE"] * len(truth))
    good = multiclass_metrics(truth, proba, truth, classes)
    lazy = multiclass_metrics(truth, proba, always_majority, classes)
    assert lazy["accuracy"] > 0.3
    assert lazy["macro_f1"] < good["macro_f1"] / 2


def test_perfect_predictions_have_zero_loss() -> None:
    truth, _proba, _predicted, classes = _multiclass_problem()
    perfect_proba = np.eye(len(classes))[np.searchsorted(classes, truth)]
    metrics = multiclass_metrics(truth, perfect_proba, truth, classes)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["brier"] == pytest.approx(0.0)


def test_confusion_matrix_rows_are_truth_and_columns_are_prediction() -> None:
    truth = np.array(["A", "A", "B", "B"])
    predicted = np.array(["A", "B", "B", "B"])
    classes = np.array(["A", "B"])
    block = multiclass_metrics(truth, np.full((4, 2), 0.5), predicted, classes)["confusion"]
    assert block["matrix"] == [[1, 1], [0, 2]]
    assert block["labels"] == ["A", "B"]


def test_brier_of_a_uniform_guess_is_the_expected_value() -> None:
    truth = np.array(["A", "B"])
    classes = np.array(["A", "B"])
    assert multiclass_brier(truth, np.full((2, 2), 0.5), classes) == pytest.approx(0.25)


def test_log_loss_is_the_ordinary_cross_entropy() -> None:
    truth, proba, predicted, classes = _multiclass_problem()
    metrics = multiclass_metrics(truth, proba, predicted, classes)
    assert metrics["log_loss"] == pytest.approx(log_loss(truth, proba, labels=list(classes)))


# ------------------------------------------------------------------------ binary block


def test_binary_metrics_needs_the_positive_label_named() -> None:
    """`"PHA"` is alphabetically *after* `"NOT_PHA"`, so sorting happens to put the
    positive class last here - but that is luck, not a contract, and the API refuses to
    rely on it."""
    truth, proba, _score, predicted = _binary_problem()
    metrics = binary_metrics(truth, proba, predicted, positive=POSITIVE, negative=NEGATIVE)
    assert metrics["prevalence"] == pytest.approx(float((truth == POSITIVE).mean()))
    assert 0.0 <= metrics["average_precision"] <= 1.0


def test_average_precision_matches_sklearn_directly() -> None:
    truth, proba, score, predicted = _binary_problem()
    metrics = binary_metrics(truth, proba, predicted, positive=POSITIVE, negative=NEGATIVE)
    assert metrics["average_precision"] == pytest.approx(
        average_precision_score((truth == POSITIVE).astype(int), score)
    )


def test_a_probability_matrix_and_a_1d_score_agree() -> None:
    truth, proba, score, predicted = _binary_problem()
    from_matrix = binary_metrics(truth, proba, predicted, positive=POSITIVE, negative=NEGATIVE)
    from_vector = binary_metrics(truth, score, predicted, positive=POSITIVE, negative=NEGATIVE)
    assert from_matrix["average_precision"] == pytest.approx(from_vector["average_precision"])
    assert from_matrix["roc_auc"] == pytest.approx(from_vector["roc_auc"])


def test_roc_auc_of_a_random_ranking_is_about_one_half() -> None:
    truth, _proba, _score, predicted = _binary_problem()
    rng = np.random.default_rng(11)
    noise = rng.random(len(truth))
    metrics = binary_metrics(truth, noise, predicted, positive=POSITIVE, negative=NEGATIVE)
    assert 0.35 < metrics["roc_auc"] < 0.65
    assert metrics["average_precision"] < max(0.5, float((truth == POSITIVE).mean()) * 3)


def test_perfect_ranking_scores_one() -> None:
    truth = np.array([NEGATIVE, NEGATIVE, POSITIVE, POSITIVE])
    score = np.array([0.05, 0.15, 0.75, 0.95])
    predicted = np.where(score > 0.5, POSITIVE, NEGATIVE)
    metrics = binary_metrics(truth, score, predicted, positive=POSITIVE, negative=NEGATIVE)
    assert metrics["average_precision"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["recall_positive"] == pytest.approx(1.0)


def test_confusion_and_per_class_share_an_orientation() -> None:
    truth, proba, _score, predicted = _binary_problem()
    metrics = binary_metrics(truth, proba, predicted, positive=POSITIVE, negative=NEGATIVE)
    assert metrics["confusion"]["labels"] == [NEGATIVE, POSITIVE]
    assert metrics["per_class"][POSITIVE]["support"] == int((truth == POSITIVE).sum())
    total = sum(sum(row) for row in metrics["confusion"]["matrix"])
    assert total == len(truth)


# -------------------------------------------------------------------- operating points


def test_operating_points_trade_precision_for_recall() -> None:
    """The question a reviewer asks is "how many hazards do I miss when I only want real
    alerts?", and the answer has to come with the threshold that produces it."""
    truth, _proba, score, _predicted = _binary_problem()
    rows = operating_points((truth == POSITIVE).astype(int), score)
    assert [row["target_precision"] for row in rows] == [0.10, 0.25, 0.50, 0.75]
    recalls = [row["recall"] for row in rows]
    assert recalls == sorted(recalls, reverse=True) or max(recalls) == recalls[0]


def test_operating_point_precision_meets_the_target_when_reachable() -> None:
    truth, _proba, score, _predicted = _binary_problem()
    binary_truth = (truth == POSITIVE).astype(int)
    for row in operating_points(binary_truth, score, target_precisions=(0.25, 0.5)):
        assert row["precision"] >= row["target_precision"] - 1e-9
        assert 0.0 <= row["threshold"] <= 1.0


def test_an_unreachable_precision_target_reports_the_best_available() -> None:
    truth, _proba, score, _predicted = _binary_problem()
    binary_truth = (truth == POSITIVE).astype(int)
    rows = operating_points(binary_truth, score, target_precisions=(0.999,))
    assert len(rows) == 1
    assert 0.0 <= rows[0]["precision"] <= 1.0


# ------------------------------------------------------------------------- calibration


def test_calibration_block_shape() -> None:
    truth, _proba, score, _predicted = _binary_problem(n=400)
    block = calibration_block((truth == POSITIVE).astype(int), score, n_bins=10)
    assert block["n_bins"] == 10
    assert len(block["mean_predicted"]) == len(block["fraction_positive"])
    assert all(0.0 <= value <= 1.0 for value in block["mean_predicted"])


def test_a_well_calibrated_score_tracks_the_diagonal() -> None:
    rng = np.random.default_rng(4)
    score = rng.random(4000)
    truth = (rng.random(4000) < score).astype(int)
    block = calibration_block(truth, score, n_bins=10)
    gaps = [
        abs(predicted - actual)
        for predicted, actual in zip(
            block["mean_predicted"], block["fraction_positive"], strict=True
        )
    ]
    assert max(gaps) < 0.08


# ------------------------------------------------------------------- the scorer builder


def test_average_precision_scorer_refuses_a_missing_positive_label() -> None:
    """The regression this file was written for. `get_scorer("average_precision")` silently
    assumes `pos_label=1`, which is not one of our labels, and the whole asteroid training
    run died inside permutation importance with
    `ValueError: pos_label=1 is not a valid label`. Asking for it without naming the class
    is now a programming error raised at build time, not at the end of a long fit."""
    with pytest.raises(ValueError, match="needs an explicit pos_label"):
        build_scorer("average_precision")
    with pytest.raises(ValueError, match="needs an explicit pos_label"):
        build_scorer("roc_auc")


def test_average_precision_scorer_matches_the_direct_metric() -> None:
    truth, _proba, score, _predicted = _binary_problem()
    model = LogisticRegression().fit(np.column_stack([score, 1 - score]), truth)
    X = np.column_stack([score, 1 - score])
    scorer = build_scorer("average_precision", pos_label=POSITIVE)
    assert scorer(model, X, truth) == pytest.approx(
        average_precision_score((truth == POSITIVE).astype(int), model.predict_proba(X)[:, 1])
    )


def test_a_named_macro_scorer_needs_no_positive_label() -> None:
    """Macro-F1 is a hard-metric on all classes, so it has no positive class to name and
    `get_scorer` is safe for it. The asymmetry is the point: only the ranking metrics
    carry the hidden `pos_label`."""
    from sklearn.metrics import f1_score

    truth, proba, _predicted, _classes = _multiclass_problem()
    model = LogisticRegression(max_iter=2000).fit(proba, truth)
    scorer = build_scorer("f1_macro")
    assert scorer(model, proba, truth) == pytest.approx(
        f1_score(truth, model.predict(proba), average="macro")
    )


# ---------------------------------------------------------------- permutation importance


def test_permutation_importance_survives_string_labels() -> None:
    """The same crash, end to end: this is the exact call that failed during the real
    asteroid training run, so it runs here with the real label strings.

    The frame is passed to `fit` as well as to the scorer, because scikit-learn warns when
    a model fitted on a bare array is later scored with a named DataFrame - and in this
    project the feature names are the contract, so every estimator here is fitted on a
    frame."""
    truth, proba, _predicted, _classes = _binary_problem(n=200)
    frame = pd.DataFrame(proba, columns=["a", "b"])
    model = LogisticRegression().fit(frame, truth)
    block = permutation_importance_block(
        model,
        frame,
        truth,
        "average_precision",
        pos_label=POSITIVE,
        n_repeats=3,
        seed=1,
    )
    assert len(block["features"]) == 2
    assert block["pos_label"] == POSITIVE
    assert block["features"] == sorted(block["features"], key=lambda row: -row["mean_decrease"])
    assert {row["feature"] for row in block["features"]} == {"a", "b"}


def test_permutation_importance_flags_an_absent_column() -> None:
    """Almost every asteroid in the real catalogue has no published diameter, so a column
    that is missing on 98.5% of rows is a real shape here. It must be labelled as
    uninformative rather than quietly contributing noise to the importance chart.

    Side note found while writing this test: `HistGradientBoostingClassifier` cannot even
    *fit* a 100%-NaN column - its binning raises "window shape cannot be larger than input
    array shape". The production data dodges that only because the column is 98.5% missing,
    not entirely missing. RandomForest tolerates both, so it is used here."""
    from sklearn.ensemble import RandomForestClassifier

    truth, proba, _predicted, _classes = _binary_problem(n=400)
    frame = pd.DataFrame(proba, columns=["a", "b"])
    frame["dead"] = math.nan
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(frame, truth)
    block = permutation_importance_block(
        model, frame, truth, "average_precision", pos_label=POSITIVE, n_repeats=2, seed=1
    )
    dead = [row for row in block["features"] if row["feature"] == "dead"]
    assert dead and dead[0]["null_importance"] is True
    assert dead[0]["mean_decrease"] == pytest.approx(0.0, abs=1e-12)


def test_permutation_importance_separates_signal_from_noise() -> None:
    """The column that carries the label should lose more score when scrambled than a
    column of pure noise. It is the sanity check on the whole importance figure in the
    README - if this flips, the published importances are not trustworthy."""
    rng = np.random.default_rng(0)
    n = 400
    signal = rng.random(n)
    truth = np.where(rng.random(n) < signal, POSITIVE, NEGATIVE)
    frame = pd.DataFrame({"signal": signal, "noise": rng.random(n)})
    model = LogisticRegression().fit(frame, truth)
    block = permutation_importance_block(
        model, frame, truth, "average_precision", pos_label=POSITIVE, n_repeats=8, seed=2
    )
    means = {row["feature"]: row["mean_decrease"] for row in block["features"]}
    assert means["signal"] > 0.01
    assert block["features"][0]["feature"] == "signal"


# --------------------------------------------------------------------- selection metric


def test_selection_metric_reads_the_documented_number_per_task() -> None:
    truth, proba, predicted, classes = _multiclass_problem()
    multi = multiclass_metrics(truth, proba, predicted, classes)
    assert selection_metric(multi, "exoplanets") == pytest.approx(multi["macro_f1"])

    binary_truth, binary_proba, _score, binary_pred = _binary_problem()
    single = binary_metrics(
        binary_truth, binary_proba, binary_pred, positive=POSITIVE, negative=NEGATIVE
    )
    assert selection_metric(single, "asteroids") == pytest.approx(single["average_precision"])


def test_selection_metric_refuses_an_unknown_task() -> None:
    with pytest.raises(KeyError):
        selection_metric({"macro_f1": 0.5}, "telemetry")


def test_selection_metric_refuses_a_missing_key_rather_than_defaulting_to_zero() -> None:
    """Defaulting to 0.0 would silently retire the winner and promote the baseline."""
    with pytest.raises(ValueError, match="no 'average_precision'"):
        selection_metric({"accuracy": 0.9}, "asteroids")


def test_metrics_are_serialisable_to_json() -> None:
    """`reports/{task}/metrics.json` is a published artifact, so a stray numpy scalar is a
    build failure rather than a cosmetic one."""
    import json

    truth, proba, predicted, classes = _multiclass_problem()
    payload = multiclass_metrics(truth, proba, predicted, classes)
    json.dumps(payload)

    binary_truth, binary_proba, _score, binary_pred = _binary_problem()
    json.dumps(
        binary_metrics(
            binary_truth, binary_proba, binary_pred, positive=POSITIVE, negative=NEGATIVE
        )
    )


def test_roc_auc_is_not_used_for_the_imbalanced_task() -> None:
    """A deliberately weak ranker still scores well on ROC when 85% of the population is
    boring. This is the argument for average precision, and it only holds if the numbers
    actually diverge - so they are asserted to."""
    rng = np.random.default_rng(9)
    n = 2000
    truth = np.where(rng.random(n) < 0.05, POSITIVE, NEGATIVE)
    score = rng.random(n) + 0.25 * (truth == POSITIVE)
    metrics = binary_metrics(
        truth,
        score,
        np.where(score > 0.8, POSITIVE, NEGATIVE),
        positive=POSITIVE,
        negative=NEGATIVE,
    )
    assert metrics["roc_auc"] > 0.6
    assert metrics["average_precision"] < 0.25
