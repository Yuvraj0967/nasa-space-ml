"""The baselines, including the one that matters most: NASA's own published PHA rule.

A learned model that cannot beat a two-line threshold is not a discovery, it is a
complicated way of restating the definition. Wrapping the rule as a scikit-learn
estimator is what lets the *same* metric code score both sides of that argument, which
is the entire point of `evaluation/metrics.py` existing as a shared module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.metrics import average_precision_score, precision_score, recall_score

from nasa_space_ml.models.baselines import (
    H_COLUMN,
    MOID_COLUMN,
    MajorityRuleClassifier,
    PHARuleClassifier,
)


@pytest.fixture()
def rule_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            MOID_COLUMN: [0.0010, 0.0499, 0.0500, 0.0501, 0.3000, np.nan, 0.0100, np.nan],
            H_COLUMN: [20.0, 21.9, 22.0, 21.0, 18.0, 20.0, np.nan, np.nan],
        }
    )


def rule_labels() -> np.ndarray:
    """Ground truth for `rule_frame`: exactly the rows the published definition covers
    (0, 1 and 2), including the boundary case at MOID = 0.0500 and H = 22.0."""
    return np.array(
        ["NOT_PHA", "PHA", "PHA", "NOT_PHA", "NOT_PHA", "NOT_PHA", "NOT_PHA", "NOT_PHA"]
    )


# ------------------------------------------------------------------ the PHA definition


def test_the_rule_is_the_published_conjunction(rule_frame: pd.DataFrame) -> None:
    """MOID <= 0.05 au AND H <= 22.0. Rows 0,1,2 qualify; row 3 misses by 0.0001 au;
    rows 4-7 miss on one of the two numbers or on both."""
    rule = PHARuleClassifier()
    assert list(rule.hazardous_mask(rule_frame)) == [
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
    ]


def test_thresholds_are_inclusive() -> None:
    """Exactly 0.05 au and exactly 22.0 mag are inside the definition. An off-by-one here
    silently redefines the class the model is being asked to learn."""
    boundary = pd.DataFrame({MOID_COLUMN: [0.05], H_COLUMN: [22.0]})
    assert PHARuleClassifier().hazardous_mask(boundary).tolist() == [True]


def test_both_conditions_are_required() -> None:
    near_but_big = pd.DataFrame({MOID_COLUMN: [0.001], H_COLUMN: [24.0]})
    small_but_far = pd.DataFrame({MOID_COLUMN: [0.9], H_COLUMN: [19.0]})
    rule = PHARuleClassifier()
    assert rule.hazardous_mask(near_but_big).tolist() == [False]
    assert rule.hazardous_mask(small_but_far).tolist() == [False]


def test_thresholds_are_configurable_and_used() -> None:
    frame = pd.DataFrame({MOID_COLUMN: [0.1], H_COLUMN: [23.0]})
    strict = PHARuleClassifier(moid_threshold_au=0.2, h_threshold=24.0)
    assert strict.hazardous_mask(frame).tolist() == [True]
    assert PHARuleClassifier().hazardous_mask(frame).tolist() == [False]


def test_missing_numbers_are_reported_as_unknown(rule_frame: pd.DataFrame) -> None:
    """Rows 5, 6 and 7 have at least one NaN. Calling them NOT_PHA is the literal reading
    of the definition; counting them is what stops that reading being a hidden assumption."""
    rule = PHARuleClassifier()
    assert rule.unknown_count(rule_frame) == 3


def test_predictions_follow_the_mask(rule_frame: pd.DataFrame) -> None:
    rule = PHARuleClassifier().fit(rule_frame, rule_labels())
    predicted = rule.predict(rule_frame)
    assert list(predicted) == np.where(rule.hazardous_mask(rule_frame), "PHA", "NOT_PHA").tolist()


def test_predict_proba_is_deterministic_zero_one(rule_frame: pd.DataFrame) -> None:
    """A rule has no uncertainty, so its probabilities are 0 and 1 and the rows sum to 1.
    Recording that explicitly is what makes it defensible to compare PR-AUC with a model
    that outputs real numbers."""
    rule = PHARuleClassifier().fit(rule_frame, rule_labels())
    proba = rule.predict_proba(rule_frame)
    assert proba.shape == (len(rule_frame), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert set(np.unique(proba)) <= {0.0, 1.0}
    assert np.allclose(proba[:, 1], rule.hazardous_mask(rule_frame).astype(float))


def test_classes_are_negative_first_like_sklearn_expects(rule_frame: pd.DataFrame) -> None:
    rule = PHARuleClassifier().fit(rule_frame, rule_labels())
    assert list(rule.classes_) == ["NOT_PHA", "PHA"]


def test_fit_refuses_training_data_that_lacks_a_label(rule_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="expects both labels"):
        PHARuleClassifier().fit(rule_frame, np.array(["PHA"] * len(rule_frame)))


def test_fit_refuses_a_frame_without_the_rule_columns() -> None:
    with pytest.raises(ValueError, match="needs columns"):
        PHARuleClassifier().fit(
            pd.DataFrame({"something_else": [1.0]}), np.array(["PHA", "NOT_PHA"])
        )


def test_predict_on_a_frame_missing_columns_names_them() -> None:
    rule = PHARuleClassifier().fit(
        pd.DataFrame({MOID_COLUMN: [0.1, 0.001], H_COLUMN: [20.0, 21.0]}),
        np.array(["NOT_PHA", "PHA"]),
    )
    with pytest.raises(ValueError, match=MOID_COLUMN):
        rule.predict(pd.DataFrame({"nope": [1.0]}))


# ---------------------------------------------------------------- sklearn protocol


@pytest.fixture()
def clone_ready(rule_frame: pd.DataFrame) -> PHARuleClassifier:
    return PHARuleClassifier().fit(rule_frame, rule_labels())


def test_the_rule_clones_and_refits(clone_ready: PHARuleClassifier) -> None:
    """`clone` is what GridSearchCV and cross_val_score use. An estimator that breaks it
    cannot be scored by the shared pipeline, which is the whole comparison design."""
    assert (
        clone(clone_ready).get_params()["moid_threshold_au"]
        == clone_ready.get_params()["moid_threshold_au"]
    )


def test_cloning_does_not_carry_the_fitted_state(clone_ready: PHARuleClassifier) -> None:
    assert not hasattr(clone(clone_ready), "classes_")


def test_estimator_attributes_are_set_by_fit_not_init(clone_ready: PHARuleClassifier) -> None:
    """The scikit-learn convention this project must not violate: `__init__` stores,
    `fit` learns. Getting it wrong makes `clone` silently carry a fitted state over."""
    assert hasattr(clone_ready, "classes_")
    assert hasattr(clone_ready, "n_features_in_")
    assert "classes_" not in PHARuleClassifier().get_params()


def test_feature_names_are_recorded(clone_ready: PHARuleClassifier) -> None:
    assert list(clone_ready.feature_names_in_) == [MOID_COLUMN, H_COLUMN]


def test_the_rule_reproduces_the_labels_it_inspired(
    asteroid_object_frame: pd.DataFrame, config
) -> None:
    """The fixture's labels *are* NASA's definition, so the rule must classify the fixture
    perfectly. If this ever fails, either the definition in `baselines.py` or the label
    rule in the fixture has drifted - and the headline "model beats baseline" claim in the
    README depends on the baseline being the real definition, not a nearby one."""
    from nasa_space_ml.features.asteroids import FEATURE_COLUMNS, derive

    X = derive(asteroid_object_frame)[list(FEATURE_COLUMNS)]
    y = np.where(asteroid_object_frame["pha_flag"].to_numpy(), "PHA", "NOT_PHA")
    rule = PHARuleClassifier(
        moid_threshold_au=config.asteroids.pha_distance_au,
        h_threshold=config.asteroids.pha_absolute_magnitude,
    ).fit(X, y)
    predicted = rule.predict(X)
    assert (predicted == y).all()
    assert precision_score(y, predicted, pos_label="PHA") == 1.0
    assert recall_score(y, predicted, pos_label="PHA") == 1.0


# ---------------------------------------------------------------- majority baseline


def test_majority_rule_always_predicts_the_plurality_label() -> None:
    X = pd.DataFrame({"anything": [1.0, 2.0, 3.0, 4.0]})
    y = np.array(["A", "A", "A", "B"])
    rule = MajorityRuleClassifier().fit(X, y)
    assert list(rule.predict(X)) == ["A"] * 4


def test_majority_rule_probability_columns_follow_classes_in_sort_order() -> None:
    """The column order is `sorted(classes_)`, not config order. This is the hazard that
    would silently pair a probability with the wrong label: `configs/settings.yaml` lists
    CONFIRMED, CANDIDATE, FALSE POSITIVE, and a response built in that order against
    `predict_proba` output would mislabel every class but one. Nothing in the API or the
    metrics layer may assume config order, and this test is where that is pinned."""
    X = pd.DataFrame({"anything": [1.0, 2.0, 3.0, 4.0]})
    y = np.array(["CONFIRMED", "FALSE POSITIVE", "CANDIDATE", "FALSE POSITIVE"])
    rule = MajorityRuleClassifier().fit(X, y)
    proba = rule.predict_proba(X)

    assert proba.shape == (4, 3)
    assert list(rule.classes_) == ["CANDIDATE", "CONFIRMED", "FALSE POSITIVE"]
    assert list(rule.classes_) != list(dict.fromkeys(y.tolist()))  # not first-seen order
    assert np.allclose(proba.sum(axis=1), 1.0)

    # The majority baseline is a hard one-hot on the plurality class, not a prior: it is a
    # deterministic rule, so it has no uncertainty to express. That makes its reported
    # log-loss and Brier score artificially terrible, which is why the model-selection
    # metric is macro-F1 / average precision and never a proper score.
    major = list(rule.classes_).index(rule.major_class_)
    assert np.allclose(proba[:, major], 1.0)
    assert np.allclose(np.delete(proba, major, axis=1), 0.0)


def test_majority_rule_records_the_prior() -> None:
    X = pd.DataFrame({"anything": np.arange(4.0)})
    y = np.array(["A", "A", "B", "C"])
    rule = MajorityRuleClassifier().fit(X, y)
    assert rule.class_prior_["A"] == pytest.approx(0.5)
    assert rule.major_class_ == "A"


def test_majority_rule_beats_nothing_by_beating_chance() -> None:
    """On the 5.3%-PHA fixture the majority baseline should have almost perfect accuracy
    and near-useless average precision. If a test ever shows the opposite, the fixture has
    stopped resembling the real imbalance."""
    X = pd.DataFrame({MOID_COLUMN: [0.9] * 19 + [0.01], H_COLUMN: [26.0] * 19 + [21.0]})
    y = np.array(["NOT_PHA"] * 19 + ["PHA"])
    rule = MajorityRuleClassifier().fit(X, y)
    proba = rule.predict_proba(X)[:, list(rule.classes_).index("PHA")]
    assert (proba == 0.0).all()
    assert average_precision_score(y == "PHA", proba) < 0.06
