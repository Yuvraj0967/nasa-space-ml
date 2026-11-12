"""Rule-based classifiers that wear sklearn's interface, so they are scored identically.

The point of this module is fairness. The spec's central claim is "NASA's published PHA
definition is the thing the learned model must beat", and that claim is only meaningful if
the rule travels through the same cross-validation, the same fold structure and the same
metric function as the gradient-boosted model. Wrapping it in `BaseEstimator`/`ClassifierMixin`
rather than hand-scoring a numpy array is what buys that.

Two baselines:

  * `MajorityRuleClassifier` - the floor. Always predicts the most frequent training class.
    Any model that cannot beat this is not learning physics, it is learning the class prior.
  * `PHARuleClassifier` - literally NASA's definition: an object is a Potentially Hazardous
    Asteroid when its MOID is at most 0.05 au AND its absolute magnitude H is at most 22.0
    (i.e. it is big enough to matter). Both numbers come from `configs/settings.yaml`, so
    the threshold in the model and the threshold in the documentation cannot drift apart.

A note on scoring a deterministic rule with average precision. The rule emits 0/1 scores,
so the precision-recall curve is a single step and PR-AUC becomes tie-sensitive. That is a
property of applying a *ranking* metric to a *non-ranking* classifier, not a bug in the
rule. The reports therefore lead the rule-vs-ML comparison with precision, recall and F1 at
the rule's own operating point - which are tie-free - and show PR-AUC alongside with exactly
this caveat attached.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

MOID_COLUMN = "moid_proxy_au"
H_COLUMN = "absolute_magnitude_h"


def _require_columns(X: Any, columns: tuple[str, ...]) -> pd.DataFrame:
    frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"the rule classifier needs columns {missing}; got {list(frame.columns)}")
    return frame


class MajorityRuleClassifier(BaseEstimator, ClassifierMixin):
    """Always predicts the most frequent training label. The floor every model must clear."""

    def fit(self, X: Any, y: Any) -> MajorityRuleClassifier:
        labels = pd.Series(np.asarray(y)).astype(str)
        counts = labels.value_counts()
        ordered = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
        self.classes_: np.ndarray = np.sort(labels.unique())
        self.major_class_: str = str(ordered[0][0])
        self.class_prior_: dict[str, float] = {
            str(label): float(count / len(labels)) for label, count in counts.items()
        }
        self.n_features_in_ = len(pd.DataFrame(X).columns)
        self.feature_names_in_ = np.asarray(pd.DataFrame(X).columns, dtype=object)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "classes_")
        frame = pd.DataFrame(X)
        probabilities = np.zeros((len(frame), len(self.classes_)), dtype=float)
        column = int(np.where(self.classes_ == self.major_class_)[0][0])
        probabilities[:, column] = 1.0
        return probabilities

    def predict(self, X: Any) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


class PHARuleClassifier(BaseEstimator, ClassifierMixin):
    """NASA's published PHA definition, evaluated feature-by-feature on the input frame.

    Missing data policy, stated rather than assumed: an object with no MOID or no H cannot
    satisfy "MOID <= 0.05 AND H <= 22.0", so the rule calls it NOT_PHA. That is the literal
    reading of the definition, and `unknown_count()` reports how many objects the policy
    touches so the number is visible instead of buried in a NaN comparison.
    """

    def __init__(
        self,
        moid_threshold_au: float = 0.05,
        h_threshold: float = 22.0,
        positive_label: str = "PHA",
        negative_label: str = "NOT_PHA",
    ) -> None:
        self.moid_threshold_au = moid_threshold_au
        self.h_threshold = h_threshold
        self.positive_label = positive_label
        self.negative_label = negative_label

    def fit(self, X: Any, y: Any) -> PHARuleClassifier:
        _require_columns(X, (MOID_COLUMN, H_COLUMN))
        labels = pd.Series(np.asarray(y)).astype(str)
        unique = np.sort(labels.unique())
        if self.positive_label not in set(unique) or self.negative_label not in set(unique):
            raise ValueError(
                "PHA rule expects both labels "
                f"{self.positive_label!r} and {self.negative_label!r} in the training data, "
                f"found {list(unique)}"
            )
        self.classes_ = np.array([self.negative_label, self.positive_label])
        self.n_features_in_ = len(pd.DataFrame(X).columns)
        self.feature_names_in_ = np.asarray(pd.DataFrame(X).columns, dtype=object)
        return self

    def hazardous_mask(self, X: Any) -> np.ndarray:
        """The rule itself: MOID <= threshold AND H <= threshold, NaN fails the AND."""
        frame = _require_columns(X, (MOID_COLUMN, H_COLUMN))
        moid = pd.to_numeric(frame[MOID_COLUMN], errors="coerce").to_numpy(dtype=float)
        magnitude = pd.to_numeric(frame[H_COLUMN], errors="coerce").to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            return (moid <= self.moid_threshold_au) & (magnitude <= self.h_threshold)

    def unknown_count(self, X: Any) -> int:
        """Objects the rule cannot judge because a required number is missing."""
        frame = _require_columns(X, (MOID_COLUMN, H_COLUMN))
        moid = pd.to_numeric(frame[MOID_COLUMN], errors="coerce")
        magnitude = pd.to_numeric(frame[H_COLUMN], errors="coerce")
        return int((moid.isna() | magnitude.isna()).sum())

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "classes_")
        hazardous = self.hazardous_mask(X)
        probabilities = np.zeros((len(hazardous), 2), dtype=float)
        probabilities[:, 1] = hazardous.astype(float)
        probabilities[:, 0] = 1.0 - probabilities[:, 1]
        return probabilities

    def predict(self, X: Any) -> np.ndarray:
        hazardous = self.hazardous_mask(X)
        return np.where(hazardous, self.positive_label, self.negative_label)
