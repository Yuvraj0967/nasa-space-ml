"""Grouped splitting - the part of this project that decides whether every other number
is real.

Kepler reports many candidates around the same star and CAD reports many approaches by
the same object. Splitting rows at random therefore puts the same star (or the same
asteroid) on both sides of the boundary, and the stellar parameters that identify it are
features. The score survives and means nothing. These tests are the reason the split is
group-based and the reason the guard exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nasa_space_ml.errors import ConfigError
from nasa_space_ml.models.splits import (
    Holdout,
    _plurality_per_group,
    assert_no_group_leakage,
    fold_report,
    grouped_cv_splits,
    grouped_holdout,
)


def _frame(labels: list[str], groups: list[str]) -> tuple[pd.Series, pd.Series]:
    return pd.Series(labels, dtype=str), pd.Series(groups, dtype=str)


def _many_rows(n: int = 300, stars: int = 60) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Three classes, several candidates per star, and one very rare class, which is the
    shape of the real KOI table."""
    rng = np.random.default_rng(7)
    group_index = rng.integers(0, stars, size=n)
    labels = []
    for index in range(n):
        if group_index[index] % 20 == 0:
            labels.append("CANDIDATE")  # rare: one star, a handful of rows
        else:
            labels.append("CONFIRMED" if index % 3 else "FALSE POSITIVE")
    groups = pd.Series([f"STAR{value:04d}" for value in group_index])
    X = pd.DataFrame({"f": rng.normal(size=n)})
    return X, pd.Series(labels), groups


# ------------------------------------------------------------------ plurality per group


def test_plurality_picks_the_majority_class_of_a_group() -> None:
    labels, groups = _frame(["A", "A", "B"], ["g1", "g1", "g1"])
    assert _plurality_per_group(labels, groups) == {"g1": "A"}


def test_plurality_breaks_a_tie_on_the_label_not_on_hash_order() -> None:
    """A tie has to resolve the same way on every machine, or the split stops being
    reproducible and the published metrics cannot be regenerated. The rule is
    highest-count-then-lexicographic, so row order must not matter."""
    forward, _ = _frame(["B", "A"], ["g1", "g1"])
    backward, _ = _frame(["A", "B"], ["g1", "g1"])
    groups = pd.Series(["g1", "g1"])
    assert (
        _plurality_per_group(forward, groups)
        == _plurality_per_group(backward, groups)
        == {"g1": "A"}
    )


def test_plurality_prefers_count_over_alphabetical_order() -> None:
    labels, groups = _frame(["ZEBRA", "ZEBRA", "APPLE"], ["g1", "g1", "g1"])
    assert _plurality_per_group(labels, groups) == {"g1": "ZEBRA"}


def test_plurality_is_per_group_not_per_row() -> None:
    labels, groups = _frame(["A", "A", "B", "A"], ["g1", "g1", "g2", "g2"])
    assert _plurality_per_group(labels, groups) == {"g1": "A", "g2": "A"}


def test_plurality_reports_the_group_not_the_dataset() -> None:
    """The whole point: the global majority is B, and yet one group is majority A."""
    labels, groups = _frame(["A", "A", "B", "B", "B"], ["g1", "g1", "g2", "g2", "g2"])
    result = _plurality_per_group(labels, groups)
    assert result == {"g1": "A", "g2": "B"}
    assert labels.value_counts().idxmax() == "B"


# ------------------------------------------------------------------------ the holdout


def test_holdout_is_group_disjoint(config) -> None:
    _, y, groups = _many_rows()
    holdout = grouped_holdout(y, groups, config)
    train = set(groups.iloc[holdout.train_index])
    val = set(groups.iloc[holdout.val_index])
    test = set(groups.iloc[holdout.test_index])
    assert not train & val
    assert not train & test
    assert not val & test


def test_holdout_covers_every_row(config) -> None:
    _, y, groups = _many_rows()
    holdout = grouped_holdout(y, groups, config)
    covered = np.concatenate([holdout.train_index, holdout.val_index, holdout.test_index])
    assert sorted(covered.tolist()) == list(range(len(y)))


def test_holdout_is_reproducible(config) -> None:
    _, y, groups = _many_rows()
    first = grouped_holdout(y, groups, config)
    second = grouped_holdout(y, groups, config)
    assert np.array_equal(first.test_index, second.test_index)
    assert np.array_equal(first.val_index, second.val_index)


def test_seed_changes_the_split(config, tmp_path) -> None:
    """Different seeds must genuinely reshuffle, otherwise the seed is decorative."""
    from dataclasses import replace as dc_replace

    _, y, groups = _many_rows()
    other = dc_replace(config, seed=config.seed + 1)
    a = set(groups.iloc[grouped_holdout(y, groups, config).test_index])
    b = set(groups.iloc[grouped_holdout(y, groups, other).test_index])
    assert a != b


def test_sizes_respect_the_configured_fractions(config) -> None:
    _, y, groups = _many_rows(n=1200, stars=240)
    sizes = grouped_holdout(y, groups, config).sizes()
    assert sizes["train"] + sizes["val"] + sizes["test"] == len(y)
    assert abs(sizes["test"] / len(y) - config.split.test_size) < 0.12
    assert abs(sizes["val"] / len(y) - config.split.val_size) < 0.12


def test_rare_class_reaches_every_fold(config) -> None:
    """Stratifying by group plurality is what stops the rare class landing entirely in
    train and making the test fold unable to score it at all."""
    _, y, groups = _many_rows(n=900, stars=120)
    holdout = grouped_holdout(y, groups, config)
    for index in (holdout.train_index, holdout.val_index, holdout.test_index):
        assert "CANDIDATE" in set(y.iloc[index])


def test_a_group_never_straddles_folds_even_when_it_is_rare(config) -> None:
    _, y, groups = _many_rows(n=900, stars=120)
    holdout = grouped_holdout(y, groups, config)
    mapping = groups.to_numpy()
    for index in (holdout.train_index, holdout.val_index, holdout.test_index):
        others = set(mapping) - set(mapping[index])
        assert not set(mapping[index]) & others


def test_too_few_groups_is_an_error_not_a_degenerate_split(config) -> None:
    labels, groups = _frame(["A", "A", "A"], ["only", "only", "only"])
    with pytest.raises(ConfigError, match="at least 3 groups"):
        grouped_holdout(labels, groups, config)


def test_length_mismatch_is_caught(config) -> None:
    labels, groups = _frame(["A", "B"], ["g1", "g2", "g3"])
    with pytest.raises(ConfigError, match="same length"):
        grouped_holdout(labels, groups, config)


def test_holdout_sizes_are_reported(config) -> None:
    _, y, groups = _many_rows()
    sizes = grouped_holdout(y, groups, config).sizes()
    assert set(sizes) == {"train", "val", "test"}
    assert all(value > 0 for value in sizes.values())


# --------------------------------------------------------------------------- cv folds


def test_cv_folds_are_group_disjoint(config) -> None:
    X, y, groups = _many_rows()
    for train_index, val_index in grouped_cv_splits(X, y, groups, config):
        train = set(groups.iloc[train_index])
        val = set(groups.iloc[val_index])
        assert not train & val


def test_cv_produces_the_configured_number_of_folds(config) -> None:
    X, y, groups = _many_rows()
    assert len(list(grouped_cv_splits(X, y, groups, config))) == config.split.n_folds


def test_cv_fold_count_is_capped_by_the_number_of_groups(config) -> None:
    X, y, groups = _many_rows(n=40, stars=3)
    assert len(list(grouped_cv_splits(X, y, groups, config))) == 3


def test_cv_needs_two_groups(config) -> None:
    X = pd.DataFrame({"f": [1.0, 2.0, 3.0]})
    y = pd.Series(["A", "A", "B"])
    groups = pd.Series(["same", "same", "same"])
    with pytest.raises(ConfigError, match="at least 2 distinct groups"):
        list(grouped_cv_splits(X, y, groups, config))


def test_cv_is_reproducible(config) -> None:
    X, y, groups = _many_rows()
    first = list(grouped_cv_splits(X, y, groups, config))
    second = list(grouped_cv_splits(X, y, groups, config))
    for (a_train, a_val), (b_train, b_val) in zip(first, second, strict=True):
        assert np.array_equal(a_val, b_val)
        assert np.array_equal(a_train, b_train)


def test_every_row_is_a_validation_row_exactly_once(config) -> None:
    X, y, groups = _many_rows()
    seen = np.concatenate([val for _train, val in grouped_cv_splits(X, y, groups, config)])
    assert sorted(seen.tolist()) == list(range(len(y)))


def test_cv_carries_every_class_into_every_fold_when_the_data_allow_it(config) -> None:
    """Ten stars per class, five folds: stratification is decidable here, and a metric
    computed per fold is comparable across folds."""
    X = pd.DataFrame({"f": np.arange(150, dtype=float)})
    labels, groups = [], []
    for star in range(30):
        for _row in range(5):
            labels.append(["CONFIRMED", "FALSE POSITIVE", "CANDIDATE"][star % 3])
            groups.append(f"S{star:03d}")
    y = pd.Series(labels)
    folds = list(grouped_cv_splits(X, y, pd.Series(groups), config))
    assert len(folds) == config.split.n_folds
    for _train, val in folds:
        assert len(set(y.iloc[val])) == 3


def test_cv_cannot_put_a_class_in_every_fold_when_groups_are_the_limit(config) -> None:
    """The honest limitation, recorded as a test: with only three CANDIDATE stars and five
    folds, two validation folds cannot contain it, no matter how good the splitter is.
    This is why the reported number is a mean over folds with a standard deviation, and
    why a rare class with few *groups* is a rarer signal than its row count suggests."""
    X = pd.DataFrame({"f": np.arange(90, dtype=float)})
    labels, groups = [], []
    for star in range(18):
        for _row in range(5):
            labels.append("CANDIDATE" if star < 3 else "CONFIRMED")
            groups.append(f"S{star:03d}")
    y = pd.Series(labels)
    folds = list(grouped_cv_splits(X, y, pd.Series(groups), config))
    with_candidate = [val for _train, val in folds if "CANDIDATE" in set(y.iloc[val])]
    assert 0 < len(with_candidate) <= 3
    assert len(with_candidate) < len(folds)


# ---------------------------------------------------------------------- the leakage guard


def test_guard_passes_on_disjoint_splits() -> None:
    assert_no_group_leakage(("train", ["a", "b"]), ("test", ["c", "d"]))  # must not raise


def test_guard_raises_with_the_honest_wording() -> None:
    """The message says the metrics are fiction on purpose. A silent leak does not look
    like a bug, it looks like a good model."""
    with pytest.raises(ConfigError) as excinfo:
        assert_no_group_leakage(("train", ["a", "shared"]), ("test", ["shared", "b"]))
    assert "fiction" in str(excinfo.value)
    assert excinfo.value.details["group"] == "shared"


def test_guard_ignores_repeats_within_one_split() -> None:
    """Rows of the same group inside the same fold are expected - that is the point of
    grouping. Only a group appearing in two *different* folds is a leak."""
    assert_no_group_leakage(("train", ["a", "a", "b"]), ("test", ["c"]))  # must not raise


def test_guard_catches_a_leak_between_train_and_cv() -> None:
    with pytest.raises(ConfigError):
        assert_no_group_leakage(
            ("train", ["x"]),
            ("val", ["y"]),
            ("test", ["x"]),
        )


def test_guard_treats_numeric_and_string_ids_as_the_same_group() -> None:
    """`kepid` arrives from the archive as an integer and from the API as a string; a leak
    must be caught either way."""
    with pytest.raises(ConfigError, match="4000000001"):
        assert_no_group_leakage(("train", [4000000001]), ("test", ["4000000001"]))


# --------------------------------------------------------------------------- reporting


def test_fold_report_covers_every_fold(config) -> None:
    _, y, groups = _many_rows()
    holdout = grouped_holdout(y, groups, config)
    report = fold_report(y, holdout, groups)
    assert set(report) == {"train", "val", "test"}
    for name, row in report.items():
        assert row["rows"] == holdout.sizes()[name]
        assert row["groups"] >= 1


def test_fold_report_counts_classes(config) -> None:
    labels, groups = _frame(["A", "A", "B", "B", "C", "C"], ["g1", "g1", "g2", "g2", "g3", "g3"])
    holdout = Holdout(
        train_index=np.array([0, 1, 2, 3]), val_index=np.array([4]), test_index=np.array([5])
    )
    report = fold_report(pd.Series(labels), holdout, pd.Series(groups))
    assert report["train"]["classes"]["A"] == 2
    assert report["val"]["classes"]["C"] == 1
