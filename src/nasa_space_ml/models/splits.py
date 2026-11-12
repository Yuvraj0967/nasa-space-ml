"""Grouped train/validation/test splitting.

The rule this module exists to enforce: a *group* is the unit that must not be cut in half.
For Kepler that group is the host star (several planets around one star share stellar
parameters), for NEOs it is the asteroid (many close-approach events for one object share
H, diameter and orbit quality). Splitting rows instead of groups leaks the answer across the
boundary and inflates every number in the report.

Two mechanisms, used for different jobs:

  * `grouped_holdout` - one deterministic 70/15/15 split by group, used for the final
    numbers we publish. Class balance is preserved by distributing groups *per plurality
    class* round-robin, so the rare PHA class does not vanish from the test fold.
  * `grouped_cv_splits` - `StratifiedGroupKFold` for model selection, so no group is ever
    simultaneously trained and validated, and no fold is missing a class.

`assert_no_group_leakage` is not decoration: it runs in the training pipeline and in tests,
and it is the cheapest possible defence against a future refactor quietly reintroducing the
bug.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ..config import Config
from ..errors import ConfigError


@dataclass(frozen=True)
class Holdout:
    train_index: np.ndarray
    val_index: np.ndarray
    test_index: np.ndarray

    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.train_index),
            "val": len(self.val_index),
            "test": len(self.test_index),
        }


def _plurality_per_group(labels: pd.Series, groups: pd.Series) -> dict[str, str]:
    """The most common class inside each group, used as the group's stratification key.

    Ties break on the label string, never on dictionary order, so the split is identical
    across machines and Python hash seeds.
    """
    table = pd.DataFrame(
        {"group": [str(value) for value in groups], "label": [str(value) for value in labels]}
    )
    sizes = table.groupby(["group", "label"], sort=False).size().reset_index(name="n")
    winners: dict[str, tuple[int, str]] = {}
    for row in sizes.itertuples(index=False):
        key, label, count = str(row.group), str(row.label), int(row.n)
        current = winners.get(key)
        if current is None or (-count, label) < (-current[0], current[1]):
            winners[key] = (count, label)
    return {group: value[1] for group, value in winners.items()}


def _round_robin_assign(
    ordered_groups: Sequence[str],
    fold_names: Sequence[str],
    weights: Sequence[float],
) -> dict[str, str]:
    """Assign groups to folds so cumulative *group* counts track the target fractions.

    Greedy on the running shortfall: each group goes to the fold that is furthest below its
    quota so far. Deterministic given the input order, and it stays stable when the catalogue
    grows by a handful of rows - which is what makes published numbers re-runnable.
    """
    quotas = {name: weight * len(ordered_groups) for name, weight in zip(fold_names, weights, strict=True)}
    taken: Counter[str] = Counter()
    assignment: dict[str, str] = {}
    for group in ordered_groups:
        best_name = fold_names[0]
        best_gap = -np.inf
        for name in fold_names:
            gap = quotas[name] - taken[name]
            if gap > best_gap:
                best_gap = gap
                best_name = name
        assignment[group] = best_name
        taken[best_name] += 1
    return assignment


def grouped_holdout(
    labels: pd.Series,
    groups: pd.Series,
    config: Config,
    *,
    fold_names: Sequence[str] = ("train", "val", "test"),
) -> Holdout:
    """Deterministic group-disjoint train/val/test split.

    Groups are bucketed by their plurality class and shuffled with a seeded generator, so a
    rare class is spread across all three folds instead of ending up entirely in train.
    """
    if len(labels) != len(groups):
        raise ConfigError("labels and groups must have the same length")
    unique_groups = pd.Series(groups).astype(str).unique()
    if len(unique_groups) < len(fold_names):
        raise ConfigError(
            f"need at least {len(fold_names)} groups to split, found {len(unique_groups)}"
        )

    plurality = _plurality_per_group(pd.Series(labels).astype(str), pd.Series(groups).astype(str))
    rng = np.random.default_rng(config.seed)
    weights = [1.0 - config.split.test_size - config.split.val_size,
               config.split.val_size,
               config.split.test_size]

    by_class: dict[str, list[str]] = defaultdict(list)
    for group in sorted(unique_groups):
        by_class[plurality[group]].append(group)

    assignment: dict[str, str] = {}
    for class_name in sorted(by_class):
        members = np.array(by_class[class_name], dtype=object)
        order = rng.permutation(len(members))
        shuffled = [str(members[index]) for index in order]
        assignment.update(_round_robin_assign(shuffled, list(fold_names), weights))

    folded_groups = pd.Series(groups).astype(str).map(assignment).to_numpy()
    index = np.arange(len(labels))
    return Holdout(
        train_index=index[folded_groups == fold_names[0]],
        val_index=index[folded_groups == fold_names[1]],
        test_index=index[folded_groups == fold_names[2]],
    )


def grouped_cv_splits(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    config: Config,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_index, validation_index) folds with no group in both sides."""
    n_folds = min(config.split.n_folds, len(pd.Series(groups).unique()))
    if n_folds < 2:
        raise ConfigError("need at least 2 distinct groups for cross-validation")
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=config.seed)
    for train_index, val_index in splitter.split(X, y.to_numpy(), groups.to_numpy()):
        yield np.asarray(train_index), np.asarray(val_index)


def assert_no_group_leakage(*named_groups: tuple[str, Sequence[str]]) -> None:
    """Raise if any group shows up in more than one of the supplied splits."""
    seen: dict[str, str] = {}
    for name, values in named_groups:
        for group in values:
            key = str(group)
            previous = seen.get(key)
            if previous is not None and previous != name:
                raise ConfigError(
                    "group leakage: "
                    f"{key!r} appears in both {previous!r} and {name!r}; the split is invalid "
                    "and every metric computed from it is fiction",
                    details={"group": key, "first": previous, "second": name},
                )
            seen[key] = name


def fold_report(
    y: pd.Series,
    holdout: Holdout,
    groups: pd.Series,
) -> dict[str, dict[str, object]]:
    """Per-fold row counts, class counts and group counts, for the published data card."""
    index_map = {
        "train": holdout.train_index,
        "val": holdout.val_index,
        "test": holdout.test_index,
    }
    report: dict[str, dict[str, object]] = {}
    for name, index in index_map.items():
        labels = pd.Series(y).iloc[index].astype(str)
        folded_groups = pd.Series(groups).iloc[index].astype(str)
        report[name] = {
            "rows": len(index),
            "groups": int(folded_groups.nunique()),
            "classes": {key: int(value) for key, value in sorted(labels.value_counts().items())},
        }
    return report
