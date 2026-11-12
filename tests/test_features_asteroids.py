"""Near-Earth object features: the absolute-magnitude -> diameter physics, event
aggregation, and the seasonal encoding.

The headline numbers in the README depend on `D = 1329 / sqrt(p_v) * 10^(-H/5)` being
implemented correctly, so the implementation is checked against Ceres, whose diameter and
albedo NASA publishes independently. That is a real astrophysical test, not a tautology:
if the constant or the exponent were wrong the Ceres assertion would fail even though the
code still "runs".
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from nasa_space_ml.errors import DataError
from nasa_space_ml.features import asteroids as features

# --------------------------------------------------------------- the magnitude relation


def test_diameter_of_ceres_matches_the_published_value() -> None:
    """Ceres: H = 3.34, geometric albedo 0.09, mean diameter 939 km (NASA planetary facts).
    The formula returns ~951 km, a 1.3% difference that comes from Ceres not being a perfect
    sphere at a single albedo. Anything outside a few percent means the algebra is wrong."""
    computed = features.diameter_from_absolute_magnitude(3.34, 0.09)
    assert computed == pytest.approx(939.0, rel=0.02)


def test_diameter_halves_when_albedo_quadruples() -> None:
    """D goes as 1/sqrt(albedo): a four-times-reflective body of the same brightness is
    half the size."""
    dark = features.diameter_from_absolute_magnitude(18.0, 0.05)
    bright = features.diameter_from_absolute_magnitude(18.0, 0.20)
    assert dark == pytest.approx(2.0 * bright, rel=1e-9)


def test_fainter_objects_are_smaller() -> None:
    assert features.diameter_from_absolute_magnitude(
        20.0, 0.14
    ) > features.diameter_from_absolute_magnitude(24.0, 0.14)


def test_five_magnitudes_shrink_the_diameter_tenfold() -> None:
    """The exponent is H/5, so +5 mag is exactly /10 in diameter. This is the property that
    makes the feature a straight line in log space, which is why trees like it."""
    big = features.diameter_from_absolute_magnitude(15.0, 0.14)
    small = features.diameter_from_absolute_magnitude(20.0, 0.14)
    assert small == pytest.approx(big / 10.0, rel=1e-9)


def test_missing_magnitude_gives_no_diameter() -> None:
    for value in (None, math.nan, "n/a"):
        assert features.diameter_from_absolute_magnitude(value, 0.14) is None


def test_albedo_round_trips_through_diameter() -> None:
    """implied_albedo is the inverse operation, so a self-consistent row must give back the
    albedo we assumed. The pair is what lets the model see 'this diameter and this
    magnitude disagree'."""
    for magnitude in (17.0, 21.5, 26.0):
        diameter = features.diameter_from_absolute_magnitude(magnitude, 0.14)
        assert features.albedo_from_diameter(magnitude, diameter) == pytest.approx(0.14, rel=1e-9)


def test_impossible_albedo_is_reported_rather_than_clamped() -> None:
    """A one-metre body cannot be as faint as H = 26 unless it reflects more than it
    receives. The function returns the absurd number instead of clipping to 1, because
    "the quoted size and magnitude disagree" is information the model is allowed to use."""
    absurd = features.albedo_from_diameter(26.0, 0.001)
    assert absurd is not None and absurd > 1.0


def test_albedo_of_ceres_comes_back_from_its_published_pair() -> None:
    assert features.albedo_from_diameter(3.34, 951.4) == pytest.approx(0.09, rel=0.01)


@pytest.mark.parametrize("bad", [None, math.nan, 0.0, -1.0, ""])
def test_albedo_of_an_unmeasurable_diameter_is_none(bad: object) -> None:
    assert features.albedo_from_diameter(20.0, bad) is None


def test_lunar_distance_constant_is_the_right_value() -> None:
    assert pytest.approx(0.00257, abs=1e-5) == features.LUNAR_DISTANCE_AU


# --------------------------------------------------------------------- aggregation


def test_one_row_per_object(cad_events: pd.DataFrame, asteroid_object_frame: pd.DataFrame) -> None:
    assert len(asteroid_object_frame) == cad_events["des"].nunique()
    assert asteroid_object_frame["des"].is_unique


def test_moid_proxy_is_the_smallest_closest_approach(asteroid_object_frame: pd.DataFrame) -> None:
    """The object that appears three times must carry the minimum of its approaches, not
    the last one and not the mean: a hazard is defined by the worst case."""
    first = asteroid_object_frame.loc[asteroid_object_frame["des"] == "2020 NP00"].iloc[0]
    assert float(first["moid_proxy_au"]) == pytest.approx(0.0040)


def test_approach_and_orbit_solution_counts(
    cad_events: pd.DataFrame, asteroid_object_frame: pd.DataFrame
) -> None:
    counts = cad_events.groupby("des").size()
    by_des = asteroid_object_frame.set_index("des")
    assert float(by_des.loc["2020 NP00", "n_approaches"]) == float(counts["2020 NP00"]) == 3
    assert float(by_des.loc["2020 NP01", "n_approaches"]) == 1.0
    assert float(by_des.loc["2020 NP00", "n_orbit_solutions"]) == 3.0
    assert float(by_des.loc["2020 NP01", "n_orbit_solutions"]) == 1.0


def test_uncertainty_ratio_comes_from_the_closest_approach(
    asteroid_object_frame: pd.DataFrame,
) -> None:
    """dist_min / dist_max on the chosen event; the fixture uses 1.6 as the spread, so the
    ratio is fixed and checkable."""
    row = asteroid_object_frame.iloc[0]
    assert float(row["uncertainty_ratio_at_closest"]) == pytest.approx(1 / 1.6, rel=1e-9)


def test_velocity_and_timing_come_from_the_same_event_as_the_moid(
    cad_events: pd.DataFrame, asteroid_object_frame: pd.DataFrame
) -> None:
    """Picking the closest approach by one column and then reading velocity from a
    different row would create a feature that never existed in the data."""
    closest = cad_events.loc[cad_events.groupby("des")["dist_min"].idxmin()]
    by_des = asteroid_object_frame.set_index("des")
    for _, event in closest.iterrows():
        assert float(by_des.loc[event["des"], "v_rel_at_closest"]) == pytest.approx(
            float(event["v_rel"])
        )


def test_median_magnitude_is_used_not_the_last_event(cad_events: pd.DataFrame, config) -> None:
    """NP00's three events share an H here, but the aggregation is a median because JPL
    republishes magnitudes as orbits improve, and the mean of an old and a new solution is
    a value that was never true."""
    events = cad_events.copy()
    events.loc[events["des"] == "2020 NP00", "h"] = [20.0, 21.4, 30.0]
    object_frame = features.aggregate_events(events, config)
    assert float(
        object_frame.set_index("des").loc["2020 NP00", "absolute_magnitude_h"]
    ) == pytest.approx(21.4)


def test_observation_span_is_zero_for_a_single_event(asteroid_object_frame: pd.DataFrame) -> None:
    assert (
        float(asteroid_object_frame.set_index("des").loc["2020 NP01", "observation_span_years"])
        == 0.0
    )
    assert (
        float(asteroid_object_frame.set_index("des").loc["2020 NP00", "observation_span_years"])
        > 0.0
    )


def test_within_lunar_distance_counts_approaches_not_objects(
    cad_events: pd.DataFrame, config
) -> None:
    """Add one approach well inside the Moon's orbit and check it moves the count for that
    object while leaving the approach count alone."""
    extra = cad_events.iloc[0].copy()
    extra["des"] = "2020 NP05"
    extra["dist_min"] = 0.0001
    frame = features.aggregate_events(
        pd.concat([cad_events, extra.to_frame().T], ignore_index=True), config
    )
    row = frame.set_index("des").loc["2020 NP05"]
    assert float(row["n_approaches_within_lunar"]) == 1.0
    assert float(row["n_approaches"]) == 2.0


def test_missing_diameter_stays_missing(cad_events: pd.DataFrame, config) -> None:
    """Only 4 of the 12 fixture objects have a reported diameter, so the rest must be NaN -
    inventing a value here would teach the model that most NEOs are ~0.31 km."""
    frame = features.aggregate_events(cad_events, config)
    assert frame["diameter_reported_km"].isna().sum() >= 8


# ------------------------------------------------------------------ derived features


def test_derive_adds_the_derived_contract(asteroid_object_frame: pd.DataFrame) -> None:
    derived = features.derive(asteroid_object_frame)
    assert set(features.FEATURE_COLUMNS) <= set(derived.columns)
    assert len(derived) == len(asteroid_object_frame)


def test_log_moid_tracks_the_raw_moid(asteroid_object_frame: pd.DataFrame) -> None:
    derived = features.derive(asteroid_object_frame)
    row = derived.iloc[0]
    assert float(row["log_moid"]) == pytest.approx(math.log10(float(row["moid_proxy_au"])))


def test_lunar_distance_scaling(asteroid_object_frame: pd.DataFrame) -> None:
    derived = features.derive(asteroid_object_frame)
    row = derived.iloc[0]
    assert float(row["moid_lunar_distances"]) == pytest.approx(
        float(row["moid_proxy_au"]) / 0.00257
    )


@pytest.mark.parametrize(
    ("raw", "month"),
    [
        ("2020-Jan-05 02:06", 1),
        ("2020-01-05 02:06", 1),
        ("2020-Jul-19 11:00", 7),
        ("2020-Dec-31", 12),
    ],
)
def test_month_parsing_accepts_both_date_styles(
    raw: str, month: int, cad_events: pd.DataFrame, config
) -> None:
    """CAD writes `2020-Jan-05`; ISO is what a hand-built request looks like. Both must
    parse, or the seasonal feature is silently all-NaN - which is exactly the bug this
    project actually hit, and this test is why it stayed fixed."""
    events = cad_events.copy()
    events.loc[events["des"] == "2020 NP01", "cd"] = raw
    row = (
        features.derive(features.aggregate_events(events, config)).set_index("des").loc["2020 NP01"]
    )
    angle = 2.0 * math.pi * (month - 1) / 12.0
    assert float(row["closest_month_sin"]) == pytest.approx(math.sin(angle), abs=1e-9)
    assert float(row["closest_month_cos"]) == pytest.approx(math.cos(angle), abs=1e-9)


def test_seasonal_encoding_is_a_unit_circle(
    asteroid_object_frame: pd.DataFrame, cad_events: pd.DataFrame, config
) -> None:
    """sin^2 + cos^2 = 1 for every parsed month, and a nonsense month must produce NaN on
    both halves rather than a point off the circle."""
    derived = features.derive(asteroid_object_frame)
    mask = derived["closest_month_sin"].notna()
    assert mask.any()
    total = (
        derived.loc[mask, "closest_month_sin"] ** 2 + derived.loc[mask, "closest_month_cos"] ** 2
    )
    assert np.allclose(total.to_numpy(), 1.0)


def test_unparsable_month_gives_nan_on_both_components(cad_events: pd.DataFrame, config) -> None:
    events = cad_events.copy()
    events.loc[events["des"] == "2020 NP02", "cd"] = "whenever"
    derived = features.derive(features.aggregate_events(events, config))
    row = derived.set_index("des").loc["2020 NP02"]
    assert math.isnan(float(row["closest_month_sin"]))
    assert math.isnan(float(row["closest_month_cos"]))


# ------------------------------------------------------------------ matrix assembly


def test_labels_become_the_configured_strings(config, asteroid_object_frame: pd.DataFrame) -> None:
    X, y, groups = features.build_matrices(asteroid_object_frame, config)
    assert set(y.unique()) == {config.asteroids.labels[0], config.asteroids.labels[1]}
    assert list(X.columns) == list(features.FEATURE_COLUMNS)
    assert groups.tolist() == asteroid_object_frame["des"].tolist()


def test_positive_label_follows_the_rule_thresholds(
    config, asteroid_object_frame: pd.DataFrame
) -> None:
    _, y, groups = features.build_matrices(asteroid_object_frame, config)
    hazardous = set(y[groups.isin([f"2020 NP{n:02d}" for n in range(4)])])
    assert hazardous == {config.asteroids.labels[1]}
    assert config.asteroids.labels[1] == "PHA"


def test_missing_base_column_names_the_fetch_command(
    config, asteroid_object_frame: pd.DataFrame
) -> None:
    broken = asteroid_object_frame.drop(columns=["moid_proxy_au"])
    with pytest.raises(DataError, match="re-run `nasa-space-ml fetch`") as excinfo:
        features.build_matrices(broken, config)
    assert "moid_proxy_au" in excinfo.value.details["missing"]


def test_derived_columns_are_not_required_from_the_caller(
    config, asteroid_object_frame: pd.DataFrame
) -> None:
    """`log_moid` is produced by `derive`, so requiring it here would make the catalogue
    path and the single-request path need different inputs - the definition of skew."""
    assert "log_moid" in features.FEATURE_COLUMNS
    assert "log_moid" not in features.BASE_INPUT_COLUMNS
    X, _, _ = features.build_matrices(asteroid_object_frame, config)
    assert X["log_moid"].notna().any()


def test_feature_count_is_nineteen() -> None:
    assert len(features.FEATURE_COLUMNS) == 19


# ------------------------------------------------------------------- serving one object


def test_payload_defaults_to_a_missing_designation(config) -> None:
    frame = features.frame_from_payload({"absolute_magnitude_h": 22.0}, config)
    assert frame.loc[0, "des"] == "api-input"


def test_a_prediction_request_has_no_label_because_the_label_is_the_answer(config) -> None:
    """`frame_from_payload` deliberately omits `pha_flag`: at serving time that column is
    what we are trying to produce. `build_matrices` therefore refuses the raw payload
    frame, which is the correct refusal."""
    frame = features.frame_from_payload({"absolute_magnitude_h": 22.0}, config)
    assert features.LABEL_COLUMN not in frame.columns
    with pytest.raises(DataError, match=features.LABEL_COLUMN):
        features.build_matrices(frame, config)


def test_diameter_is_derived_from_the_configured_albedo_not_the_request(config) -> None:
    """If the caller could set the albedo, the same asteroid would score differently
    depending on who asked. The value is owned by settings."""
    payload = {"absolute_magnitude_h": 21.0, "geometric_albedo": 0.9}
    frame = features.frame_from_payload(payload, config)
    expected = features.diameter_from_absolute_magnitude(
        21.0, config.asteroids.default_geometric_albedo
    )
    assert float(frame.loc[0, "diameter_from_h_km"]) == pytest.approx(expected)
    assert "geometric_albedo" not in frame.columns


def test_sparse_payload_reaches_a_full_feature_vector(config) -> None:
    frame = features.frame_from_payload(
        {"moid_proxy_au": 0.03, "absolute_magnitude_h": 23.0}, config
    )
    frame[features.LABEL_COLUMN] = False
    X, y, _groups = features.build_matrices(frame, config)
    assert X.shape == (1, len(features.FEATURE_COLUMNS))
    assert y.iloc[0] == config.asteroids.labels[0]


def test_one_request_is_featurised_exactly_like_a_catalogue_row(
    config, asteroid_object_frame: pd.DataFrame
) -> None:
    """A real object and a hand-entered object with the same numbers must produce the same
    features, or the console is lying to the person using it. This is the train/serve skew
    test for the asteroid channel."""
    catalogue = asteroid_object_frame.iloc[0]
    payload = {column: catalogue[column] for column in features.BASE_INPUT_COLUMNS}
    payload["closest_approach_month"] = catalogue["closest_approach_month"]
    payload["designation"] = "2020 NP00"
    rebuilt = features.frame_from_payload(payload, config)

    left = features.derive(asteroid_object_frame.iloc[[0]])[list(features.FEATURE_COLUMNS)]
    right = features.derive(rebuilt)[list(features.FEATURE_COLUMNS)]
    pd.testing.assert_frame_equal(
        left.reset_index(drop=True).astype(float), right.reset_index(drop=True).astype(float)
    )
