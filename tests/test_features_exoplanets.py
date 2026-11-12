"""Exoplanet feature engineering.

The claims under test are the physical ones. If `prad_inconsistency` stops being the
comparison between a quoted radius and the radius implied by the transit depth, then it
stops being a leak-free way to spot blended eclipsing binaries, and the model quietly
loses the feature that carries the most astrophysics in it.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from nasa_space_ml.errors import DataError
from nasa_space_ml.features import exoplanets as features


def _single(**overrides: object) -> pd.DataFrame:
    row = dict.fromkeys(features.RAW_INPUT_COLUMNS, np.nan)
    row.update(
        {
            "koi_period": 3.0,
            "koi_duration": 3.0,
            "koi_depth": 10510.0,
            "koi_prad": 11.2,
            "koi_srad": 1.0,
            "koi_sma": 0.05,
            "koi_smass": 1.0,
            "koi_impact": 0.3,
            "koi_eccen": 0.0,
            "koi_incl": 89.9,
            "koi_teq": 1000.0,
            "koi_steff": 5770.0,
            "koi_slogg": 4.44,
            "koi_num_transits": 20,
            "koi_model_snr": 400.0,
            "koi_max_sngle_ev": 25.0,
            "koi_kepmag": 13.0,
            "koi_count": 1,
        }
    )
    row.update(overrides)
    return pd.DataFrame([row])


# --------------------------------------------------------------------- log transforms


def test_log_period_is_base_ten_of_the_raw_value() -> None:
    derived = features.derive(_single(koi_period=100.0))
    assert float(derived.loc[0, "log_period"]) == pytest.approx(2.0)


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_non_positive_values_become_missing_not_the_log_of_an_error(value: float) -> None:
    """log(0) is -inf and log(-1) is undefined; either one poisoning a column would drag
    a whole tree ensemble off the rails, so "not physically meaningful" maps to NaN."""
    derived = features.derive(_single(koi_duration=value))
    assert math.isnan(float(derived.loc[0, "log_duration"]))


def test_missing_values_stay_missing() -> None:
    derived = features.derive(_single(koi_sma=np.nan))
    assert math.isnan(float(derived.loc[0, "log_sma"]))


def test_log_transform_preserves_ordering() -> None:
    """A monotone transform must not invert a comparison, or the trees split on nonsense."""
    short = features.derive(_single(koi_period=1.0))["log_period"].iloc[0]
    long = features.derive(_single(koi_period=1000.0))["log_period"].iloc[0]
    assert short < long


# ------------------------------------------------------- the transit-depth consistency


def test_consistent_planet_row_has_almost_no_radius_inconsistency() -> None:
    """A Jupiter-radius planet (11.2 R_earth) transiting a 1 R_sun star blocks
    (11.2 / 109.25)^2 = 1.05% of the light, i.e. ~10510 ppm. Feed the archive's own
    arithmetic back in and the inconsistency must be ~0."""
    derived = features.derive(_single(koi_depth=10510.0, koi_prad=11.2, koi_srad=1.0))
    assert float(derived.loc[0, "prad_inconsistency"]) == pytest.approx(0.0, abs=0.05)


def test_radius_quoted_far_too_large_is_flagged() -> None:
    """This is the eclipsing-binary signature: a quoted 30 R_earth "planet" blocking only
    100 ppm around a solar star is not a planet."""
    derived = features.derive(_single(koi_depth=100.0, koi_prad=30.0, koi_srad=1.0))
    inconsistency = float(derived.loc[0, "prad_inconsistency"])
    implied = 30.0 - inconsistency
    assert inconsistency > 25.0
    assert implied == pytest.approx(math.sqrt(100e-6) * 109.25, abs=0.05)


def test_inconsistency_is_symmetric_about_the_implied_radius() -> None:
    """Over-estimating and under-estimating the radius are equally suspicious, so the
    feature is a distance from the implied value rather than a signed error."""
    implied = math.sqrt(10510e-6) * 1.0 * features.SOLAR_RADIUS_IN_EARTH_RADII
    low = float(
        features.derive(_single(koi_depth=10510.0, koi_prad=implied - 3.0))[
            "prad_inconsistency"
        ].iloc[0]
    )
    high = float(
        features.derive(_single(koi_depth=10510.0, koi_prad=implied + 3.0))[
            "prad_inconsistency"
        ].iloc[0]
    )
    assert low == pytest.approx(high, abs=1e-9)
    assert low == pytest.approx(3.0, abs=1e-9)


def test_unmeasurable_depth_gives_no_consistency_verdict() -> None:
    """Zero depth means nothing was detected, so there is no implied radius to disagree
    with. The honest answer is missing, not "12 R_earth wrong": mapping it to a large
    inconsistency would hand the model a feature that is really a detector failure."""
    derived = features.derive(_single(koi_depth=0.0, koi_prad=12.0, koi_srad=1.0))
    assert math.isnan(float(derived.loc[0, "prad_inconsistency"]))
    negative = features.derive(_single(koi_depth=-5.0, koi_prad=12.0, koi_srad=1.0))
    assert math.isnan(float(negative.loc[0, "prad_inconsistency"]))


def test_duration_over_period_is_a_dimensionless_ratio() -> None:
    derived = features.derive(_single(koi_duration=1.5, koi_period=3.0))
    assert float(derived.loc[0, "duration_over_period"]) == pytest.approx(0.5)


def test_stellar_density_uses_mass_over_radius_cubed() -> None:
    derived = features.derive(_single(koi_smass=2.0, koi_srad=1.0))
    assert float(derived.loc[0, "stellar_density"]) == pytest.approx(2.0)
    doubled = features.derive(_single(koi_smass=2.0, koi_srad=2.0))
    assert float(doubled.loc[0, "stellar_density"]) == pytest.approx(0.25)


def test_solar_radius_conversion_constant_is_astrophysically_right() -> None:
    assert pytest.approx(109.25, abs=0.01) == features.SOLAR_RADIUS_IN_EARTH_RADII


# ------------------------------------------------------------------------- passthrough


def test_passthrough_columns_are_numeric_copies() -> None:
    derived = features.derive(_single())
    assert float(derived.loc[0, "impact"]) == pytest.approx(0.3)
    assert float(derived.loc[0, "stellar_teff"]) == pytest.approx(5770.0)
    assert float(derived.loc[0, "n_transits"]) == pytest.approx(20.0)


def test_source_and_target_names_are_both_present() -> None:
    derived = features.derive(_single())
    for source, target in features.PASSTHROUGH:
        assert source in derived.columns
        assert target in derived.columns


# --------------------------------------------------------------------- matrix assembly


def test_derive_produces_every_feature_column(exo_frame: pd.DataFrame) -> None:
    derived = features.derive(exo_frame)
    assert set(features.FEATURE_COLUMNS) <= set(derived.columns)


def test_build_matrices_shapes_and_alignment(config, exo_frame: pd.DataFrame) -> None:
    X, y, groups = features.build_matrices(exo_frame, config)
    assert list(X.columns) == list(features.FEATURE_COLUMNS)
    assert len(X) == len(y) == len(groups) == len(exo_frame)
    assert list(X.index) == list(range(len(X)))
    assert list(y.index) == list(range(len(y)))


def test_targets_are_the_configured_label_strings(config, exo_matrices) -> None:
    _, y, _ = exo_matrices
    assert set(y.unique()) <= set(config.exoplanets.labels)


def test_groups_are_host_star_ids(config, exo_frame: pd.DataFrame) -> None:
    _, _, groups = features.build_matrices(exo_frame, config)
    assert groups.astype(str).nunique() == exo_frame["kepid"].nunique()


def test_features_are_all_float_dtypes(exo_matrices) -> None:
    X, _, _ = exo_matrices
    assert all(str(dtype).startswith("float") for dtype in X.dtypes)


def test_no_feature_column_is_a_label_in_disguise() -> None:
    """`koi_pdisposition` and friends must not appear even as *derived* columns, because
    build_matrices selects by name and a rename elsewhere would let one sneak in."""
    forbidden = {"koi_pdisposition", "koi_vet_stat", "koi_quarters", "koi_delivname"}
    assert not set(features.FEATURE_COLUMNS) & forbidden
    for name in features.FEATURE_COLUMNS:
        assert "disposition" not in name.lower()
        assert "pdisp" not in name.lower()


def test_missing_required_column_raises_with_the_fetch_hint(
    config, exo_frame: pd.DataFrame
) -> None:
    broken = exo_frame.drop(columns=["koi_depth"])
    with pytest.raises(DataError, match="missing required columns"):
        features.build_matrices(broken, config)


def test_group_column_missing_is_also_fatal(config, exo_frame: pd.DataFrame) -> None:
    with pytest.raises(DataError):
        features.build_matrices(exo_frame.drop(columns=["kepid"]), config)


def test_label_column_missing_is_also_fatal(config, exo_frame: pd.DataFrame) -> None:
    with pytest.raises(DataError):
        features.build_matrices(exo_frame.drop(columns=["koi_disposition"]), config)


# ------------------------------------------------------------------- serving one row


def test_frame_from_payload_fills_absent_keys_with_nan() -> None:
    """A partial prediction request must behave exactly like a partially-measured object,
    otherwise the model sees a distribution it was not trained on."""
    frame = features.frame_from_payload({"koi_period": 3.4, "koi_depth": 500.0})
    assert len(frame) == 1
    assert float(frame.loc[0, "koi_period"]) == pytest.approx(3.4)
    assert math.isnan(float(frame.loc[0, "koi_steff"]))


def test_frame_from_payload_can_be_featurised_end_to_end(config) -> None:
    frame = features.frame_from_payload({"koi_period": 3.4, "koi_depth": 500.0, "koi_prad": 4.0})
    derived = features.derive(frame)
    assert set(features.FEATURE_COLUMNS) <= set(derived.columns)
    assert float(derived.loc[0, "log_period"]) == pytest.approx(math.log10(3.4))


def test_payload_ignores_unexpected_keys() -> None:
    """An unknown key in a request must not become a phantom feature column."""
    frame = features.frame_from_payload({"koi_period": 3.0, "totally_made_up": 1.0})
    assert "totally_made_up" not in frame.columns


def test_feature_count_matches_the_shipped_artifact(config) -> None:
    """21 features, and the number is not decoration: it is the contract the registry
    guard enforces when the API loads the model. A column added here without retraining
    is caught at load time rather than producing silently wrong predictions."""
    assert len(features.FEATURE_COLUMNS) == 21
    one_row = _single().iloc[0].to_dict()
    one_row.update({"koi_disposition": "CONFIRMED", "kepid": 4000000000})
    X, _, _ = features.build_matrices(pd.DataFrame([one_row]), config)
    assert X.shape == (1, len(features.FEATURE_COLUMNS))
