"""Exoplanet candidate features.

Everything here is derived from quantities NASA publishes for a Kepler Object of
Interest. The point is not to add columns for the sake of it - each one answers a
question a validation scientist actually asks.

The most valuable feature is `prad_inconsistency`. Transit depth measures the ratio of
areas, so it implies a planet radius once you know the star's radius. If the catalogue's
own quoted planet radius disagrees with the radius implied by the depth, something is
wrong with the transit interpretation - which is precisely how blends and eclipsing
binaries betray themselves.

`koi_count` (how many candidates share that star) is kept but flagged in the learning
guide as a confound: multi-candidate systems were statistically easier to validate, so it
encodes a discovery-pipeline advantage, not physics.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..config import Config
from ..errors import DataError

# Unit conversion: stellar radii are quoted in solar radii, planetary radii in Earth radii.
SOLAR_RADIUS_IN_EARTH_RADII = 109.25

#: Columns fed to the model, in the exact order the trained artifact expects.
FEATURE_COLUMNS: tuple[str, ...] = (
    "log_period",
    "log_duration",
    "log_depth",
    "log_prad",
    "prad_inconsistency",
    "duration_over_period",
    "impact",
    "eccentricity",
    "inclination",
    "log_sma",
    "equilibrium_temp",
    "stellar_teff",
    "log_stellar_radius",
    "stellar_mass",
    "stellar_logg",
    "stellar_density",
    "n_transits",
    "model_snr",
    "max_single_event_snr",
    "kepler_mag",
    "koi_count",
)

#: Raw NASA columns the API must be given in order to build the features above.
RAW_INPUT_COLUMNS: tuple[str, ...] = (
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_impact",
    "koi_eccen",
    "koi_incl",
    "koi_sma",
    "koi_teq",
    "koi_steff",
    "koi_srad",
    "koi_smass",
    "koi_slogg",
    "koi_num_transits",
    "koi_model_snr",
    "koi_max_sngle_ev",
    "koi_kepmag",
    "koi_count",
)

#: Source column -> feature column for the values that pass straight through.
PASSTHROUGH: tuple[tuple[str, str], ...] = (
    ("koi_impact", "impact"),
    ("koi_eccen", "eccentricity"),
    ("koi_incl", "inclination"),
    ("koi_teq", "equilibrium_temp"),
    ("koi_steff", "stellar_teff"),
    ("koi_smass", "stellar_mass"),
    ("koi_slogg", "stellar_logg"),
    ("koi_num_transits", "n_transits"),
    ("koi_model_snr", "model_snr"),
    ("koi_max_sngle_ev", "max_single_event_snr"),
    ("koi_kepmag", "kepler_mag"),
    ("koi_count", "koi_count"),
)


def _log10_positive(values: pd.Series) -> pd.Series:
    """log10 for quantities that must be positive; non-positive or blank becomes NaN.

    Several of these span three or more orders of magnitude. Trees cope with the raw
    scale, logistic regression does not, and a log transform makes the relationships
    close to linear. NaN is preserved on purpose: the survey genuinely did not measure
    some of these, and gradient boosting can use that fact directly.
    """
    numeric = pd.to_numeric(values, errors="coerce")

    def convert(value: object) -> float:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return float("nan")
        number = float(value)
        if number <= 0.0:
            return float("nan")
        return math.log10(number)

    return numeric.map(convert).astype(float)


def derive(frame: pd.DataFrame) -> pd.DataFrame:
    """Add every engineered column. Works identically on one row or eighty thousand."""
    out = frame.copy()

    out["log_period"] = _log10_positive(out["koi_period"])
    out["log_duration"] = _log10_positive(out["koi_duration"])
    out["log_depth"] = _log10_positive(out["koi_depth"])
    out["log_prad"] = _log10_positive(out["koi_prad"])
    out["log_sma"] = _log10_positive(out["koi_sma"])
    out["log_stellar_radius"] = _log10_positive(out["koi_srad"])

    # Transit depth is a ratio of areas: depth = (Rp / Rs)^2, quoted in parts per million.
    depth_ppm = pd.to_numeric(out["koi_depth"], errors="coerce")
    stellar_radius = pd.to_numeric(out["koi_srad"], errors="coerce")
    radius_ratio = (depth_ppm * 1e-6).where(depth_ppm > 0) ** 0.5
    prad_from_depth = radius_ratio * stellar_radius * SOLAR_RADIUS_IN_EARTH_RADII
    quoted_prad = pd.to_numeric(out["koi_prad"], errors="coerce")
    out["prad_inconsistency"] = (quoted_prad - prad_from_depth).abs()

    period = pd.to_numeric(out["koi_period"], errors="coerce")
    duration = pd.to_numeric(out["koi_duration"], errors="coerce")
    out["duration_over_period"] = (duration / period).where(period > 0)

    stellar_mass = pd.to_numeric(out["koi_smass"], errors="coerce")
    out["stellar_density"] = (stellar_mass / stellar_radius.pow(3)).where(stellar_radius > 0)

    for source, target in PASSTHROUGH:
        out[target] = pd.to_numeric(out[source], errors="coerce")

    return out


def build_matrices(frame: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X, y, groups) ready for splitting.

    Raises rather than silently dropping a column: a renamed NASA column is a break in
    the data contract and must be loud, not absorbed.
    """
    label = config.exoplanets.label_column
    group = config.exoplanets.group_column
    required = (*RAW_INPUT_COLUMNS, label, group)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DataError(
            f"exoplanet frame is missing required columns: {missing}",
            details={"missing": missing},
        )

    derived = derive(frame)
    features = derived[list(FEATURE_COLUMNS)].astype(float)
    targets = derived[label].astype(str).str.strip()
    groups = derived[group].astype(str)

    if not all(str(dtype).startswith("float") for dtype in features.dtypes):
        bad = [name for name, dtype in features.dtypes.items() if not str(dtype).startswith("float")]
        raise DataError(f"non-numeric feature columns: {bad}", details={"columns": bad})

    return (
        features.reset_index(drop=True),
        targets.reset_index(drop=True),
        groups.reset_index(drop=True),
    )


def frame_from_payload(payload: dict) -> pd.DataFrame:
    """Build the one-row raw frame the API needs before calling `derive`.

    Absent keys become NaN, which is the same treatment a blank catalogue cell gets, so
    a partial prediction request behaves exactly like a partially-measured object.
    """
    row = {column: payload.get(column, np.nan) for column in RAW_INPUT_COLUMNS}
    return pd.DataFrame([row])
