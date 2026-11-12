"""Feature engineering for near-Earth objects (event rows -> one row per asteroid).

The important structural decision is in `aggregate_events`. JPL's CAD endpoint hands us one
row per close-approach *event*, but hazard status is a property of the *object*: 65803
Didymos is a PHA whether we look at its 1996 or its 2199 approach. So the unit of analysis
has to be the object, and every event feature becomes a summary over that object's events.

Everything else here is physics or bookkeeping:

  * `moid_proxy_au` - the smallest `dist_min` across the window. Real MOID is an orbital
    element computed from the full solution; this is the closest approach we were *told*
    about, so it is a lower bound on nothing and an upper bound on the true MOID only if
    the window contains it. Documented in the model card, not hidden.
  * `diameter_from_h_km` - NASA's standard absolute-magnitude conversion
    D = 1329 / sqrt(p_v) * 10^(-H/5) with the assumed geometric albedo p_v from settings.
  * `implied_albedo` - the same equation solved for p_v using JPL's *reported* diameter.
    When the reported diameter and the assumed-albedo diameter disagree, the object is
    either darker or brighter than 0.14, which correlates with composition and with how
    well observed it is. That disagreement is the feature; we do not pick a winner.
  * `n_orbit_solutions` - distinct `orbit_id` values. Objects whose orbit has been
    re-fitted many times are long-tracked and well-determined. This is an honest
    "data quality" signal, and it is also the feature most likely to be a shortcut, which
    is why it is reported separately in the permutation-importance chart.

The aggregation loop is a plain Python loop over groups. It reads clearly, it is a couple
of seconds for the whole catalogue, and it does not depend on pandas groupby semantics
that have already bitten this project once.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from ..errors import DataError

LUNAR_DISTANCE_AU = 0.00257
DIAMETER_CONSTANT_KM = 1329.0
SECONDS_PER_YEAR = 365.25 * 86400.0
DAYS_PER_YEAR = 365.25

FEATURE_COLUMNS: tuple[str, ...] = (
    "moid_proxy_au",
    "log_moid",
    "moid_lunar_distances",
    "n_approaches",
    "n_approaches_within_lunar",
    "v_rel_at_closest",
    "v_inf_at_closest",
    "uncertainty_ratio_at_closest",
    "absolute_magnitude_h",
    "diameter_from_h_km",
    "log_diameter_from_h",
    "diameter_reported_km",
    "diameter_inconsistency_km",
    "implied_albedo",
    "n_orbit_solutions",
    "t_sigma_seconds_at_closest",
    "observation_span_years",
    "closest_month_sin",
    "closest_month_cos",
)

# Which of the above `derive` computes versus which must be supplied by the caller. Keeping
# this as an explicit set (rather than duplicating the list) means FEATURE_COLUMNS can only
# ever have one definition.
DERIVED_COLUMNS: frozenset[str] = frozenset(
    {"log_moid", "moid_lunar_distances", "log_diameter_from_h",
     "closest_month_sin", "closest_month_cos"}
)

BASE_INPUT_COLUMNS: tuple[str, ...] = tuple(
    column for column in FEATURE_COLUMNS if column not in DERIVED_COLUMNS
)

LABEL_COLUMN = "pha_flag"


def _log10_positive(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = np.full(len(numeric), np.nan, dtype=float)
    mask = numeric.notna().to_numpy() & (numeric.to_numpy(dtype=float, na_value=np.nan) > 0)
    result[mask] = np.log10(numeric.to_numpy(dtype=float, na_value=np.nan)[mask])
    return pd.Series(result, index=values.index, name=numeric.name)


def diameter_from_absolute_magnitude(h: Any, albedo: float) -> float | None:
    """NASA's standard H -> diameter relation, D_km = 1329/sqrt(p) * 10^(-H/5)."""
    if h is None or (isinstance(h, float) and math.isnan(h)):
        return None
    if albedo <= 0:
        raise DataError("geometric albedo must be positive", details={"albedo": albedo})
    return DIAMETER_CONSTANT_KM / math.sqrt(albedo) * 10.0 ** (-float(h) / 5.0)


def albedo_from_diameter(h: Any, diameter_km: Any) -> float | None:
    """Invert the same relation for p_v; outside (0, 1] it is not a physical albedo."""
    if h is None or diameter_km is None:
        return None
    try:
        h_value = float(h)
        diameter = float(diameter_km)
    except (TypeError, ValueError):
        return None
    if math.isnan(h_value) or math.isnan(diameter) or diameter <= 0:
        return None
    implied = (DIAMETER_CONSTANT_KM / diameter) ** 2 * 10.0 ** (-0.4 * h_value)
    if not math.isfinite(implied) or implied <= 0:
        return None
    return implied


_CAD_MONTHS: dict[str, int] = {
    name: number
    for number, name in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun",
         "jul", "aug", "sep", "oct", "nov", "dec"],
        start=1,
    )
}


def _month_from_cd(cd: Any) -> float | None:
    """CAD `cd` is a date-time string like `2020-Jan-01 02:06`; return 1..12 or None.

    Verified against the live feed, which uses `%Y-%b-%d`, not ISO dates. The numeric branch
    is kept anyway, so a future switch to `2020-01-01` degrades into working correctly
    rather than into a column of silent NaN.
    """
    text = str(cd or "").strip()
    if not text:
        return None
    bits = text.split()[0].split("-")
    if len(bits) != 3:
        return None
    middle = bits[1].lower()
    month = _CAD_MONTHS.get(middle)
    if month is None and middle.isdigit():
        month = int(middle)
    return float(month) if month is not None and 1 <= month <= 12 else None


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _median(values: list[float]) -> float | None:
    clean = sorted(values)
    if not clean:
        return None
    middle = len(clean) // 2
    if len(clean) % 2:
        return clean[middle]
    return (clean[middle - 1] + clean[middle]) / 2.0


def _aggregate_one(des: str, group: pd.DataFrame, albedo: float) -> dict[str, Any]:
    """Collapse one object's approach events into a single feature row."""
    closest_index = None
    best_distance = None
    for index, raw in group["dist_min"].items():
        value = _float(raw)
        if value is not None and (best_distance is None or value < best_distance):
            best_distance = value
            closest_index = index

    record: dict[str, Any] = {"des": des, LABEL_COLUMN: bool(group["pha_flag"].any())}
    record["moid_proxy_au"] = best_distance
    record["n_approaches"] = float(len(group))

    within_lunar = 0
    for raw in group["dist_min"]:
        value = _float(raw)
        if value is not None and value <= LUNAR_DISTANCE_AU:
            within_lunar += 1
    record["n_approaches_within_lunar"] = float(within_lunar)

    if closest_index is not None:
        closest = group.loc[closest_index]
        record["v_rel_at_closest"] = _float(closest.get("v_rel"))
        record["v_inf_at_closest"] = _float(closest.get("v_inf"))
        record["t_sigma_seconds_at_closest"] = _float(closest.get("t_sigma_seconds"))
        record["uncertainty_ratio_at_closest"] = _ratio(
            closest.get("dist_min"), closest.get("dist_max")
        )
        record["closest_approach_month"] = _month_from_cd(closest.get("cd"))
    else:
        record["v_rel_at_closest"] = None
        record["v_inf_at_closest"] = None
        record["t_sigma_seconds_at_closest"] = None
        record["uncertainty_ratio_at_closest"] = None
        record["closest_approach_month"] = _seasonal_fallback(group)

    magnitudes = [_float(value) for value in group["h"]]
    magnitudes = [value for value in magnitudes if value is not None]
    h_median = _median(magnitudes)
    record["absolute_magnitude_h"] = h_median

    diameters = [_float(value) for value in group["diameter"]]
    diameters = [value for value in diameters if value is not None]
    reported = _median(diameters)
    record["diameter_reported_km"] = reported

    from_h = (
        diameter_from_absolute_magnitude(h_median, albedo) if h_median is not None else None
    )
    record["diameter_from_h_km"] = from_h
    record["diameter_inconsistency_km"] = (
        abs(reported - from_h) if reported is not None and from_h is not None else None
    )

    implied = albedo_from_diameter(h_median, reported)
    record["implied_albedo"] = implied

    orbit_ids = group["orbit_id"].dropna().astype(str).unique() if "orbit_id" in group else []
    record["n_orbit_solutions"] = float(len(orbit_ids))

    spans = [_float(value) for value in group["jd"]]
    spans = [value for value in spans if value is not None]
    record["observation_span_years"] = (
        (max(spans) - min(spans)) / DAYS_PER_YEAR if len(spans) > 1 else 0.0
    )
    return record


def _seasonal_fallback(group: pd.DataFrame) -> float | None:
    months = [_month_from_cd(value) for value in group.get("cd", [])]
    months = [value for value in months if value is not None]
    return _median(months) if months else None


def _ratio(numerator: Any, denominator: Any) -> float | None:
    top = _float(numerator)
    bottom = _float(denominator)
    if top is None or bottom is None or bottom <= 0:
        return None
    return top / bottom


def aggregate_events(events: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Event-level CAD rows -> one row per asteroid, labelled by object PHA status."""
    required = {"des", "dist_min", "pha_flag"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise DataError(
            "CAD event table is missing required columns", details={"missing": missing}
        )
    albedo = config.asteroids.default_geometric_albedo
    frame = events.copy()
    frame["des"] = frame["des"].astype(str)
    records = [
        _aggregate_one(des, group, albedo)
        for des, group in frame.groupby("des", sort=True)
    ]
    return pd.DataFrame.from_records(records)


DERIVE_INPUTS: tuple[str, ...] = (
    "moid_proxy_au",
    "diameter_from_h_km",
    "closest_approach_month",
)


def derive(frame: pd.DataFrame) -> pd.DataFrame:
    """Object table -> model-ready columns. Adds derived quantities only, never drops.

    Missing base columns become NaN columns rather than a KeyError, because a single API
    request legitimately supplies a subset of what the catalogue aggregation produces.
    """
    out = frame.copy()
    for column in DERIVE_INPUTS:
        if column not in out.columns:
            out[column] = np.nan

    moid = pd.to_numeric(out["moid_proxy_au"], errors="coerce")
    out["log_moid"] = _log10_positive(moid)
    out["moid_lunar_distances"] = moid / LUNAR_DISTANCE_AU
    diameter = pd.to_numeric(out["diameter_from_h_km"], errors="coerce")
    out["log_diameter_from_h"] = _log10_positive(diameter)

    month = pd.to_numeric(out["closest_approach_month"], errors="coerce")
    in_season = month.between(1.0, 12.0)
    angle = 2.0 * math.pi * (month - 1.0) / 12.0
    out["closest_month_sin"] = np.where(in_season, np.sin(angle), np.nan)
    out["closest_month_cos"] = np.where(in_season, np.cos(angle), np.nan)
    return out


def build_matrices(frame: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Object table -> (X, y, groups). Groups are designations, so one object never splits.

    Only the *base* columns are required from the caller: `derive` produces the derived
    ones, which is why the catalogue aggregation and a one-row API payload can share this
    function unchanged.
    """
    required = [*list(BASE_INPUT_COLUMNS), LABEL_COLUMN, "des"]
    missing = sorted(column for column in required if column not in frame.columns)
    if missing:
        raise DataError(
            "asteroid feature table is missing columns; re-run `nasa-space-ml fetch` and "
            "check the CAD field list",
            details={"missing": missing},
        )
    derived = derive(frame)
    features = pd.DataFrame(index=derived.index)
    for column in FEATURE_COLUMNS:
        features[column] = pd.to_numeric(derived[column], errors="coerce")
    flags = derived[LABEL_COLUMN].astype(bool)
    positive, negative = config.asteroids.labels[1], config.asteroids.labels[0]
    targets = flags.map({True: positive, False: negative}).astype(str)
    groups = derived["des"].astype(str)
    return (
        features.reset_index(drop=True),
        targets.reset_index(drop=True),
        groups.reset_index(drop=True),
    )


def frame_from_payload(payload: dict[str, Any], config: Config) -> pd.DataFrame:
    """One API request -> the same object-level row shape `build_matrices` expects.

    Every base feature column is created, defaulting to None, so a sparse request and a
    full catalogue row travel through identical code. The albedo used for the H -> diameter
    conversion comes from settings, never from the caller: if the client could choose it,
    the same object would score differently depending on who asked.
    """
    albedo = config.asteroids.default_geometric_albedo
    row: dict[str, Any] = {"des": str(payload.get("designation") or "api-input")}
    for column in BASE_INPUT_COLUMNS:
        row[column] = payload.get(column)
    row["closest_approach_month"] = payload.get("closest_approach_month")

    magnitude = _float(row.get("absolute_magnitude_h"))
    reported = _float(row.get("diameter_reported_km"))
    if row.get("diameter_from_h_km") is None and magnitude is not None:
        row["diameter_from_h_km"] = diameter_from_absolute_magnitude(magnitude, albedo)
    if row.get("implied_albedo") is None:
        row["implied_albedo"] = albedo_from_diameter(magnitude, reported)
    if row.get("diameter_inconsistency_km") is None:
        from_h = _float(row.get("diameter_from_h_km"))
        row["diameter_inconsistency_km"] = (
            abs(reported - from_h) if reported is not None and from_h is not None else None
        )
    return pd.DataFrame.from_records([row])
