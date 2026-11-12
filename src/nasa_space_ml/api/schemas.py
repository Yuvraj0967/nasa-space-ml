"""Request and response contracts.

Every number a caller can send carries a physical bound, and the bounds come from what the
instrument can actually measure - not from what the model happens to tolerate. A Kepler
period of -3 days, a depth of 900,000 ppm (that is a 95% transit: a star being eclipsed by a
blackboard) and an inclination of 400 degrees are all things real API clients send within a
week of a service going public. Rejecting them at the boundary is cheaper than explaining
why the model returned 0.5000001 for them.

Bounds are deliberately generous. They are "no physically existing object is refused", not
"only values in the training set are allowed" - a service that only accepts its own past is a
lookup table, not a model. The one place where extrapolation genuinely hurts is noted in the
response as `outside_training_range`, so a caller can see it without being blocked.

`PredictResponse` always echoes the feature vector that was actually scored plus the model's
own provenance (`model_trained_utc`, `dataset_fingerprint`). That is not decoration: it is
what lets somebody reproduce a suspicious prediction two weeks later instead of shrugging.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExoplanetRequest(BaseModel):
    """One Kepler Objects of Interest candidate, as measured by the pipeline."""

    model_config = ConfigDict(extra="forbid")

    kepoi_name: str | None = Field(default=None, description="candidate id, e.g. K00106.02")
    hostname: str | None = Field(default=None, description="host star name, if known")
    koi_period: float | None = Field(
        default=None, ge=1e-4, le=1e5, description="orbital period, days"
    )
    koi_duration: float | None = Field(
        default=None, ge=0.0, le=1e4, description="transit duration, hours"
    )
    koi_depth: float | None = Field(default=None, ge=0.0, le=1e6, description="transit depth, ppm")
    koi_prad: float | None = Field(
        default=None, ge=0.0, le=1e5, description="planet radius, Earth radii"
    )
    koi_impact: float | None = Field(default=None, ge=0.0, le=2.0, description="impact parameter b")
    koi_eccen: float | None = Field(default=None, ge=0.0, lt=1.0, description="eccentricity")
    koi_incl: float | None = Field(
        default=None, ge=0.0, le=180.0, description="inclination, degrees"
    )
    koi_sma: float | None = Field(default=None, ge=0.0, le=1e3, description="semi-major axis, au")
    koi_teq: float | None = Field(
        default=None, ge=-273.15, le=1e5, description="equilibrium temperature, K"
    )
    koi_steff: float | None = Field(
        default=None, ge=500.0, le=1e5, description="stellar effective temperature, K"
    )
    koi_srad: float | None = Field(
        default=None, gt=0.0, le=5e3, description="stellar radius, solar radii"
    )
    koi_smass: float | None = Field(
        default=None, gt=0.0, le=1e3, description="stellar mass, solar masses"
    )
    koi_slogg: float | None = Field(
        default=None, ge=0.0, le=7.0, description="stellar surface gravity, log g"
    )
    koi_num_transits: float | None = Field(
        default=None, ge=0.0, le=1e4, description="observed transits"
    )
    koi_model_snr: float | None = Field(
        default=None, ge=0.0, le=1e7, description="model signal-to-noise"
    )
    koi_max_sngle_ev: float | None = Field(
        default=None, ge=0.0, le=1e7, description="strongest single event SNR"
    )
    koi_kepmag: float | None = Field(default=None, ge=1.0, le=30.0, description="Kepler magnitude")
    koi_count: float | None = Field(
        default=None, ge=0.0, le=1e4, description="candidates on this star"
    )


class AsteroidRequest(BaseModel):
    """One near-Earth object, summarised from its close-approach events.

    The caller supplies object-level quantities, not event rows, because that is what the
    model was trained on: hazard is a property of the asteroid, not of a single date.
    """

    model_config = ConfigDict(extra="forbid")

    designation: str | None = Field(default=None, description="JPL small-body designation")
    moid_proxy_au: float | None = Field(
        default=None,
        ge=0.0,
        le=5.0,
        description="smallest close-approach distance in the window, au (MOID proxy)",
    )
    n_approaches: float | None = Field(
        default=None, ge=0.0, le=1e5, description="approaches recorded"
    )
    n_approaches_within_lunar: float | None = Field(
        default=None, ge=0.0, le=1e5, description="approaches inside 1 lunar distance"
    )
    v_rel_at_closest: float | None = Field(
        default=None, ge=0.0, le=200.0, description="relative velocity, km/s"
    )
    v_inf_at_closest: float | None = Field(
        default=None, ge=0.0, le=200.0, description="velocity at infinity, km/s"
    )
    uncertainty_ratio_at_closest: float | None = Field(
        default=None, ge=0.0, le=1.0, description="dist_min / dist_max at the closest approach"
    )
    absolute_magnitude_h: float | None = Field(
        default=None,
        ge=1.0,
        le=40.0,
        description="absolute magnitude H (brighter = smaller number)",
    )
    diameter_reported_km: float | None = Field(
        default=None, ge=0.0, le=1000.0, description="JPL-reported diameter, km"
    )
    implied_albedo: float | None = Field(
        default=None, ge=0.0, le=2.0, description="albedo implied by H and diameter"
    )
    n_orbit_solutions: float | None = Field(
        default=None, ge=0.0, le=1e4, description="distinct orbit_id values"
    )
    t_sigma_seconds_at_closest: float | None = Field(
        default=None,
        ge=0.0,
        le=1e9,
        description="3-sigma timing uncertainty at closest approach, seconds",
    )
    observation_span_years: float | None = Field(
        default=None, ge=0.0, le=1e4, description="JD span of the recorded approaches"
    )
    closest_approach_month: int | None = Field(
        default=None, ge=1, le=12, description="calendar month of the closest approach"
    )


class ClassProbability(BaseModel):
    label: str
    probability: float


class PredictResponse(BaseModel):
    task: str
    model: str
    prediction: str
    probabilities: list[ClassProbability]
    decision_threshold_note: str
    features_used: dict[str, float | None]
    derived_features: dict[str, float | None]
    model_trained_utc: str
    dataset_fingerprint: str
    scikit_learn: str
    classes_order: list[str]
    feature_ranges_available: bool
    outside_training_range: list[str] = Field(default_factory=list)
    request_id: str | None = None
    explanation: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    models: dict[str, Any]
    version: str


class SchemaField(BaseModel):
    name: str
    type: str
    required: bool
    units: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    description: str | None = None


class TaskSchema(BaseModel):
    task: str
    endpoint: str
    fields: list[SchemaField]
    labels: list[str]
    feature_columns: list[str]
    notes: list[str]
