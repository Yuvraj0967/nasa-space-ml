"""`/api/v1/meta` and `/api/v1/health`.

The front end is built schema-first: it asks these endpoints what exists and renders that,
rather than hard-coding a copy of the contract. Two consequences we care about: adding a
feature to `configs/settings.yaml` shows up in the UI without a front-end change, and a
half-trained repository reports itself honestly instead of rendering a form for a model that
is not there.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ... import __version__
from ...data.exoplanets import LEAKAGE_FORBIDDEN
from ..deps import ASTEROID_MODEL, EXOPLANET_MODEL, expected_features
from ..schemas import HealthResponse, SchemaField

router = APIRouter(prefix="/api/v1", tags=["meta"])


def fields_from_model(model: type[BaseModel]) -> list[SchemaField]:
    """Describe a Pydantic request model as plain data the front end can render.

    The UI builds its forms from this rather than from a duplicated list of inputs, so a new
    field with a bound in `schemas.py` appears as a bounded numeric input with no
    front-end edit. That is the whole reason this function exists.
    """
    fields: list[SchemaField] = []
    for name, field in model.model_fields.items():
        minimum = maximum = None
        for constraint in field.metadata or []:
            for attribute in ("ge", "gt"):
                value = getattr(constraint, attribute, None)
                if value is not None:
                    minimum = float(value)
            for attribute in ("le", "lt"):
                value = getattr(constraint, attribute, None)
                if value is not None:
                    maximum = float(value)
        fields.append(
            SchemaField(
                name=name,
                type=_type_name(field.annotation),
                required=bool(field.is_required()),
                minimum=minimum,
                maximum=maximum,
                description=field.description,
                units=_UNITS.get(name),
            )
        )
    return fields


def _type_name(annotation: object) -> str:
    text = str(annotation)
    if "int" in text and "float" not in text:
        return "integer"
    if "float" in text:
        return "number"
    if "str" in text:
        return "string"
    return text


_UNITS = {
    "koi_period": "days",
    "koi_duration": "hours",
    "koi_depth": "ppm",
    "koi_prad": "Earth radii",
    "koi_sma": "au",
    "koi_teq": "K",
    "koi_steff": "K",
    "koi_srad": "solar radii",
    "koi_smass": "solar masses",
    "koi_slogg": "log10(cm/s^2)",
    "koi_incl": "degrees",
    "koi_kepmag": "Kepler mag",
    "moid_proxy_au": "au",
    "v_rel_at_closest": "km/s",
    "v_inf_at_closest": "km/s",
    "diameter_reported_km": "km",
    "t_sigma_seconds_at_closest": "seconds",
    "observation_span_years": "years",
    "absolute_magnitude_h": "mag",
}


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    state = request.app.state.nsm
    models = {name: state.describe_model(name) for name in (EXOPLANET_MODEL, ASTEROID_MODEL)}
    ready = all(item["loaded"] for item in models.values())
    return HealthResponse(
        status="ready" if ready else "degraded",
        models=models,
        version=__version__,
    )


@router.get("/meta")
def meta(request: Request) -> dict:
    """Everything a client needs to talk to this service correctly, in one call."""
    state = request.app.state.nsm
    config = state.config
    return {
        "service": "nasa-space-ml",
        "version": __version__,
        "started_utc": state.started_utc,
        "models": {name: state.describe_model(name) for name in (EXOPLANET_MODEL, ASTEROID_MODEL)},
        "sources": {
            "exoplanets": {
                "endpoint": config.exoplanets.source,
                "table": config.exoplanets.table,
                "label_column": config.exoplanets.label_column,
                "group_column": config.exoplanets.group_column,
                "labels": list(config.exoplanets.labels),
            },
            "asteroids": {
                "endpoint": config.asteroids.source,
                "api_version_expected": config.asteroids.expected_api_version,
                "date_window": [config.asteroids.date_min, config.asteroids.date_max],
                "dist_max_au": config.asteroids.dist_max_au,
                "group_column": config.asteroids.group_column,
                "labels": list(config.asteroids.labels),
                "pha_rule": {
                    "moid_au_max": config.asteroids.pha_distance_au,
                    "absolute_magnitude_max": config.asteroids.pha_absolute_magnitude,
                    "assumed_geometric_albedo": config.asteroids.default_geometric_albedo,
                },
            },
        },
        "feature_contract": {
            "exoplanets": expected_features("exoplanets", config),
            "asteroids": expected_features("asteroids", config),
        },
        "cache": {
            "directory": str(state.cache.directory),
            "artifacts": sorted(state.cache.manifest()),
            "dataset_fingerprint": state.cache.dataset_fingerprint(),
        },
        "leakage_guard": {
            "exoplanets_forbidden_columns": sorted(LEAKAGE_FORBIDDEN),
            "why": (
                "koi_pdisposition is the pipeline's own preliminary verdict, effectively a copy "
                "of the label. It is never requested, and the request builder raises if asked."
            ),
        },
        "split": {
            "test_size": config.split.test_size,
            "val_size": config.split.val_size,
            "n_folds": config.split.n_folds,
            "strategy": "grouped by host star (kepid) / designation (des)",
            "seed": config.seed,
        },
    }
