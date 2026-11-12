"""Near-Earth object hazard routes.

The `/sample` endpoint here is doing double duty. It shows a real object with its real
JPL-supplied PHA status, and because the model's job is to reproduce a published rule from
the *other* measurable properties, the demo can display the rule's own verdict alongside the
model's. When they disagree, that is the interesting case, and the response says so instead
of hiding it.
"""

from __future__ import annotations

import math
from typing import Any

from fastapi import APIRouter, Query, Request

from ...errors import DataError
from ...features import asteroids as ast_features
from ..deps import ASTEROID_MODEL, predict_proba
from ..schemas import AsteroidRequest, PredictResponse, TaskSchema
from .meta import fields_from_model

router = APIRouter(prefix="/api/v1/asteroids", tags=["asteroids"])

#: Which aggregated object columns a caller is actually allowed to send. Everything the
#: aggregation produces except the two quantities `derive()` recomputes at predict time from
#: `absolute_magnitude_h` and the settings albedo. Intersecting with the request model rather
#: than hand-listing means `/sample` cannot offer a body `/predict` will reject - which is
#: exactly what it did before this line existed, as an `extra_forbidden` 422.
SAMPLE_COLUMNS: tuple[str, ...] = tuple(
    column for column in ast_features.BASE_INPUT_COLUMNS if column in AsteroidRequest.model_fields
)

NOTES = [
    "the label is JPL's published PHA membership, not a re-derivation from distance and "
    "magnitude, so the model has to find the pattern rather than copy the rule",
    "features are object-level summaries of every approach in the configured window; supply "
    "the ones you know and leave the rest blank",
    "this is not a conjunction-screening tool and must not be used for operational warnings",
]


@router.get("/schema", response_model=TaskSchema)
def schema(request: Request) -> TaskSchema:
    config = request.app.state.nsm.config
    return TaskSchema(
        task="asteroids",
        endpoint="/api/v1/asteroids/predict",
        fields=fields_from_model(AsteroidRequest),
        labels=list(config.asteroids.labels),
        feature_columns=list(ast_features.FEATURE_COLUMNS),
        notes=NOTES,
    )


@router.get("/sample")
def sample(
    request: Request,
    designation: str | None = Query(default=None),
    index: int | None = Query(default=None, ge=0),
    pha: bool | None = Query(default=None, description="restrict to hazardous / non-hazardous"),
) -> dict[str, Any]:
    """A real object from the cached approach data, as a valid request body."""
    state = request.app.state.nsm
    config = state.config
    events, objects = state.asteroid_tables()
    if pha is not None:
        objects = objects[objects[ast_features.LABEL_COLUMN] == bool(pha)]
    if objects.empty:
        raise DataError("the cached snapshot has no objects matching that filter")

    if designation:
        chosen = objects[objects["des"].astype(str) == designation]
        if chosen.empty:
            raise DataError(
                f"designation {designation!r} has no close approaches in the cached window",
                details={"objects": len(objects)},
            )
        row = chosen.iloc[0]
    else:
        # Same contract as the exoplanet route: no `index` means the first object the filter
        # left, not a crash. `int(index)` here raised TypeError on the front end's default
        # "load a real object" click, which is how this line ended up written out.
        position = (0 if index is None else index) % len(objects)
        row = objects.iloc[position]

    payload: dict[str, Any] = {column: _number(row[column]) for column in SAMPLE_COLUMNS}
    month = _number(row.get("closest_approach_month"))
    payload["designation"] = str(row["des"])
    payload["closest_approach_month"] = int(month) if month is not None else None

    rule = (
        payload.get("moid_proxy_au") is not None
        and payload.get("absolute_magnitude_h") is not None
        and payload["moid_proxy_au"] <= config.asteroids.pha_distance_au
        and payload["absolute_magnitude_h"] <= config.asteroids.pha_absolute_magnitude
    )
    return {
        "request": payload,
        "truth": {
            "pha_flag": bool(row[ast_features.LABEL_COLUMN]),
            "n_events": int((events["des"].astype(str) == str(row["des"])).sum()),
        },
        "nasa_rule_verdict": "PHA" if rule else "NOT_PHA",
        "note": (
            "`truth` and `nasa_rule_verdict` are shown for the demo only; the model never "
            "sees them. The rule verdict is computed from the same two published thresholds "
            "the baseline model implements."
        ),
    }


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


@router.post("/predict", response_model=PredictResponse)
def predict(body: AsteroidRequest, request: Request) -> PredictResponse:
    state = request.app.state.nsm
    config = state.config
    payload = body.model_dump(exclude_none=True)
    scored = predict_proba("asteroids", payload, state)
    scored["request_id"] = getattr(request.state, "request_id", None)

    moid = payload.get("moid_proxy_au")
    magnitude = payload.get("absolute_magnitude_h")
    if moid is None or magnitude is None:
        verdict = "unknown: the published rule needs both MOID and H"
    else:
        verdict = (
            "PHA"
            if moid <= config.asteroids.pha_distance_au
            and magnitude <= config.asteroids.pha_absolute_magnitude
            else "NOT_PHA"
        )
    scored["decision_threshold_note"] = (
        f"argmax at 0.5; NASA's published rule on the same inputs says {verdict} "
        f"(MOID <= {config.asteroids.pha_distance_au} au AND H <= "
        f"{config.asteroids.pha_absolute_magnitude}). Where the two disagree, the rule is "
        "the reference - that disagreement is what the model was trained to reduce."
    )
    scored["model"] = ASTEROID_MODEL
    return PredictResponse(**scored)
