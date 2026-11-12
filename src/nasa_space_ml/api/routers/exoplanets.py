"""Exoplanet disposition routes.

Three endpoints, and the middle one is the reason the service is usable at all:
`/sample` hands back a real row from the cached KOI table, so a reviewer can click Predict
and get a genuine NASA object scored rather than inventing plausible-looking numbers. It
also guarantees the form and the model are being exercised on the same distribution the
report describes.
"""

from __future__ import annotations

import math
from typing import Any

from fastapi import APIRouter, Query, Request

from ...data import exoplanets as exo_data
from ...errors import DataError
from ...features import exoplanets as exo_features
from ..deps import EXOPLANET_MODEL, predict_proba
from ..schemas import ExoplanetRequest, PredictResponse, TaskSchema
from .meta import fields_from_model

router = APIRouter(prefix="/api/v1/exoplanets", tags=["exoplanets"])

NOTES = [
    "all fields are optional: a blank is treated as 'not measured', exactly as a blank cell "
    "in the catalogue is, and the boosted model handles the gap directly",
    "koi_pdisposition is deliberately absent from the contract - it is the label wearing a "
    "different hat",
    "probabilities are the model's score, not a statement about whether the planet exists",
]


@router.get("/schema", response_model=TaskSchema)
def schema(request: Request) -> TaskSchema:
    config = request.app.state.nsm.config
    return TaskSchema(
        task="exoplanets",
        endpoint="/api/v1/exoplanets/predict",
        fields=fields_from_model(ExoplanetRequest),
        labels=list(config.exoplanets.labels),
        feature_columns=list(exo_features.FEATURE_COLUMNS),
        notes=NOTES,
    )


@router.get("/sample")
def sample(
    request: Request,
    kepoi_name: str | None = Query(default=None, description="specific candidate id"),
    index: int | None = Query(default=None, ge=0),
    label: str | None = Query(default=None, description="pick a row with this disposition"),
) -> dict[str, Any]:
    """A real candidate, straight out of the cached snapshot, as a valid request body."""
    state = request.app.state.nsm
    frame = exo_data.load(state.config, state.cache)
    if label:
        allowed = list(state.config.exoplanets.labels)
        if label not in allowed:
            raise DataError(f"unknown disposition {label!r}", details={"known": allowed})
        frame = frame[frame[state.config.exoplanets.label_column].astype(str) == label]
    if frame.empty:
        raise DataError("the cached snapshot has no rows matching that filter")

    if kepoi_name:
        chosen = frame[frame["kepoi_name"].astype(str) == kepoi_name]
        if chosen.empty:
            raise DataError(
                f"candidate {kepoi_name!r} is not in the cached snapshot",
                details={"rows": len(frame)},
            )
        row = chosen.iloc[0]
    else:
        # `index` is optional, so the no-parameter call - the one the front end makes when
        # the row box is left blank - has to mean something. It means "the first row of
        # whatever the filters left", which keeps a reviewer's screenshot reproducible. The
        # modulo lets any non-negative integer be passed without turning into a 500.
        position = (0 if index is None else index) % len(frame)
        row = frame.iloc[position]

    payload = {
        column: _number(row[column])
        for column in exo_features.RAW_INPUT_COLUMNS
        if column in frame.columns
    }
    payload["kepoi_name"] = str(row["kepoi_name"])
    return {
        "request": payload,
        "truth": {
            "koi_disposition": str(row[state.config.exoplanets.label_column]),
            "kepid": str(row.get("kepid", "")),
        },
        "note": (
            "`truth` is returned so the demo can show agreement with the catalogue. It is "
            "never part of the prediction request."
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
def predict(body: ExoplanetRequest, request: Request) -> PredictResponse:
    state = request.app.state.nsm
    payload = body.model_dump(exclude_none=True)
    scored = predict_proba("exoplanets", payload, state)
    scored["request_id"] = getattr(request.state, "request_id", None)
    scored["decision_threshold_note"] = (
        "argmax over the class probabilities; no threshold was applied. Macro-F1 was "
        "selected on grouped cross-validation, so the operating point is the default one."
    )
    scored["model"] = EXOPLANET_MODEL
    return PredictResponse(**scored)
