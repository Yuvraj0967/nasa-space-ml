"""RFC 9457 problem+json responses.

One place that turns our exception hierarchy into an HTTP answer, so no route has to remember
to do it and no caller ever receives a bare stack trace. The `type` URI is a documentation
anchor rather than a resolvable page, which is what RFC 9457 expects when you do not have a
public error catalogue.

`code` is the machine-readable half of the contract. The front end keys its messaging off
that, never off `detail`, so a reworded error message cannot break a UI.
"""

from __future__ import annotations

from typing import Any

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ..errors import (
    ConfigError,
    DataError,
    InvalidFeaturesError,
    ModelNotTrainedError,
    NasaSpaceMlError,
    UpstreamNASAError,
)

STATUS_BY_CODE: dict[str, int] = {
    "NasaSpaceMlError": status.HTTP_500_INTERNAL_SERVER_ERROR,
    "ConfigError": status.HTTP_500_INTERNAL_SERVER_ERROR,
    "DataError": status.HTTP_502_BAD_GATEWAY,
    "UpstreamNASAError": status.HTTP_502_BAD_GATEWAY,
    "ModelNotTrainedError": status.HTTP_503_SERVICE_UNAVAILABLE,
    "InvalidFeaturesError": status.HTTP_422_UNPROCESSABLE_CONTENT,
}

# The `type` field of an RFC 9457 document must be a stable identifier; it does not have to
# resolve. It resolves anyway, deliberately: the `/blob/main/` form is the URL a reviewer in
# front of a 503 can actually click, and an error contract that sends people to a 404 is a
# contract nobody can read under pressure.
TYPE_URIS: dict[str, str] = {
    "ConfigError": "https://github.com/Yuvraj0967/nasa-space-ml/blob/main/docs/errors.md#config",
    "DataError": "https://github.com/Yuvraj0967/nasa-space-ml/blob/main/docs/errors.md#data",
    "UpstreamNASAError": "https://github.com/Yuvraj0967/nasa-space-ml/blob/main/docs/errors.md#upstream",
    "ModelNotTrainedError": "https://github.com/Yuvraj0967/nasa-space-ml/blob/main/docs/errors.md#not-trained",
    "InvalidFeaturesError": "https://github.com/Yuvraj0967/nasa-space-ml/blob/main/docs/errors.md#invalid-features",
}


def problem(
    request: Request,
    *,
    code: str,
    title: str,
    detail: str,
    status_code: int,
    instance: str | None = None,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    body: dict[str, Any] = {
        "type": TYPE_URIS.get(
            code, "https://github.com/Yuvraj0967/nasa-space-ml/blob/main/docs/errors.md"
        ),
        "title": title,
        "status": status_code,
        "detail": detail,
        "instance": instance or str(request.url.path),
        "code": code,
        "request_id": getattr(request.state, "request_id", None),
    }
    if details:
        body["errors"] = details
    return JSONResponse(
        status_code=status_code,
        content=body,
        media_type="application/problem+json",
        headers={"X-Request-Id": body["request_id"] or "unknown"},
    )


def from_internal_error(request: Request, exc: NasaSpaceMlError) -> JSONResponse:
    code = type(exc).__name__
    return problem(
        request,
        code=code,
        title=code.replace("Error", "").replace("NasaSpaceMl", "Internal"),
        detail=str(exc),
        status_code=STATUS_BY_CODE.get(code, exc.http_status),
        details=exc.details,
    )


def from_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Pydantic's field errors, re-expressed in problem+json rather than FastAPI's default."""
    fields: list[dict[str, Any]] = []
    for item in exc.errors():
        location = [str(part) for part in item.get("loc", [])]
        fields.append(
            {
                "field": ".".join(location[1:]) or ".".join(location) or "<body>",
                "in": location[0] if location else "body",
                "message": str(item.get("msg", "invalid value")),
                "type": str(item.get("type", "value_error")),
                "input": _safe_input(item.get("input")),
            }
        )
    return problem(
        request,
        code="RequestValidationError",
        title="Invalid request body",
        detail=(
            "the submitted values do not satisfy the field contract; the `errors` array names "
            "each offending field"
        ),
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        details={"fields": fields},
    )


def _safe_input(value: Any) -> Any:
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [str(entry)[:64] for entry in value[:8]]
    return str(value)[:120]


def register_handlers(app: Any) -> None:
    """Attach the exception handlers to an application instance."""
    app.add_exception_handler(NasaSpaceMlError, from_internal_error)
    app.add_exception_handler(RequestValidationError, from_validation_error)
    app.add_exception_handler(InvalidFeaturesError, from_internal_error)
    app.add_exception_handler(ModelNotTrainedError, from_internal_error)
    app.add_exception_handler(UpstreamNASAError, from_internal_error)
    app.add_exception_handler(DataError, from_internal_error)
    app.add_exception_handler(ConfigError, from_internal_error)
