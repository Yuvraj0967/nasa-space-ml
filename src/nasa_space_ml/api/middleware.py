"""Request-id middleware.

Every response carries `X-Request-Id`, echoed from the request if the caller supplied one.
For a service whose whole purpose is "explain why this object scored the way it did", being
able to say "send me the request id from 03:41 UTC" is the difference between a reproducible
incident and a shrug.

The id also lands in the problem+json body and in the log line, so a single grep across both
finds the whole story. Logging is intentionally a flat line rather than structured JSON:
this is a capstone, and `tail -f` should be readable without a parser.
"""

from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

HEADER = "X-Request-Id"


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        incoming = request.headers.get(HEADER)
        request_id = incoming if incoming and len(incoming) <= 64 else _new_id()
        request.state.request_id = request_id
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        response.headers[HEADER] = request_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
        response.headers["Cache-Control"] = "no-store"
        print(
            f"{request.method} {request.url.path} -> {response.status_code} "
            f"({elapsed_ms:.1f} ms) id={request_id}",
            flush=True,
        )
        return response
