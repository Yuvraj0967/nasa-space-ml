"""Shared HTTP plumbing: one client factory and one retrying fetch pair.

Responsibilities kept here so the dataset modules stay about data, not networking:
  * a real timeout on every request (NASA endpoints are public and sometimes slow),
  * a User-Agent that identifies the project, because we are a polite guest,
  * bounded retries with exponential backoff and jitter on 429/5xx,
  * translation of transport failures into UpstreamNASAError so callers never see
    a raw httpx exception.

The client is injected everywhere, which is what lets the test suite drive every branch
with httpx.MockTransport and never touch the network.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable

import httpx

from ..config import HttpSettings
from ..errors import UpstreamNASAError

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


def make_client(settings: HttpSettings) -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(settings.timeout_seconds),
        headers={"User-Agent": settings.user_agent, "Accept": "*/*"},
        follow_redirects=True,
    )


def _sleep(seconds: float, sleeper: Callable[[float], None]) -> None:
    sleeper(seconds)


def _attempt_fetch(
    client: httpx.Client,
    url: str,
    params: dict | None,
    settings: HttpSettings,
    sleeper: Callable[[float], None],
    on_response: Callable[[httpx.Response], str],
) -> str:
    last_error: str = "no attempt made"
    for attempt in range(1, settings.max_attempts + 1):
        try:
            response = client.get(url, params=params)
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            if response.status_code in RETRY_STATUS:
                last_error = f"HTTP {response.status_code}"
            elif response.is_error:
                # 4xx other than 429 will not fix itself; do not waste retries.
                raise UpstreamNASAError(
                    f"{url} rejected our request with HTTP {response.status_code}: "
                    f"{response.text[:240]}",
                    details={"status": response.status_code, "attempt": attempt},
                )
            else:
                return on_response(response)

        if attempt < settings.max_attempts:
            backoff = min(
                settings.backoff_cap_seconds,
                settings.backoff_base_seconds * (2 ** (attempt - 1)),
            )
            _sleep(backoff * (1.0 + random.random() * 0.25), sleeper)

    raise UpstreamNASAError(
        f"{url} failed after {settings.max_attempts} attempts ({last_error})",
        details={"attempts": settings.max_attempts},
    )


def fetch_text(
    client: httpx.Client,
    url: str,
    params: dict | None = None,
    *,
    settings: HttpSettings,
    sleeper: Callable[[float], None] = time.sleep,
) -> str:
    """GET a URL and return its body as text, retrying transient failures."""
    return _attempt_fetch(client, url, params, settings, sleeper, on_response=lambda r: r.text)


def fetch_json(
    client: httpx.Client,
    url: str,
    params: dict | None = None,
    *,
    settings: HttpSettings,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict:
    """GET a URL and return its JSON body, retrying transient failures.

    A JSON parse failure is *not* retried. When a NASA host answers 200 with an HTML
    maintenance page, the status code is a lie and no amount of waiting fixes it, so we
    fail once, loudly, with enough of the body in the message to identify the page. This
    is the case where the transport succeeded and the contract did not; letting
    `json.JSONDecodeError` escape would put a Python traceback in front of a caller that
    was told it only has to handle `UpstreamNASAError`.
    """

    captured: dict[str, str] = {}

    def parse(response: httpx.Response) -> str:
        captured["content_type"] = response.headers.get("content-type", "")
        return response.text

    body = _attempt_fetch(client, url, params, settings, sleeper, on_response=parse)
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        snippet = body[:240].replace("\n", " ").strip()
        raise UpstreamNASAError(
            f"{url} answered 200 but the body is not JSON ({exc.msg} at line {exc.lineno} "
            f"column {exc.colno}); it starts {snippet!r}. Usually this is an HTML outage "
            "page or a proxy error document, not a change to the API.",
            details={"content_type": captured.get("content_type", ""), "body_start": snippet[:120]},
        ) from exc
