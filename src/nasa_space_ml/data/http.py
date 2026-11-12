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
    return _attempt_fetch(
        client, url, params, settings, sleeper, on_response=lambda r: r.text
    )


def fetch_json(
    client: httpx.Client,
    url: str,
    params: dict | None = None,
    *,
    settings: HttpSettings,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict:
    """GET a URL and return its JSON body, retrying transient failures."""
    import json

    def parse(response: httpx.Response) -> str:
        return response.text

    return json.loads(
        _attempt_fetch(client, url, params, settings, sleeper, on_response=parse)
    )
