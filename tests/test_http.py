"""HTTP plumbing under `httpx.MockTransport` - zero network, every branch.

These tests exist because a retry layer is almost impossible to reason about by reading
it: does a 404 burn four attempts? does a 503 stop at the cap? The only honest answers
come from counting the calls. `http.Client` is injected into every dataset function for
exactly this reason.
"""

from __future__ import annotations

import httpx
import pytest

from nasa_space_ml.config import HttpSettings
from nasa_space_ml.data.http import RETRY_STATUS, fetch_json, fetch_text, make_client
from nasa_space_ml.errors import UpstreamNASAError

URL = "https://example.invalid/thing"


@pytest.fixture()
def http_settings() -> HttpSettings:
    return HttpSettings(
        timeout_seconds=7.0,
        max_attempts=4,
        backoff_base_seconds=0.5,
        backoff_cap_seconds=12.0,
        user_agent="nasa-space-ml-test",
    )


@pytest.fixture()
def sleeps() -> list[float]:
    return []


def client_for(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_success_returns_the_body(http_settings: HttpSettings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="kepler said hello")

    assert fetch_text(client_for(handler), URL, settings=http_settings) == "kepler said hello"


def test_query_parameters_are_forwarded(http_settings: HttpSettings) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, text="ok")

    fetch_text(
        client_for(handler),
        URL,
        params={"query": "select 1", "format": "csv"},
        settings=http_settings,
    )
    assert "query=select+1" in seen[0]
    assert "format=csv" in seen[0]


def test_json_body_is_parsed(http_settings: HttpSettings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"fields": ["des"], "data": [["x"]]})

    payload = fetch_json(client_for(handler), URL, settings=http_settings)
    assert payload["data"] == [["x"]]


def test_malformed_json_raises_upstream_not_json_decode(http_settings: HttpSettings) -> None:
    """A NASA outage page served with a 200 status is the nastiest failure mode: the
    transport says success and the body says HTML. Callers must still only ever see
    `UpstreamNASAError`, and the message has to contain enough of the body to tell an
    outage apart from a schema change."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, text="<html>maintenance</html>", headers={"content-type": "text/html"}
        )

    with pytest.raises(UpstreamNASAError, match="not JSON") as excinfo:
        fetch_json(client_for(handler), URL, settings=http_settings)
    assert excinfo.value.details["content_type"] == "text/html"
    assert "<html>" in excinfo.value.details["body_start"]


@pytest.mark.parametrize("status", sorted(RETRY_STATUS))
def test_transient_statuses_are_retried_then_succeed(
    status: int, http_settings: HttpSettings, sleeps: list[float]
) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(status if len(calls) < 3 else 200, text="eventually")

    body = fetch_text(client_for(handler), URL, settings=http_settings, sleeper=sleeps.append)
    assert body == "eventually"
    assert len(calls) == 3
    assert len(sleeps) == 2


def test_exhausting_attempts_names_the_count_and_last_error(
    http_settings: HttpSettings, sleeps: list[float]
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="unavailable")

    with pytest.raises(UpstreamNASAError, match="failed after 4 attempts") as excinfo:
        fetch_text(client_for(handler), URL, settings=http_settings, sleeper=sleeps.append)
    assert excinfo.value.details["attempts"] == 4
    assert len(sleeps) == 3  # no sleep after the final attempt


def test_permanent_client_errors_are_not_retried(
    http_settings: HttpSettings, sleeps: list[float]
) -> None:
    """A 400 will not fix itself. Retrying four times wastes a minute and hammers a
    endpoint that already told us no."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(400, text="ORA-00904: 'HOSTNAME': invalid identifier")

    with pytest.raises(UpstreamNASAError, match="rejected our request with HTTP 400") as excinfo:
        fetch_text(client_for(handler), URL, settings=http_settings, sleeper=sleeps.append)
    assert len(calls) == 1
    assert sleeps == []
    assert excinfo.value.details["status"] == 400
    assert "ORA-00904" in str(excinfo.value)  # the server's message is the useful part


def test_transport_failures_are_translated_not_raised_raw(http_settings: HttpSettings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("connection to exoplanetarchive timed out")

    with pytest.raises(UpstreamNASAError, match="ConnectTimeout") as excinfo:
        fetch_text(client_for(handler), URL, settings=http_settings, sleeper=lambda _s: None)
    assert excinfo.value.details["attempts"] == http_settings.max_attempts


def test_backoff_grows_and_is_capped(
    http_settings: HttpSettings, sleeps: list[float], monkeypatch
) -> None:
    monkeypatch.setattr("nasa_space_ml.data.http.random.random", lambda: 0.0)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="slow down")

    with pytest.raises(UpstreamNASAError):
        fetch_text(client_for(handler), URL, settings=http_settings, sleeper=sleeps.append)
    assert sleeps == pytest.approx([0.5, 1.0, 2.0])


def test_backoff_cap_applies_to_a_long_retry_budget(
    http_settings: HttpSettings, sleeps: list[float], monkeypatch
) -> None:
    monkeypatch.setattr("nasa_space_ml.data.http.random.random", lambda: 0.0)
    settings = HttpSettings(7.0, 6, 3.0, 4.0, "ua")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with pytest.raises(UpstreamNASAError):
        fetch_text(client_for(handler), URL, settings=settings, sleeper=sleeps.append)
    assert max(sleeps) <= settings.backoff_cap_seconds


def test_jitter_is_present_but_bounded(http_settings: HttpSettings, sleeps: list[float]) -> None:
    """Full jitter would be 0..backoff; we add 0..25% on top so simultaneous retries from
    parallel fetches do not synchronise onto the same retry slot."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="boom")

    with pytest.raises(UpstreamNASAError):
        fetch_text(client_for(handler), URL, settings=http_settings, sleeper=sleeps.append)
    assert all(0.5 <= value <= 12.0 * 1.25 for value in sleeps)


def test_real_client_carries_a_user_agent_and_a_timeout(http_settings: HttpSettings) -> None:
    """Two things we owe the NASA ops team: identify ourselves, and never hang forever."""
    with make_client(http_settings) as client:
        assert client.headers["User-Agent"] == "nasa-space-ml-test"
        assert client.timeout.connect == 7.0
        assert client.follow_redirects is True


def test_redirects_are_followed(http_settings: HttpSettings) -> None:
    """NASA moves endpoints; `follow_redirects=True` is what makes that a 200 not a 301."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old":
            return httpx.Response(301, headers={"location": "https://example.invalid/thing"})
        return httpx.Response(200, text="moved fine")

    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
        assert (
            fetch_text(client, "https://example.invalid/old", settings=http_settings)
            == "moved fine"
        )
