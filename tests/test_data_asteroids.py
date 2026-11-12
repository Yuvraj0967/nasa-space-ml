"""JPL CAD client: positional JSON, versioned signature, and a pagination loop that has
already been broken once in the wild.

The CAD response is an array-of-arrays plus a `fields` header, so a silently reordered
field would mis-assign every column without raising anything. That is why the field order
is pinned across pages and the API version is checked before a single row is parsed - and
why both behaviours have tests here.
"""

from __future__ import annotations

import math
from dataclasses import replace

import httpx
import pandas as pd
import pytest

from nasa_space_ml.data import asteroids
from nasa_space_ml.data.cache import Cache
from nasa_space_ml.errors import DataError

FIELDS = list(asteroids.EVENT_FIELDS)


def cad_row(
    des: str,
    *,
    moid: float = 0.01,
    magnitude: float = 21.0,
    orbit_id: str = "1",
    jd: float = 2459000.5,
    cd: str = "2020-Jan-05 02:06",
    diameter: object = 0.31,
) -> list[object]:
    """One approach, in the exact positional order the `fields` array declares."""
    return [
        des,
        orbit_id,
        jd,
        cd,
        moid * 1.3,
        moid,
        moid * 1.6,
        11.0,
        5.0,
        "2_04:30",
        magnitude,
        diameter,
        0.03,
    ]


def payload(
    rows: list[list[object]],
    *,
    fields: list[str] | None = None,
    total: int | None = None,
    version: str = "1.5",
    code: str = "200",
) -> dict:
    body: dict[str, object] = {
        "signature": {"source": "SBDB Close Approach Data", "version": version},
        "fields": FIELDS if fields is None else fields,
        "data": rows,
        "total": len(rows) if total is None else total,
        "count": len(rows),
    }
    if code != "200":
        body["code"] = code
        body["message"] = "query error"
    else:
        body["code"] = "200"
    return body


def client_for(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def paging_config(tmp_config, **overrides):
    """Small pages so pagination is testable without 5000 synthetic rows."""
    settings = replace(tmp_config.asteroids, **overrides)
    return replace(tmp_config, asteroids=settings)


# ------------------------------------------------------------------ t_sigma formatting


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2_04:30", 2 * 86400 + 4 * 3600 + 30 * 60),
        ("13:02", 13 * 3600 + 2 * 60),
        ("< 00:01", 60.0),
        ("0_00:00", 0.0),
    ],
)
def test_t_sigma_formats_seen_in_the_live_feed(raw: str, expected: float) -> None:
    assert asteroids.parse_t_sigma(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "   ", "None", "not a duration", "4:5:6:7"])
def test_unparseable_t_sigma_is_none_never_zero(raw: object) -> None:
    """Unknown timing precision is not perfect timing precision; mapping it to 0.0 would
    tell the model the orbit is exactly known when it is not known at all."""
    assert asteroids.parse_t_sigma(raw) is None  # type: ignore[arg-type]


# ------------------------------------------------------------------------- request params


def test_params_carry_the_configured_window(tmp_config) -> None:
    params = asteroids.build_params(tmp_config, limit_from=1)
    assert params["date-min"] == tmp_config.asteroids.date_min
    assert params["date-max"] == tmp_config.asteroids.date_max
    assert params["dist-max"] == str(tmp_config.asteroids.dist_max_au)


def test_params_request_diameters_and_sort_by_date(tmp_config) -> None:
    params = asteroids.build_params(tmp_config, limit_from=1)
    assert params["diameter"] == "1"  # without it the diameter column is simply absent
    assert params["sort"] == "date"
    assert params["limit"] == str(tmp_config.asteroids.page_limit)


def test_pha_filter_is_only_present_when_asked(tmp_config) -> None:
    assert "pha" not in asteroids.build_params(tmp_config, limit_from=1)
    assert asteroids.build_params(tmp_config, limit_from=1, pha_only=True)["pha"] == "1"


def test_limit_from_advances(tmp_config) -> None:
    assert asteroids.build_params(tmp_config, limit_from=5001)["limit-from"] == "5001"


# --------------------------------------------------------------------------- validation


def test_error_code_is_reported_with_the_servers_message(tmp_config, cache: Cache) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload([], code="500"))

    with pytest.raises(DataError, match="error code 500"):
        asteroids._page(client_for(handler), tmp_config, pha_only=False, sleeper=lambda _s: None)


def test_api_version_change_is_fatal_before_parsing(tmp_config, cache: Cache) -> None:
    """The payload is positional. A new version means the field list may have moved, and
    an unnoticed move silently turns `v_inf` into `h`. Stop, do not adapt."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload([cad_row("2020 AB1")], version="1.6"))

    with pytest.raises(DataError, match="version changed") as excinfo:
        asteroids._page(client_for(handler), tmp_config, pha_only=False, sleeper=lambda _s: None)
    assert excinfo.value.details["found"] == "1.6"


def test_a_single_row_is_parsed_into_named_columns(tmp_config) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload([cad_row("2020 AB1", moid=0.02)]))

    frame = asteroids._page(
        client_for(handler), tmp_config, pha_only=False, sleeper=lambda _s: None
    )
    assert list(frame.columns) == FIELDS
    assert float(frame.loc[0, "dist_min"]) == pytest.approx(0.02)
    assert frame.loc[0, "des"] == "2020 AB1"


def test_reordered_fields_between_pages_is_fatal(tmp_config) -> None:
    config = paging_config(tmp_config, page_limit=1, max_pages=5)
    swapped = [FIELDS[1], FIELDS[0], *FIELDS[2:]]
    responses = [payload([cad_row("A")], total=2), payload([cad_row("B")], fields=swapped, total=2)]
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = responses[min(len(calls), len(responses) - 1)]
        calls.append(1)
        return httpx.Response(200, json=body)

    with pytest.raises(DataError, match="field order changed"):
        asteroids._page(client_for(handler), config, pha_only=False, sleeper=lambda _s: None)


def test_rows_without_a_fields_array_are_fatal(tmp_config) -> None:
    """Without names the arrays are meaningless; guessing a mapping is how you get a model
    trained on velocities labelled as magnitudes."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = payload([cad_row("A")])
        del body["fields"]
        return httpx.Response(200, json=body)

    with pytest.raises(DataError, match="without a 'fields' array"):
        asteroids._page(client_for(handler), tmp_config, pha_only=False, sleeper=lambda _s: None)


def test_zero_rows_for_the_window_is_fatal(tmp_config) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload([]))

    with pytest.raises(DataError, match="no rows"):
        asteroids._page(client_for(handler), tmp_config, pha_only=False, sleeper=lambda _s: None)


# --------------------------------------------------------------------------- pagination


def test_short_page_ends_the_walk(tmp_config) -> None:
    """The reliable end-of-data signal: fewer rows than asked for means there are no more."""
    config = paging_config(tmp_config, page_limit=3, max_pages=10)
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, json=payload([cad_row(f"O{j}") for j in range(2)], total=99))

    frame = asteroids._page(client_for(handler), config, pha_only=False, sleeper=lambda _s: None)
    assert len(frame) == 2
    assert len(calls) == 1


def test_empty_page_after_full_pages_ends_the_walk(tmp_config) -> None:
    """`total` drifts by a few rows between live requests as JPL re-runs solutions, so
    paging one past the end is normal. It must not be an error and must not loop."""
    config = paging_config(tmp_config, page_limit=2, max_pages=10)
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(200, json=payload([cad_row("A"), cad_row("B")], total=3))
        return httpx.Response(200, json=payload([], total=3))

    frame = asteroids._page(client_for(handler), config, pha_only=False, sleeper=lambda _s: None)
    assert len(frame) == 2
    assert len(calls) == 2


def test_total_drift_upwards_does_not_extend_the_walk(tmp_config) -> None:
    config = paging_config(tmp_config, page_limit=2, max_pages=10)
    totals = [4, 100]
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        index = min(len(calls), len(totals) - 1)
        calls.append(1)
        rows = [cad_row(f"P{index}R{j}") for j in range(2)]
        return httpx.Response(200, json=payload(rows, total=totals[index]))

    frame = asteroids._page(client_for(handler), config, pha_only=False, sleeper=lambda _s: None)
    # first page says total=4, so limit_from becomes 3 <= 4 and we take a second page;
    # after that limit_from=5 > 4 (the frozen first total) and the walk stops.
    assert len(frame) == 4
    assert len(calls) == 2


def test_unbounded_pagination_is_stopped_by_max_pages(tmp_config) -> None:
    config = paging_config(tmp_config, page_limit=2, max_pages=3)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload([cad_row("A"), cad_row("B")], total=10_000))

    with pytest.raises(DataError, match="exceeded max_pages=3"):
        asteroids._page(client_for(handler), config, pha_only=False, sleeper=lambda _s: None)


def test_limit_from_advances_by_rows_received_not_by_page_size(tmp_config) -> None:
    config = paging_config(tmp_config, page_limit=2, max_pages=4)
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.params["limit-from"])
        rows = [cad_row("A"), cad_row("B")] if len(seen) == 1 else [cad_row("C")]
        return httpx.Response(200, json=payload(rows, total=9))

    frame = asteroids._page(client_for(handler), config, pha_only=False, sleeper=lambda _s: None)
    assert seen[:2] == ["1", "3"]
    assert len(frame) == 3


# --------------------------------------------------------------------- download and load


def test_download_writes_events_and_pha_names(tmp_config, cache: Cache) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("pha") == "1":
            return httpx.Response(200, json=payload([cad_row("2020 PHA2"), cad_row("2020 PHA1")]))
        return httpx.Response(
            200,
            json=payload([cad_row("2020 PHA1"), cad_row("2020 AB9"), cad_row("2020 PHA2")]),
        )

    written = asteroids.download(tmp_config, cache, client=client_for(handler))
    settings = tmp_config.asteroids
    assert set(written) == {settings.raw_file, settings.pha_file}
    names = cache.read_text(settings.pha_file).splitlines()
    assert names == ["2020 PHA1", "2020 PHA2"]  # sorted and de-duplicated


def test_cached_events_round_trip_through_csv(tmp_config, cache: Cache) -> None:
    """The cache stores CSV, so the positional JSON must survive a write/read cycle with
    its column names and numeric types intact - that cycle is what training reads."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("pha") == "1":
            return httpx.Response(200, json=payload([cad_row("A")], total=1))
        return httpx.Response(200, json=payload([cad_row("A", moid=0.045)], total=1))

    asteroids.download(tmp_config, cache, client=client_for(handler))
    assert cache.read_text(tmp_config.asteroids.raw_file).startswith("des,orbit_id,jd")
    frame = asteroids.load_events(tmp_config, cache)
    assert float(frame.loc[0, "dist_min"]) == pytest.approx(0.045)


def test_cached_pha_list_round_trips(tmp_config, cache: Cache) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("pha") == "1":
            return httpx.Response(200, json=payload([cad_row("2020 X1")], total=1))
        return httpx.Response(200, json=payload([cad_row("2020 X1")], total=1))

    asteroids.download(tmp_config, cache, client=client_for(handler))
    assert cache.read_text(tmp_config.asteroids.pha_file) == "2020 X1"


def test_download_refuses_to_cache_an_empty_event_table(tmp_config, cache: Cache) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload([], total=0))

    with pytest.raises(DataError):
        asteroids.download(tmp_config, cache, client=client_for(handler))
    assert not cache.exists(tmp_config.asteroids.raw_file)


def test_second_download_makes_no_requests(tmp_config, cache: Cache) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, json=payload([cad_row("A")]))

    for _ in range(2):
        asteroids.download(tmp_config, cache, client=client_for(handler))
    assert len(calls) == 2  # one for events, one for the PHA list - and never again


def test_load_events_labels_by_designation_membership(tmp_config, cache: Cache) -> None:
    settings = tmp_config.asteroids
    events = pd.DataFrame(
        [cad_row("2020 HAZ1"), cad_row("2020 SAFE1"), cad_row("2020 HAZ2")], columns=FIELDS
    )
    cache.write_text(settings.raw_file, events.to_csv(index=False))
    cache.write_text(settings.pha_file, "2020 HAZ1\n2020 HAZ2\n")

    frame = asteroids.load_events(tmp_config, cache)
    assert list(frame["pha_flag"]) == [True, False, True]
    assert frame["pha_flag"].dtype == bool


def test_load_events_converts_timing_uncertainty_per_row(tmp_config, cache: Cache) -> None:
    settings = tmp_config.asteroids
    first, second = cad_row("A"), cad_row("B")
    first[FIELDS.index("t_sigma_f")] = "1_00:00"
    second[FIELDS.index("t_sigma_f")] = "< 00:05"
    events = pd.DataFrame([first, second], columns=FIELDS)
    cache.write_text(settings.raw_file, events.to_csv(index=False))
    cache.write_text(settings.pha_file, "")

    frame = asteroids.load_events(tmp_config, cache)
    assert frame.loc[0, "t_sigma_seconds"] == 86400.0
    assert frame.loc[1, "t_sigma_seconds"] == 300.0


def test_load_events_with_no_pha_list_marks_everything_safe(tmp_config, cache: Cache) -> None:
    settings = tmp_config.asteroids
    events = pd.DataFrame([cad_row("A"), cad_row("B")], columns=FIELDS)
    cache.write_text(settings.raw_file, events.to_csv(index=False))
    cache.write_text(settings.pha_file, "")
    frame = asteroids.load_events(tmp_config, cache)
    assert not frame["pha_flag"].any()


def test_load_events_typed_numeric_columns(tmp_config, cache: Cache) -> None:
    settings = tmp_config.asteroids
    events = pd.DataFrame([cad_row("A", moid=0.031)], columns=FIELDS)
    cache.write_text(settings.raw_file, events.to_csv(index=False))
    cache.write_text(settings.pha_file, "")
    frame = asteroids.load_events(tmp_config, cache)
    assert float(frame.loc[0, "dist_min"]) == pytest.approx(0.031)
    assert math.isnan(float(frame.loc[0, "t_sigma_seconds"])) is False
    assert frame.loc[0, "t_sigma_seconds"] == asteroids.parse_t_sigma("2_04:30")
