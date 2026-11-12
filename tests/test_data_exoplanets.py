"""Exoplanet Archive TAP client, driven entirely by `httpx.MockTransport`.

The tests encode the three verified quirks of this endpoint (documented in the module
docstring) so that a future "cleanup" of the query or the parser trips over a recorded
fact rather than rediscovering it through a failing training run.
"""

from __future__ import annotations

import math

import httpx
import pandas as pd
import pytest

from nasa_space_ml.data import exoplanets
from nasa_space_ml.data.cache import Cache
from nasa_space_ml.errors import DataError, UpstreamNASAError

HEADER = ",".join(exoplanets.BASE_COLUMNS)


def _csv(rows: list[str]) -> str:
    return "\n".join([HEADER, *rows]) + "\n"


def _row(index: int, disposition: str = "CONFIRMED", **overrides: str) -> str:
    values = {column: str(float(index) + 0.5) for column in exoplanets.BASE_COLUMNS}
    values["kepoi_name"] = f"K0{index:04d}"
    values["kepid"] = str(4000000000 + index)
    values["koi_disposition"] = disposition
    values.update(overrides)
    return ",".join(values[column] for column in exoplanets.BASE_COLUMNS)


def client_for(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


# --------------------------------------------------------------------- the ADQL query


def test_query_selects_named_columns_never_star() -> None:
    query = exoplanets.build_query(table="Q1_Q17_DR25_KOI")
    assert query.startswith("select ")
    assert "select *" not in query
    assert query.endswith("from Q1_Q17_DR25_KOI")
    assert "koi_disposition" in query
    assert "kepid" in query


def test_query_has_no_limit_clause() -> None:
    """Verified against the live endpoint: `limit` raises ORA-00933 here. If someone adds
    it back for symmetry with the CAD client, this test is the archaeology."""
    query = exoplanets.build_query(table="T")
    assert "limit" not in query.lower()


def test_query_refuses_leakage_columns() -> None:
    """`koi_pdisposition` is the pipeline's own preliminary verdict - a copy of the label.
    Refusal lives at query-build time so the column never reaches the machine at all."""
    tainted = (*exoplanets.BASE_COLUMNS, "koi_pdisposition")
    with pytest.raises(DataError, match="leakage-prone"):
        exoplanets.build_query(tainted, table="T")


@pytest.mark.parametrize("column", sorted(exoplanets.LEAKAGE_FORBIDDEN))
def test_every_forbidden_column_is_blocked(column: str) -> None:
    with pytest.raises(DataError):
        exoplanets.build_query((*exoplanets.BASE_COLUMNS, column), table="T")


def test_empty_column_list_is_rejected_instead_of_producing_bad_sql() -> None:
    with pytest.raises(DataError, match="at least one column"):
        exoplanets.build_query((), table="T")


def test_forbidden_columns_are_not_in_the_default_selection() -> None:
    assert not set(exoplanets.BASE_COLUMNS) & exoplanets.LEAKAGE_FORBIDDEN


def test_numeric_columns_exclude_ids_and_label() -> None:
    """`kepid` is an identifier, not a measurement: coercing it to a float and feeding it
    to a model would let the model learn 'bigger EPIC number means bigger star'."""
    assert "kepid" not in exoplanets.NUMERIC_COLUMNS
    assert "kepoi_name" not in exoplanets.NUMERIC_COLUMNS
    assert "koi_disposition" not in exoplanets.NUMERIC_COLUMNS


# --------------------------------------------------------------------------- the parser


def test_parse_csv_types_numbers_and_keeps_strings() -> None:
    frame = exoplanets.parse_csv(_csv([_row(1), _row(2)]))
    assert len(frame) == 2
    assert frame["koi_period"].dtype.kind == "f"
    assert frame["kepoi_name"].dtype.kind in "OU"


def test_header_is_lower_case_as_the_archive_returns_it() -> None:
    frame = exoplanets.parse_csv(_csv([_row(1)]))
    assert all(column == column.lower() for column in frame.columns)


def test_blank_cells_stay_missing_not_zero() -> None:
    """An unmeasured quantity is not a measurement of zero; the difference is the whole
    reason gradient boosting gets to see NaN."""
    frame = exoplanets.parse_csv(_csv([_row(1, koi_eccen="")]))
    assert math.isnan(float(frame.loc[0, "koi_eccen"]))
    assert frame.loc[0, "koi_period"] == 1.5


def test_error_document_is_reported_not_parsed() -> None:
    with pytest.raises(DataError, match="error document"):
        exoplanets.parse_csv('<?xml version="1.0"?><error>ORA-00904</error>')


def test_empty_body_is_reported() -> None:
    with pytest.raises(DataError, match="error document"):
        exoplanets.parse_csv("   ")


def test_missing_label_column_is_fatal() -> None:
    frame = pd.DataFrame({"koi_period": [1.0]})
    with pytest.raises(DataError, match="koi_disposition"):
        exoplanets.parse_csv(frame.to_csv(index=False))


# ------------------------------------------------------------------------- download/load


def test_download_writes_the_snapshot_and_manifest(tmp_config, cache: Cache) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_csv([_row(1), _row(2, "CANDIDATE")]))

    path = exoplanets.download(tmp_config, cache, client=client_for(handler))
    assert path.name == tmp_config.exoplanets.raw_file
    assert cache.verify(tmp_config.exoplanets.raw_file)


def test_download_sends_the_query_and_csv_format(tmp_config, cache: Cache) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text=_csv([_row(1)]))

    exoplanets.download(tmp_config, cache, client=client_for(handler))
    params = dict(httpx.QueryParams(seen[0].url.query))
    assert params["format"] == "csv"
    assert "select" in params["query"]
    assert seen[0].url.params["query"].startswith("select")


def test_download_validates_before_committing_to_disk(tmp_config, cache: Cache) -> None:
    """A bad payload must not become a cached bad payload: that turns one network glitch
    into a permanently poisoned dataset until someone notices the checksum."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="koi_period\n1.0\n")

    with pytest.raises(DataError):
        exoplanets.download(tmp_config, cache, client=client_for(handler))
    assert not cache.exists(tmp_config.exoplanets.raw_file)


def test_second_download_reuses_the_cache_without_a_request(tmp_config, cache: Cache) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, text=_csv([_row(1)]))

    exoplanets.download(tmp_config, cache, client=client_for(handler))
    exoplanets.download(tmp_config, cache, client=client_for(handler))
    assert len(calls) == 1


def test_tampered_cache_is_refetched_not_trusted(tmp_config, cache: Cache) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, text=_csv([_row(1), _row(2)]))

    exoplanets.download(tmp_config, cache, client=client_for(handler))
    cache.path(tmp_config.exoplanets.raw_file).write_text("corrupted\n", encoding="utf-8")
    exoplanets.download(tmp_config, cache, client=client_for(handler))
    assert len(calls) == 2
    assert cache.verify(tmp_config.exoplanets.raw_file)


def test_load_drops_rows_without_a_disposition(tmp_config, cache: Cache) -> None:
    """Unvalidated rows are not a fourth class; they are rows we have no label for."""
    cache.write_text(
        tmp_config.exoplanets.raw_file,
        _csv([_row(1), _row(2, disposition=""), _row(3)]),
    )
    frame = exoplanets.load(tmp_config, cache)
    assert len(frame) == 2
    assert frame.index[0] == 0  # index is reset, so downstream positional splits are safe


def test_load_keeps_every_column_we_asked_for(tmp_config, cache: Cache) -> None:
    cache.write_text(tmp_config.exoplanets.raw_file, _csv([_row(1)]))
    frame = exoplanets.load(tmp_config, cache)
    assert set(exoplanets.BASE_COLUMNS) <= set(frame.columns)


def test_upstream_failure_does_not_leave_a_partial_cache_entry(tmp_config, cache: Cache) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    with pytest.raises(UpstreamNASAError, match="failed after"):
        exoplanets.download(tmp_config, cache, client=client_for(handler))
    assert not cache.exists(tmp_config.exoplanets.raw_file)


def test_sentinel_text_in_a_numeric_column_becomes_nan(tmp_config, cache: Cache) -> None:
    """The archive does contain non-numeric junk in numeric columns; `errors="coerce"`
    turns it into NaN rather than crashing a long training run at row 40000."""
    frame = exoplanets.parse_csv(_csv([_row(1, koi_depth="N/A")]))
    assert math.isnan(float(frame.loc[0, "koi_depth"]))


def test_the_group_column_is_present_and_usable(tmp_config, cache: Cache) -> None:
    """`kepid` is the leakage boundary for the whole exoplanet task; without it in the
    snapshot the grouped split cannot even be built."""
    cache.write_text(tmp_config.exoplanets.raw_file, _csv([_row(1), _row(1, "CANDIDATE")]))
    frame = exoplanets.load(tmp_config, cache)
    assert frame[tmp_config.exoplanets.group_column].nunique() == 1
    assert len(frame) == 2
