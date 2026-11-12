"""JPL Small-Body Database Close Approach Data (CAD) client.

Verified API surface (from the official doc page plus live probes), encoded as tests:
  * pagination is `limit` + `limit-from`; `count` and `rows_count` are rejected with 400
  * `pha=1` is a FILTER on objects, not a column - the response `fields` array has no
    hazardous flag
  * `signature.version` must be checked, because the payload is positional arrays and a
    silent field reordering would corrupt every number we compute
  * `t_sigma_f` is a formatted string ("< 00:01", "13:02", "2_09:08"), not a number
  * `diameter=1` is what makes the optional diameter columns appear

Label construction is the interesting part: PHA status belongs to the asteroid, while CAD
hands us one row per close-approach event. So we page through all events, separately page
through the PHA events, and label by object membership. Deriving the label from `dist` and
`h` here instead would quietly make the task a rule-reimplementation exercise.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import httpx
import pandas as pd

from ..config import Config
from ..errors import DataError
from .cache import Cache
from .http import fetch_json

EVENT_FIELDS: tuple[str, ...] = (
    "des",
    "orbit_id",
    "jd",
    "cd",
    "dist",
    "dist_min",
    "dist_max",
    "v_rel",
    "v_inf",
    "t_sigma_f",
    "h",
    "diameter",
    "diameter_sigma",
)

NUMERIC_EVENT_COLUMNS: tuple[str, ...] = (
    "jd",
    "dist",
    "dist_min",
    "dist_max",
    "v_rel",
    "v_inf",
    "h",
    "diameter",
    "diameter_sigma",
)

_T_SIGMA = re.compile(r"^(?:\s*<\s*)?(?:(?P<days>\d+)_)?(?:(?P<hours>\d+):)?(?P<minutes>\d+)$")


def parse_t_sigma(raw: str | None) -> float | None:
    """Convert CAD's formatted 3-sigma time uncertainty into seconds.

    Formats seen in the wild: "< 00:01" (under a minute), "13:02" (13h02m),
    "2_09:08" (2 days, 9 hours, 8 minutes). Returns None when unparseable rather than
    guessing zero, because "unknown timing precision" is not "perfect timing precision".
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.lower() == "none":
        return None
    match = _T_SIGMA.match(text)
    if not match:
        return None
    groups = match.groupdict()
    days = int(groups["days"] or 0)
    hours = int(groups["hours"] or 0)
    minutes = int(groups["minutes"] or 0)
    return float(days * 86400 + hours * 3600 + minutes * 60)


def build_params(
    config: Config,
    *,
    limit_from: int,
    pha_only: bool = False,
) -> dict:
    settings = config.asteroids
    params = {
        "date-min": settings.date_min,
        "date-max": settings.date_max,
        "dist-max": str(settings.dist_max_au),
        "diameter": "1",
        "limit": str(settings.page_limit),
        "limit-from": str(limit_from),
        "sort": "date",
    }
    if pha_only:
        params["pha"] = "1"
    return params


def _validate_signature(payload: dict, config: Config) -> None:
    """Check the error code and the API version before trusting anything else."""
    if "code" in payload and str(payload.get("code")) != "200":
        raise DataError(
            f"CAD API returned error code {payload.get('code')}: {payload.get('message')}",
            details={"payload_keys": sorted(payload)[:8]},
        )
    signature = payload.get("signature") or {}
    version = str(signature.get("version", ""))
    if version != config.asteroids.expected_api_version:
        raise DataError(
            "CAD API version changed: this client was written against "
            f"{config.asteroids.expected_api_version!r} but the endpoint reports "
            f"{version!r}. The response is positional, so field order must be re-checked "
            "before parsing.",
            details={"found": version, "expected": config.asteroids.expected_api_version},
        )


def _page(
    client: httpx.Client,
    config: Config,
    *,
    pha_only: bool,
    sleeper,
) -> pd.DataFrame:
    """Page through the CAD endpoint until exhausted or max_pages is reached.

    Stopping rules, in the order they are checked, because the live endpoint taught us all
    three the hard way:
      * an empty `data` array ends the walk (the feed's `total` drifts by a few rows
        between requests as JPL re-runs its own solutions, so a page past the end is a
        normal occurrence, not a corrupt response);
      * a *short* page ends the walk - the reliable signal that we have the tail;
      * `total` is read once, from the first page, so later drift cannot make us loop.

    A payload with rows but no `fields` array is fatal: without the schema the rows are
    positional arrays we would silently mis-assign to the wrong columns.
    """
    settings = config.asteroids
    collected: list[list] = []
    seen_fields: list[str] | None = None
    total: int | None = None
    limit_from = 1
    pages = 0

    while pages < settings.max_pages:
        pages += 1
        params = build_params(config, limit_from=limit_from, pha_only=pha_only)
        payload = fetch_json(
            client, settings.source, params=params, settings=config.http, sleeper=sleeper
        )
        _validate_signature(payload, config)

        data = payload.get("data") or []
        if not data:
            break

        fields = payload.get("fields")
        if not isinstance(fields, list):
            raise DataError(
                "CAD returned rows without a 'fields' array, so the columns cannot be "
                "named; refusing to guess a positional mapping",
                details={"page": pages, "keys": sorted(payload), "rows": len(data)},
            )
        if seen_fields is None:
            seen_fields = list(fields)
            total = int(payload.get("total", 0))
        elif list(fields) != seen_fields:
            raise DataError(
                "CAD field order changed between pages; refusing to align columns",
                details={"page": pages},
            )

        collected.extend(data)
        if len(data) < settings.page_limit:
            break
        limit_from += len(data)
        if total is not None and limit_from > total:
            break
    else:
        raise DataError(
            f"CAD pagination exceeded max_pages={settings.max_pages}; the window is too "
            "large or the endpoint is returning duplicate pages",
            details={"rows": len(collected)},
        )

    if seen_fields is None:
        raise DataError("CAD returned no rows for this window", details={"pages": pages})
    frame = pd.DataFrame(collected, columns=seen_fields)
    for column in NUMERIC_EVENT_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def download(
    config: Config,
    cache: Cache,
    client: httpx.Client | None = None,
    *,
    sleeper=lambda seconds: None,
) -> dict[str, Path]:
    """Cache both the full event table and the list of hazardous object names."""
    settings = config.asteroids
    wrote: dict[str, Path] = {}
    owns = client is None
    http = client or _client_for(config)
    try:
        if not (cache.exists(settings.raw_file) and cache.verify(settings.raw_file)):
            events = _page(http, config, pha_only=False, sleeper=sleeper)
            if events.empty:
                raise DataError("CAD returned zero close approaches for this window")
            wrote[settings.raw_file] = cache.write_text(
                settings.raw_file, events.to_csv(index=False)
            )

        if not (cache.exists(settings.pha_file) and cache.verify(settings.pha_file)):
            pha_events = _page(http, config, pha_only=True, sleeper=sleeper)
            designations = (
                sorted(set(pha_events["des"].astype(str))) if not pha_events.empty else []
            )
            wrote[settings.pha_file] = cache.write_text(settings.pha_file, "\n".join(designations))
    finally:
        if owns:
            http.close()
    return wrote


def load_events(config: Config, cache: Cache) -> pd.DataFrame:
    """Cached events as a dataframe, with the object-level PHA label joined on."""
    frame = pd.read_csv(io.StringIO(cache.read_text(config.asteroids.raw_file)))
    designations = _pha_designations(config, cache)
    frame["pha_flag"] = frame["des"].astype(str).isin(designations)
    for column in NUMERIC_EVENT_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["t_sigma_seconds"] = frame["t_sigma_f"].map(parse_t_sigma)
    return frame


def _pha_designations(config: Config, cache: Cache) -> frozenset[str]:
    text = cache.read_text(config.asteroids.pha_file).strip()
    if not text:
        return frozenset()
    return frozenset(line.strip() for line in text.splitlines() if line.strip())


def io_string(payload: str):
    """Wrap cached bytes for pandas without touching the filesystem twice."""
    return io.StringIO(payload)


def _client_for(config: Config) -> httpx.Client:
    from .http import make_client

    return make_client(config.http)
