"""NASA Exoplanet Archive client (TAP / ADQL over HTTP).

The archive exposes a Table Access Protocol endpoint that runs ADQL (an SQL dialect)
and can return CSV. We ask for exactly the columns we need rather than `select *`: the
full table carries 153 columns, most of them error bars and HTML reference links.

Three verified quirks that are encoded as tests, not folklore:
  * `limit` is NOT accepted by this endpoint (it raises ORA-00933). We take the whole
    table, which is only a few thousand rows.
  * Column names are case-folded to upper case by the archive but the CSV header comes
    back lower case.
  * The DR25 KOI table has no `hostname` column, even though the confirmed-planets tables
    do. Requesting it fails with ORA-00904, so the star-level grouping key is `kepid`.
    The full 153-column list is checkable with
    `select column_name from TAP_SCHEMA.columns where table_name='Q1_Q17_DR25_KOI'`.
"""

from __future__ import annotations

import io
from pathlib import Path

import httpx
import pandas as pd

from ..config import Config
from ..errors import DataError
from .cache import Cache
from .http import fetch_text

# Columns we actually use. Split into the ones that become features and the ones that
# exist purely to group or label rows.
BASE_COLUMNS: tuple[str, ...] = (
    "kepoi_name",
    "kepid",
    "koi_disposition",
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_impact",
    "koi_eccen",
    "koi_incl",
    "koi_sma",
    "koi_teq",
    "koi_steff",
    "koi_srad",
    "koi_smass",
    "koi_slogg",
    "koi_num_transits",
    "koi_model_snr",
    "koi_max_sngle_ev",
    "koi_kepmag",
    "koi_count",
)

# Columns that must never reach a training matrix. `koi_pdisposition` is the pipeline's
# own preliminary verdict - effectively a copy of the label, and including it produces a
# model that scores beautifully and means nothing.
LEAKAGE_FORBIDDEN: frozenset[str] = frozenset(
    {"koi_pdisposition", "koi_vet_stat", "koi_quarters", "koi_delivname"}
)

NUMERIC_COLUMNS: tuple[str, ...] = tuple(
    column
    for column in BASE_COLUMNS
    if column not in {"kepoi_name", "kepid", "koi_disposition"}
)


def build_query(columns: tuple[str, ...] = BASE_COLUMNS, *, table: str) -> str:
    """Compose the ADQL select. Deliberately avoids `limit` - see module docstring."""
    if not columns:
        raise DataError("build_query needs at least one column")
    overlap = set(columns) & LEAKAGE_FORBIDDEN
    if overlap:
        raise DataError(
            f"refusing to request leakage-prone columns: {sorted(overlap)}",
            details={"columns": sorted(overlap)},
        )
    return f"select {', '.join(columns)} from {table}"


def parse_csv(text: str) -> pd.DataFrame:
    """Turn the archive's CSV into numeric-typed dataframe.

    Empty cells in a survey catalogue mean "not measured", not zero, so we let pandas
    read them as NaN and handle them explicitly downstream.
    """
    if not text.strip() or text.lstrip().startswith("<?xml"):
        snippet = text[:240].replace("\n", " ")
        raise DataError(
            f"TAP returned an error document instead of CSV: {snippet}",
            details={"kind": "bad_payload"},
        )
    frame = pd.read_csv(io.StringIO(text))
    if "koi_disposition" not in frame.columns:
        raise DataError("TAP CSV is missing the koi_disposition label column")
    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def download(
    config: Config,
    cache: Cache,
    client: httpx.Client | None = None,
) -> Path:
    """Fetch the KOI table into the cache, or reuse what is already there."""
    settings = config.exoplanets
    if cache.exists(settings.raw_file) and cache.verify(settings.raw_file):
        return cache.path(settings.raw_file)

    query = build_query(table=settings.table)
    owns_client = client is None
    http = client or _client_for(config)
    try:
        text = fetch_text(
            http,
            settings.source,
            params={"query": query, "format": "csv"},
            settings=config.http,
        )
    finally:
        if owns_client:
            http.close()

    parse_csv(text)  # validate before we commit it to disk
    return cache.write_text(settings.raw_file, text)


def load(config: Config, cache: Cache) -> pd.DataFrame:
    """Read the cached snapshot into a dataframe, verifying provenance first."""
    frame = parse_csv(cache.read_text(config.exoplanets.raw_file))
    frame = frame[frame[config.exoplanets.label_column].notna()].copy()
    return frame.reset_index(drop=True)


def _client_for(config: Config) -> httpx.Client:
    from .http import make_client

    return make_client(config.http)
