"""Live probes against the two real NASA endpoints.

Deselected by default (`addopts = "-m 'not network'"` in pyproject.toml) because CI must not
depend on a government web server being up, and because a student on hotel wifi should get a
green suite. Run them deliberately:

    pytest -m network -s

Each writes into a temporary cache, never into `data/raw`, so a live probe cannot quietly
change the snapshot the shipped models were trained on. What they prove is the thing that
mocked tests structurally cannot: that the endpoint still answers, still speaks the same
shape, and still has the columns this project asked for. The mock tests prove our parsing;
these prove the world.
"""

from __future__ import annotations

import pytest

from nasa_space_ml.config import Config, resolve_default_config_path
from nasa_space_ml.data import asteroids, exoplanets
from nasa_space_ml.data.cache import Cache

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def live_config() -> Config:
    return Config.load(resolve_default_config_path())


def test_tap_endpoint_still_answers_with_the_columns_we_ask_for(
    live_config: Config, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """One real ADQL query through the exact code path `nasa-space-ml fetch` uses."""
    settings = live_config.exoplanets
    cache = Cache(tmp_path_factory.mktemp("live_tap"))
    path = exoplanets.download(live_config, cache)
    assert path.name == settings.raw_file and path.exists()

    frame = exoplanets.load(live_config, cache)
    assert len(frame) > 5_000, f"archive returned only {len(frame)} rows; has DR25 been retired?"
    assert set(settings.labels).issubset(set(frame[settings.label_column].dropna()))
    missing = [column for column in exoplanets.BASE_COLUMNS if column not in frame.columns]
    assert not missing, f"archive no longer serves {missing}"


def test_cad_paging_builds_both_artifacts_in_a_temp_cache(
    live_config: Config, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """The full paginated walk, including the version assertion, against the live API."""
    cache = Cache(tmp_path_factory.mktemp("live_raw"))
    wrote = asteroids.download(live_config, cache)

    assert set(wrote) == {
        live_config.asteroids.raw_file,
        live_config.asteroids.pha_file,
    }
    events = asteroids.load_events(live_config, cache)
    assert len(events) > 10_000, "CAD returned suspiciously few close approaches"
    assert {"des", "dist_min", "v_rel", "t_sigma_f", "h"} <= set(events.columns)

    designations = cache.read_text(live_config.asteroids.pha_file).splitlines()
    assert len(designations) > 500, "the hazardous list looks truncated"


def test_the_snapshot_the_shipped_models_were_built_from_is_still_reachable(
    live_config: Config, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Compare a fresh fingerprint to the one recorded in the committed artifacts.

    This is allowed to differ - NASA re-publishes - so it reports rather than fails. It only
    fails if the *two artifacts disagree with each other*, which would mean one was rebuilt
    without the other and is a genuine repo bug.
    """
    import json

    cache = Cache(tmp_path_factory.mktemp("live_fp"))
    exoplanets.download(live_config, cache)
    asteroids.download(live_config, cache)
    fresh = cache.dataset_fingerprint()

    recorded = {}
    for name in ("exoplanet_disposition", "asteroid_pha"):
        path = live_config.paths.model_dir / f"{name}.json"
        if not path.exists():
            pytest.skip(f"no {name}.json; run `nasa-space-ml train`")
        recorded[name] = json.loads(path.read_text())["dataset_fingerprint"]

    assert set(recorded.values()) == {next(iter(recorded.values()))}, (
        "the two committed artifacts were trained on different snapshots"
    )
    if fresh != next(iter(recorded.values())):
        pytest.skip(
            "NASA has re-published since the shipped snapshot; numbers in README and the "
            "model cards no longer describe live data. Re-fetch, retrain, re-document."
        )
