"""Shared fixtures: a config pointed at a throwaway directory and small synthetic
NASA-shaped frames.

Nothing in here touches the network. The data-layer tests drive `httpx.MockTransport`,
which is the whole reason `http.py` takes an injected client instead of calling
`httpx.get` inline. The synthetic frames are deliberately ugly - NaNs, out-of-range
values, tied groups - because the code paths that matter are the ones that handle bad
survey data, not the ones that handle tidy data.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nasa_space_ml.config import Config, resolve_default_config_path
from nasa_space_ml.data.cache import Cache
from nasa_space_ml.features import exoplanets as exoplanet_features


@pytest.fixture()
def config() -> Config:
    """The real shipped config, so a test that depends on a value catches a bad value.

    Resolved through the same search the CLI uses, not through `DEFAULT_CONFIG_PATH`,
    because a test that hardcodes a different path rule passes locally and fails wherever
    the package was installed non-editably. That is not hypothetical: the repository's
    second CI run died exactly there.
    """
    return Config.load(resolve_default_config_path())


@pytest.fixture()
def tmp_config(config: Config, tmp_path: Path) -> Config:
    """The same config with every output path redirected into `tmp_path`."""
    return replace(
        config,
        root=tmp_path,
        paths=replace(
            config.paths,
            raw_dir=tmp_path / "raw",
            processed_dir=tmp_path / "processed",
            model_dir=tmp_path / "models",
            report_dir=tmp_path / "reports",
        ),
    )


@pytest.fixture()
def cache(tmp_config: Config) -> Cache:
    tmp_config.paths.ensure()
    return Cache(tmp_config.paths.raw_dir)


def _exo_row(index: int, label: str, star: int) -> dict[str, object]:
    """One plausible Kepler row.

    Numbers are anchored on real DR25 ranges: a hot Jupiter (period ~3 d, depth ~1000 ppm,
    SNR ~1200) reads very differently from an eclipsing binary false positive (depth
    ~10000 ppm, huge single-event SNR), and the fixtures rely on that being true.
    """
    if label == "FALSE POSITIVE":
        depth, snr, single, prad, period = 12000.0, 40.0, 60.0, 28.0, 1.4
    elif label == "CANDIDATE":
        depth, snr, single, prad, period = 60.0, 9.0, 8.0, 2.2, 9.7
    else:
        depth, snr, single, prad, period = 300.0, 400.0, 25.0, 6.0, 3.1
    return {
        "kepoi_name": f"{100000000 + index:09d}.01",
        "kepid": 4000000000 + star,
        "koi_disposition": label,
        "koi_period": period + index * 1e-4,
        "koi_duration": 3.0 + index * 1e-3,
        "koi_depth": depth,
        "koi_prad": prad,
        "koi_impact": 0.4 if label != "FALSE POSITIVE" else 0.95,
        "koi_eccen": 0.0,
        "koi_incl": 89.8,
        "koi_sma": 0.05 + index * 1e-4,
        "koi_teq": 600.0 + index,
        "koi_steff": 5500.0 + star,
        "koi_srad": 0.95,
        "koi_smass": 0.95,
        "koi_slogg": 4.45,
        "koi_num_transits": 12 + index % 4,
        "koi_model_snr": snr,
        "koi_max_sngle_ev": single,
        "koi_kepmag": 13.5,
        "koi_count": 1,
    }


@pytest.fixture()
def exo_frame(config: Config) -> pd.DataFrame:
    """A balanced-ish 60-row frame across 20 distinct host stars, so group logic is testable."""
    labels = ["CONFIRMED", "FALSE POSITIVE", "CANDIDATE"]
    rows = [_exo_row(i, labels[i % 3], star=i % 20) for i in range(60)]
    frame = pd.DataFrame(rows)
    # A survey catalogue always has unmeasured cells; the model has to tolerate them.
    frame.loc[0, "koi_eccen"] = np.nan
    frame.loc[1, "koi_max_sngle_ev"] = np.nan
    return frame


@pytest.fixture()
def exo_matrices(config: Config, exo_frame: pd.DataFrame):
    return exoplanet_features.build_matrices(exo_frame, config)


def pha_event(index: int, des: str, *, hazardous: bool, orbit_id: str = "1") -> dict[str, object]:
    """One CAD close-approach row.

    A PHA in this fixture means MOID under 0.05 au and absolute magnitude under 22.0, which
    is NASA's published definition; the rule baseline and the label must therefore agree,
    and that agreement is asserted in test_baselines.
    """
    moid = 0.0040 + index * 1e-5 if hazardous else 0.12 + index * 1e-3
    mag = 21.4 if hazardous else 25.8
    return {
        "des": des,
        "orbit_id": orbit_id,
        "jd": 2459000.5 + index,
        "cd": f"2020-Jan-{(index % 28) + 1:02d} 02:06",
        "dist": moid * 1.4,
        "dist_min": moid,
        "dist_max": moid * 1.6,
        "v_rel": 12.0 + index,
        "v_inf": 6.5 + index,
        "t_sigma_f": "2_04:30" if hazardous else "60_12:00",
        "h": mag,
        "diameter": 0.31 if hazardous else math.nan,
        "diameter_sigma": 0.04,
    }


HAZARDOUS_COUNT = 4
OBJECT_COUNT = 12


@pytest.fixture()
def cad_raw_events() -> pd.DataFrame:
    """The event table exactly as CAD's positional arrays decode: 12 objects, the first of
    them seen on three separate approaches with three different orbit solutions."""
    rows: list[dict[str, object]] = []
    index = 0
    for object_number in range(OBJECT_COUNT):
        hazardous = object_number < HAZARDOUS_COUNT
        des = f"2020 NP{object_number:02d}"
        copies = 3 if object_number == 0 else 1
        for copy in range(copies):
            rows.append(pha_event(index, des, hazardous=hazardous, orbit_id=str(copy + 1)))
            index += 1
    return pd.DataFrame(rows)


@pytest.fixture()
def event_cache(config: Config, cad_raw_events: pd.DataFrame, tmp_path: Path) -> Cache:
    """The two CAD artifacts written into a real Cache, as `download` would leave them."""
    settings = config.asteroids
    event_cache = Cache(tmp_path / "event_cache")
    event_cache.write_text(settings.raw_file, cad_raw_events.to_csv(index=False))
    hazardous = sorted(f"2020 NP{number:02d}" for number in range(HAZARDOUS_COUNT))
    event_cache.write_text(settings.pha_file, "\n".join(hazardous))
    return event_cache


@pytest.fixture()
def cad_events(config: Config, event_cache: Cache) -> pd.DataFrame:
    """The *enriched* event table, produced by the real `load_events`.

    Going through the cached-CSV path rather than fabricating the two extra columns means
    these tests exercise the same `pha_flag` join and `t_sigma_seconds` conversion that
    training sees - a fixture built by hand can drift from production unnoticed.
    """
    from nasa_space_ml.data.asteroids import load_events

    return load_events(config, event_cache)


@pytest.fixture()
def asteroid_object_frame(config: Config, cad_events: pd.DataFrame) -> pd.DataFrame:
    """One row per asteroid, labelled by object-level PHA status - the model's input table."""
    from nasa_space_ml.features.asteroids import aggregate_events

    return aggregate_events(cad_events, config).reset_index(drop=True)
