"""Ad-hoc live probe: fetch both NASA sources and build both feature matrices.

Not part of the shipped package (it lives in scripts/ and is excluded from the wheel).
Used once to prove the data layer against the real endpoints.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nasa_space_ml.config import Config
from nasa_space_ml.data import asteroids as ast
from nasa_space_ml.data import exoplanets as exo
from nasa_space_ml.data.cache import Cache
from nasa_space_ml.features import asteroids as fast
from nasa_space_ml.features import exoplanets as fexo


def main() -> int:
    config = Config.load()
    raw = Cache(config.paths.raw_dir)
    started = time.time()

    print("=== exoplanets: TAP query ===", flush=True)
    wrote = exo.download(config, raw)
    print("wrote", str(wrote), flush=True)
    frame = exo.load(config, raw)
    print("rows", len(frame), "cols", len(frame.columns), flush=True)
    print(
        "disposition counts",
        frame[config.exoplanets.label_column].value_counts().to_dict(),
        flush=True,
    )
    print("unique hosts", frame[config.exoplanets.group_column].nunique(), flush=True)
    X, y, groups = fexo.build_matrices(frame, config)
    print("X", X.shape, "y", y.shape, "groups", groups.nunique(), flush=True)
    print("class balance", y.value_counts().to_dict(), flush=True)
    print("nan per feature", {k: int(v) for k, v in X.isna().sum().items()}, flush=True)

    print("=== asteroids: CAD paging ===", flush=True)
    wrote = ast.download(config, raw)
    print("wrote", {k: str(v) for k, v in wrote.items()}, flush=True)
    events = ast.load_events(config, raw)
    print("event rows", len(events), "cols", list(events.columns), flush=True)
    print("pha event rows", int(events["pha_flag"].sum()), flush=True)
    objects = fast.aggregate_events(events, config)
    print("objects", len(objects), "pha objects", int(objects["pha_flag"].sum()), flush=True)
    Xa, ya, _ga = fast.build_matrices(objects, config)
    print("Xa", Xa.shape, "classes", ya.value_counts().to_dict(), flush=True)
    print("nan per feature", {k: int(v) for k, v in Xa.isna().sum().items()}, flush=True)
    print(
        "moid range", float(Xa["moid_proxy_au"].min()), float(Xa["moid_proxy_au"].max()), flush=True
    )
    print(
        "h range",
        float(Xa["absolute_magnitude_h"].min()),
        float(Xa["absolute_magnitude_h"].max()),
        flush=True,
    )

    print("elapsed seconds", round(time.time() - started, 1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
