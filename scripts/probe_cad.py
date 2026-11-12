"""Focused CAD probe: show the exact URL httpx builds and the body it gets back."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx  # noqa: E402

from nasa_space_ml.config import Config  # noqa: E402
from nasa_space_ml.data.asteroids import build_params  # noqa: E402


def main() -> int:
    config = Config.load()
    for limit_from in (1, 5001):
        params = build_params(config, limit_from=limit_from)
        request = httpx.Request("GET", config.asteroids.source, params=params)
        print("params:", params)
        print("url  :", str(request.url))
        response = httpx.get(
            config.asteroids.source,
            params=params,
            timeout=120,
            headers={"User-Agent": config.http.user_agent},
        )
        print("status:", response.status_code, "bytes:", len(response.text))
        print("body :", response.text[:260])
        try:
            payload = response.json()
            print("keys :", sorted(payload), "count:", payload.get("count"))
        except Exception as exc:  # noqa: BLE001
            print("json failure:", exc)
        print("-" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
