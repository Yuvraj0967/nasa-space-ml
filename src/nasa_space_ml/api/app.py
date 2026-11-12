"""Application factory.

`create_app` takes an optional config path so tests can build an app against a temporary
repository with a toy model, and the CLI can build one against whatever `--config` said. The
module-level `app` exists only for `uvicorn nasa_space_ml.api.app:app`; the CLI passes the
factory's return value directly.

Static assets are mounted from `web/` at `/`, with the API under `/api/v1`. Nothing else is
served, and there are no external origins: the front end loads no CDN JavaScript or CSS, so
the page works on a laptop with the wifi off - which is the only way a demo survives a
review session.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..config import Config
from ..data.cache import Cache
from .deps import initialise
from .middleware import RequestIdMiddleware
from .problem import register_handlers
from .routers import asteroids, exoplanets, meta

CONFIG_ENV = "NSM_CONFIG"
WEB_ENV = "NSM_WEB_DIR"


def resolve_config_path(config_path: str | Path | None = None) -> Path | None:
    """Explicit argument, then `$NSM_CONFIG`, then the package-relative default.

    The environment variable exists because a Docker image installs this package into
    site-packages, where `parents[2]/configs/settings.yaml` is not a real directory. Without
    it the honest alternatives were an editable install in a production image or a config
    path that silently resolved to nothing, and a service that cannot find its thresholds is
    worse than one that refuses to start.
    """
    if config_path is not None:
        return Path(config_path)
    from_env = os.environ.get(CONFIG_ENV)
    return Path(from_env) if from_env else None


def web_candidates() -> list[Path]:
    """The places the front end could live, best match first.

    Split out from `web_dir` so the degradation path - "nothing found" - is testable without
    moving the repository underneath a running process.
    """
    candidates: list[Path] = []
    from_env = os.environ.get(WEB_ENV)
    if from_env:
        candidates.append(Path(from_env))
    candidates.append(Path(__file__).resolve().parents[3] / "web")
    candidates.append(Path.cwd() / "web")
    return candidates


def web_dir() -> Path | None:
    """Where the front end lives, or None when there is nothing to mount.

    See `web_candidates` for the order. The first candidate that actually contains
    `index.html` wins, so a missing directory degrades to "API only" instead of an
    import-time crash - which is what a container that copies only the Python package gets.
    """
    for candidate in web_candidates():
        if (candidate / "index.html").is_file():
            return candidate
    return None


WEB_DIR = web_dir()


def create_app(config_path: str | Path | None = None) -> FastAPI:
    config = Config.load(resolve_config_path(config_path))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config.paths.ensure()
        app.state.nsm = initialise(config, Cache(config.paths.raw_dir))
        loaded = [name for name, item in app.state.nsm.models.items()]
        for name, reason in app.state.nsm.model_errors.items():
            print(f"[startup] model {name}: {reason}")
        print(f"[startup] models loaded: {loaded or 'none'}")
        yield

    app = FastAPI(
        title="nasa-space-ml",
        description=(
            "Two classifiers over real NASA data - Kepler candidate dispositions and "
            "JPL potentially-hazardous asteroid status - with the feature contracts, "
            "provenance and failure modes exposed rather than hidden."
        ),
        version=__version__,
        lifespan=lifespan,
    )
    app.add_middleware(RequestIdMiddleware)
    register_handlers(app)

    app.include_router(meta.router)
    app.include_router(exoplanets.router)
    app.include_router(asteroids.router)

    assets = web_dir()
    if assets is not None:
        app.mount("/static", StaticFiles(directory=str(assets)), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> Any:
            return FileResponse(str(assets / "index.html"))

    @app.get("/api/v1", tags=["meta"], include_in_schema=False)
    def api_root() -> dict[str, Any]:
        return {
            "service": "nasa-space-ml",
            "version": __version__,
            "endpoints": [
                "GET  /api/v1/health",
                "GET  /api/v1/meta",
                "GET  /api/v1/exoplanets/schema",
                "GET  /api/v1/exoplanets/sample",
                "POST /api/v1/exoplanets/predict",
                "GET  /api/v1/asteroids/schema",
                "GET  /api/v1/asteroids/sample",
                "POST /api/v1/asteroids/predict",
            ],
            "docs": "/docs",
        }

    return app


app = create_app()
