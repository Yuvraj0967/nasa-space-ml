"""Typed access to configs/settings.yaml.

Why a hand-rolled loader instead of a framework: the config is small, read-only, and we
want an explicit error naming the missing key. Training and serving both load the same
object, which is what prevents the classic "the API scores a different feature list than
the model was trained on" bug.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError

#: The environment variable a container uses to point at its own copy of the settings.
#: `api/app.py` imports this rather than repeating the literal, because two spellings of a
#: variable name is how a container ends up silently reading the wrong thresholds.
CONFIG_ENV = "NSM_CONFIG"

#: Where the settings file lives in a source checkout, computed from the installed package
#: location. Kept as a constant because tests and error messages want a stable name to
#: point at; it is the *last* candidate, not the answer.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "settings.yaml"


def config_candidates(start: Path | None = None) -> list[Path]:
    """Every place the settings file could be, best match first.

    The order is the one that survives the three ways this project is actually run:

    1. `$NSM_CONFIG` - a container copies `configs/` somewhere the repository layout does
       not imply, and an explicitly set path that gets ignored is a silent misconfiguration.
    2. The nearest `configs/settings.yaml` at or above the current working directory -
       because `pip install .` puts the package in site-packages, where `parents[2]` is a
       directory that has never seen this file. The first CI run of this repository found
       that out: an installed-but-not-editable `nasa-space-ml status` looked for its config
       two levels above site-packages and failed.
    3. The package-relative checkout default, which is correct for the editable install
       everyone develops against.

    Split from the lookup so the ordering is assertable without moving directories.
    """
    base = (start or Path.cwd()).resolve()
    candidates: list[Path] = []
    from_env = os.environ.get(CONFIG_ENV)
    if from_env:
        candidates.append(Path(from_env))
    candidates.extend(
        directory / "configs" / "settings.yaml" for directory in (base, *base.parents)
    )
    candidates.append(DEFAULT_CONFIG_PATH)
    return candidates


def resolve_default_config_path(start: Path | None = None) -> Path:
    """The first candidate that exists, or the package default so errors name something real."""
    for candidate in config_candidates(start):
        if candidate.is_file():
            return candidate
    return DEFAULT_CONFIG_PATH


def _need(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"{where}: required key {key!r} is missing", details={"path": where})
    return mapping[key]


@dataclass(frozen=True)
class Paths:
    raw_dir: Path
    processed_dir: Path
    model_dir: Path
    report_dir: Path

    def ensure(self) -> None:
        for directory in (self.raw_dir, self.processed_dir, self.model_dir, self.report_dir):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class HttpSettings:
    timeout_seconds: float
    max_attempts: int
    backoff_base_seconds: float
    backoff_cap_seconds: float
    user_agent: str


@dataclass(frozen=True)
class ExoplanetSettings:
    source: str
    table: str
    raw_file: str
    processed_file: str
    label_column: str
    group_column: str
    labels: tuple[str, ...]


@dataclass(frozen=True)
class AsteroidSettings:
    source: str
    expected_api_version: str
    date_min: str
    date_max: str
    dist_max_au: float
    page_limit: int
    max_pages: int
    raw_file: str
    pha_file: str
    processed_file: str
    group_column: str
    labels: tuple[str, ...]
    pha_distance_au: float
    pha_absolute_magnitude: float
    default_geometric_albedo: float


@dataclass(frozen=True)
class SplitSettings:
    test_size: float
    val_size: float
    n_folds: int

    def validate(self) -> None:
        if self.test_size + self.val_size >= 1.0:
            raise ConfigError("split.test_size + split.val_size must be < 1.0")
        if self.n_folds < 2:
            raise ConfigError("split.n_folds must be at least 2")


@dataclass(frozen=True)
class Config:
    root: Path
    seed: int
    paths: Paths
    http: HttpSettings
    exoplanets: ExoplanetSettings
    asteroids: AsteroidSettings
    split: SplitSettings
    raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        """Read settings.yaml and return a validated Config.

        With no argument the file is found by `resolve_default_config_path`, and relative
        paths inside the file then resolve against the repository root (the parent of the
        configs/ directory that was actually read) - so the CLI works from any current
        working directory, and an installed wheel still finds the checkout it was run from.
        """
        config_path = Path(path) if path else resolve_default_config_path()
        if not config_path.exists():
            raise ConfigError(f"config file not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as handle:
            try:
                data = yaml.safe_load(handle)
            except yaml.YAMLError as exc:  # pragma: no cover - defensive
                raise ConfigError(f"config file is not valid YAML: {exc}") from exc

        if not isinstance(data, dict):
            raise ConfigError("config file must contain a mapping at the top level")

        root = config_path.resolve().parent.parent
        paths_raw = _need(data, "paths", "settings")
        paths = Paths(
            raw_dir=root / _need(paths_raw, "raw_dir", "paths"),
            processed_dir=root / _need(paths_raw, "processed_dir", "paths"),
            model_dir=root / _need(paths_raw, "model_dir", "paths"),
            report_dir=root / _need(paths_raw, "report_dir", "paths"),
        )

        http_raw = _need(data, "http", "settings")
        http = HttpSettings(
            timeout_seconds=float(_need(http_raw, "timeout_seconds", "http")),
            max_attempts=int(_need(http_raw, "max_attempts", "http")),
            backoff_base_seconds=float(_need(http_raw, "backoff_base_seconds", "http")),
            backoff_cap_seconds=float(_need(http_raw, "backoff_cap_seconds", "http")),
            user_agent=str(_need(http_raw, "user_agent", "http")),
        )

        exo_raw = _need(data, "exoplanets", "settings")
        exoplanets = ExoplanetSettings(
            source=_need(exo_raw, "source", "exoplanets"),
            table=_need(exo_raw, "table", "exoplanets"),
            raw_file=_need(exo_raw, "raw_file", "exoplanets"),
            processed_file=_need(exo_raw, "processed_file", "exoplanets"),
            label_column=_need(exo_raw, "label_column", "exoplanets"),
            group_column=_need(exo_raw, "group_column", "exoplanets"),
            labels=tuple(_need(exo_raw, "labels", "exoplanets")),
        )

        ast_raw = _need(data, "asteroids", "settings")
        asteroids = AsteroidSettings(
            source=_need(ast_raw, "source", "asteroids"),
            expected_api_version=str(_need(ast_raw, "expected_api_version", "asteroids")),
            date_min=str(_need(ast_raw, "date_min", "asteroids")),
            date_max=str(_need(ast_raw, "date_max", "asteroids")),
            dist_max_au=float(_need(ast_raw, "dist_max_au", "asteroids")),
            page_limit=int(_need(ast_raw, "page_limit", "asteroids")),
            max_pages=int(_need(ast_raw, "max_pages", "asteroids")),
            raw_file=_need(ast_raw, "raw_file", "asteroids"),
            pha_file=_need(ast_raw, "pha_file", "asteroids"),
            processed_file=_need(ast_raw, "processed_file", "asteroids"),
            group_column=_need(ast_raw, "group_column", "asteroids"),
            labels=tuple(_need(ast_raw, "labels", "asteroids")),
            pha_distance_au=float(_need(ast_raw, "pha_distance_au", "asteroids")),
            pha_absolute_magnitude=float(_need(ast_raw, "pha_absolute_magnitude", "asteroids")),
            default_geometric_albedo=float(_need(ast_raw, "default_geometric_albedo", "asteroids")),
        )

        split_raw = _need(data, "split", "settings")
        split = SplitSettings(
            test_size=float(_need(split_raw, "test_size", "split")),
            val_size=float(_need(split_raw, "val_size", "split")),
            n_folds=int(_need(split_raw, "n_folds", "split")),
        )
        split.validate()

        return cls(
            root=root,
            seed=int(_need(data, "random_seed", "settings")),
            paths=paths,
            http=http,
            exoplanets=exoplanets,
            asteroids=asteroids,
            split=split,
            raw=data,
        )
