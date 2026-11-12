"""Command line entry point: fetch, train, report, serve.

Deliberately thin. Every subcommand resolves a `Config` and a `Cache`, then calls into the
layer that actually knows what it is doing, so the API and the CLI can never drift: they run
the same functions on the same artifacts.

`--offline` is the switch that makes this usable in a test runner or on a plane. It refuses
to touch the network and reads only what is already in `data/raw`, which is also how the CI
suite proves the pipeline end-to-end without depending on NASA being up.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import Config
from .data.cache import Cache
from .errors import NasaSpaceMlError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nasa-space-ml",
        description="Two NASA-backed classifiers and the API that serves them.",
    )
    parser.add_argument("--config", type=Path, default=None, help="path to settings.yaml")
    parser.add_argument("--offline", action="store_true", help="never touch the network")
    parser.add_argument("--version", action="version", version=f"nasa-space-ml {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fetch", help="download and cache both NASA sources")
    train = sub.add_parser("train", help="cross-validate candidates, select, refit, score")
    train.add_argument("--task", choices=["exoplanets", "asteroids", "both"], default="both")
    train.add_argument("--no-importance", action="store_true", help="skip permutation importance")
    sub.add_parser("status", help="show what is cached and what is trained")

    serve = sub.add_parser("serve", help="run the FastAPI service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    everything = sub.add_parser("all", help="fetch, then train")
    everything.add_argument("--task", choices=["exoplanets", "asteroids", "both"], default="both")
    return parser


def _print_status(config: Config, cache: Cache) -> int:
    print(f"repository root : {config.root}")
    print(f"raw cache dir   : {cache.directory}")
    manifest = cache.manifest()
    if manifest:
        for name in sorted(manifest):
            entry = manifest[name]
            print(f"  cached {name:26s} {entry['bytes']:>10,} bytes  {entry['written_utc']}")
            print(f"         sha256 {entry['sha256'][:16]}  verified={cache.verify(name)}")
    else:
        print("  (nothing cached yet - run `nasa-space-ml fetch`)")

    from .models.registry import load_metadata

    for name in ("exoplanet_disposition", "asteroid_pha"):
        try:
            metadata = load_metadata(config.paths.model_dir, name)
        except NasaSpaceMlError as exc:
            print(f"model {name:22s} : not trained ({exc})")
            continue
        metrics = metadata.get("metrics") or {}
        print(f"model {name:22s} : {metadata['algorithm']}")
        print(
            f"  trained {metadata['trained_utc']}  rows={metadata['rows']:,}  "
            f"groups={metadata['groups']:,}  features={metadata['n_features']}"
        )
        print(f"  headline metrics: {_headline(metrics)}")
    return 0


def _headline(metrics: dict) -> str:
    parts = []
    for key in ("macro_f1", "average_precision", "roc_auc", "brier"):
        if key in metrics and metrics[key] is not None:
            parts.append(f"{key}={metrics[key]:.4f}")
    return "  ".join(parts) or "n/a"


def _fetch(config: Config, cache: Cache, *, offline: bool) -> int:
    from .data import asteroids as ast_data
    from .data import exoplanets as exo_data

    if offline:
        print("--offline: skipping download, reading the cache only")
    else:
        print(f"fetching exoplanet KOI table from {config.exoplanets.source}")
        path = exo_data.download(config, cache)
        print(f"  -> {path}")
        print(f"fetching JPL close-approach events from {config.asteroids.source}")
        written = ast_data.download(config, cache)
        for _name, location in written.items():
            print(f"  -> {location}")
    return 0


def _train(config: Config, cache: Cache, args: argparse.Namespace) -> int:
    import numpy as np

    from .evaluation.reports import build_reports, summary_table
    from .models import registry
    from .models.splits import grouped_holdout
    from .models.train import ASTEROID_TASK, EXOPLANET_TASK, calibration_for, train_task

    tasks = [EXOPLANET_TASK, ASTEROID_TASK] if args.task == "both" else [args.task]
    summaries: list[dict] = []

    for task in tasks:
        model_name = "exoplanet_disposition" if task == EXOPLANET_TASK else "asteroid_pha"
        print(f"\n=== training {task} ===", flush=True)
        result = train_task(task, config, cache, with_importance=not args.no_importance)
        print(f"selected {result.winner!r} on {result.selection_metric_name}", flush=True)

        # The figures need test-fold probabilities, and `train_task` deliberately does not
        # keep them (a TrainingResult holding raw data would end up pickled into the model
        # file). Recompute the same split deterministically and score once more, on the
        # frozen winner - the split is a pure function of the seed and the labels, so this
        # is the identical fold, not a second look at the test set.
        extra: dict = {"settings_path": str(config.root / "configs" / "settings.yaml")}
        probabilities = truth = calibration = None
        if task == ASTEROID_TASK:
            X, y, groups = _matrices(config, cache, task)
            holdout = grouped_holdout(y, groups, config)
            X_test = X.iloc[holdout.test_index]
            y_test = y.iloc[holdout.test_index]
            probabilities = np.asarray(result.estimator.predict_proba(X_test))
            truth = y_test.to_numpy()
            calibration = calibration_for(result, X_test, y_test, result.labels)
            extra["n_approach_events"] = _event_count(config, cache)
            extra["pha_rule_unknowns"] = result.baseline_test_metrics.get("unknown_at_threshold")

        registry.save_artifact(
            config.paths.model_dir,
            model_name,
            result.estimator,
            feature_names=result.feature_names,
            labels=result.labels,
            task=task,
            algorithm=result.algorithm,
            metrics=result.test_metrics,
            dataset_fingerprint=result.dataset_fingerprint,
            rows=result.rows,
            groups=result.groups,
            extra={
                "cv": [candidate.as_row() for candidate in result.candidates],
                "holdout": result.holdout,
                "folds": result.folds,
                "baseline": {
                    "name": result.selected_baseline_name,
                    "metrics": result.baseline_test_metrics,
                },
                "train_metrics": result.train_metrics,
                "feature_ranges": result.feature_ranges,
                **extra,
            },
        )
        index = build_reports(
            result,
            config.paths.report_dir,
            probabilities=probabilities,
            truth=truth,
            calibration=calibration,
            extra=extra,
        )
        print(f"model  -> {config.paths.model_dir / (model_name + '.joblib')}")
        print(f"metrics-> {index['metrics_path']}")
        summaries.append(index)

    print()
    print(summary_table(summaries).to_string(index=False))
    return 0


def _matrices(config: Config, cache: Cache, task: str):
    from .models.train import load_asteroid_matrices, load_exoplanet_matrices

    if task == "asteroids":
        return load_asteroid_matrices(config, cache)
    return load_exoplanet_matrices(config, cache)


def _event_count(config, cache) -> int:
    from .data import asteroids as ast_data

    return len(ast_data.load_events(config, cache))


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api.app import create_app

    app = create_app(args.config)
    print(f"serving on http://{args.host}:{args.port}  (docs at /docs)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = Config.load(args.config)
        config.paths.ensure()
        cache = Cache(config.paths.raw_dir)
    except NasaSpaceMlError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.command == "fetch":
        return _fetch(config, cache, offline=args.offline)
    if args.command == "status":
        return _print_status(config, cache)
    if args.command == "train":
        if args.offline:
            print("--offline: training from the cached snapshot only")
        return _train(config, cache, args)
    if args.command == "all":
        if not args.offline:
            _fetch(config, cache, offline=False)
        return _train(config, cache, args)
    if args.command == "serve":
        return _serve(args)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
