# NASA Space Object Classification Service — Implementation Plan

> **For agentic implementations:** REQUIRED SUB-SKILL: Use superpowers:subagents-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps are atomic and checkbox-driven.

**Source spec:** `docs/superpowers/specs/2026-11-11-nasa-space-ml-design.md`

**Goal:** Ship one FastAPI service that classifies exoplanet candidates (3-class) and near-Earth objects (PHA risk), backed by two real NASA datasets, with tests, CI, Docker, a front-end, and a learning guide.

**Architecture:** A single `src/nasa_space_ml` package with three layers: `data/` (fetch + cache NASA snapshots), `ml/` (features → train → evaluate → registry artifacts), `api/` (FastAPI app that loads artifacts and serves JSON + static front-end). The CLI is the only thing that trains; the API only loads.

**Tech stack:** Python 3.12, numpy, pandas, scikit-learn 1.9, matplotlib, joblib, httpx, fastapi, uvicorn, pydantic v2, pytest, ruff, GitHub Actions, Docker.

---

## Ground rules for every task

1. Every function gets a docstring that explains the **why**, not just the what.
2. No task ends with a failing or skipped test. If a test cannot pass yet, the task is not done.
3. Tests never touch the network. Every HTTP call is behind an injectable `httpx.Client`.
4. Run `uv run ruff check .` and `uv run ruff format .` before each commit.
5. Commit message style: `feat:`, `fix:`, `test:`, `docs:`, `chore:`, `ci:`.

## Verified external facts (do not re-guess these)

| Fact | Value | How verified |
|---|---|---|
| KOI table name | `Q1_Q17_DR25_KOI` | `TAP_SCHEMA.tables` |
| KOI row count | 8,054 | live CSV export, 1,726,634 bytes |
| KOI class counts | FP 3,965 / CONFIRMED 2,731 / CANDIDATE 1,358 | `group by koi_disposition` |
| Inclination column | `koi_incl` (not `koi_inc`) | `TAP_SCHEMA.columns` |
| TAP CSV URL | `https://exoplanetarchive.ipac.caltech.edu/TAP/sync` | HTTP 200 |
| TAP `limit` keyword | **unsupported** — use `select ... from` and slice in pandas | ORA-00933 on `limit 3` |
| CAD endpoint | `https://ssd-api.jpl.nasa.gov/cad.api.json` | HTTP 200 |
| CAD pagination | `limit` + `limit-from` (1-based), **not** `count`/`rows_count` | 400 on `count=200` |
| CAD PHA filter | `pha=1` (boolean filter) | doc + live |
| CAD fields | `des, orbit_id, jd, cd, dist, dist_min, dist_max, v_rel, v_inf, t_sigma_f, h` | live `fields` array |
| CAD window 2020→2031, dist≤0.2 au | 32,358 events / 1,634 PHA events | `total-only=1` |
| CAD version string | `signature.version == "1.5"` | live |
| sklearn | 1.9.0 — `HistGradientBoostingClassifier` accepts `class_weight`; `LogisticRegression` is multinomial by default (`multi_class` param removed); `StratifiedGroupKFold` exists | local probe |
| No API keys | neither NASA source needs one | live |

---

## File map

```
nasa-space-ml/
├── pyproject.toml
├── README.md
├── .gitignore
├── .github/workflows/ci.yml
├── Dockerfile
├── configs/settings.yaml
├── data/raw/.gitkeep                     # gitignored contents
├── models/.gitkeep
├── src/nasa_space_ml/
│   ├── __init__.py
│   ├── config.py                         # yaml -> Config dataclass
│   ├── errors.py                         # NasaMlError hierarchy
│   ├── cli.py                            # fetch / train / evaluate / serve / all
│   ├── data/
│   │   ├── __init__.py
│   │   ├── cache.py                      # sha256 manifest + atomic write
│   │   ├── exoplanet.py                  # TAP query builder + fetch
│   │   └── asteroid.py                   # CAD paged fetch + PHA label fetch
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── features.py                   # both featurizers, pure
│   │   ├── baselines.py                  # majority + NASA PHA rule
│   │   ├── splits.py                     # stratified GROUPED splits
│   │   ├── train.py                      # CV search, pick, refit
│   │   ├── metrics.py                    # macro-F1, AP, PR/ROC, calibration
│   │   ├── registry.py                   # joblib save/load + metadata + version guard
│   │   └── reports.py                    # matplotlib figures -> reports/figures
│   └── api/
│       ├── __init__.py
│       ├── schemas.py                    # pydantic v2 request/response
│       ├── deps.py                       # app state, registry accessor
│       ├── errors.py                     # exception -> JSON problem
│       ├── middleware.py                 # request id + timing
│       ├── app.py                        # create_app(settings)
│       └── routers/
│           ├── __init__.py
│           ├── meta.py                   # /health, /v1/models, /v1/metrics/{task}
│           ├── exoplanet.py              # /v1/exoplanet/predict(+batch)
│           └── asteroid.py               # /v1/asteroid/predict(+batch)
├── web/
│   ├── index.html
│   ├── styles.css
│   └── app.js
└── tests/
    ├── conftest.py                       # synthetic frames, fake transport
    ├── test_config.py
    ├── test_cache.py
    ├── test_data_exoplanet.py
    ├── test_data_拔asteroid.py
    ├── test_features.py
    ├── test_splits.py
    ├── test_baselines.py
    ├── test_metrics.py
    ├── test_train.py
    ├── test_registry.py
    ├── test_api_meta.py
    ├── test_api_exoplanet.py
    ├── test_api_asteroid.py
    └── test_e2e_pipeline.py
```

---

## Task 1: Project scaffold, config, errors

**Files:** `pyproject.toml`, `.gitignore`, `configs/settings.yaml`, `src/nasa_space_ml/{__init__,config,errors}.py`, `tests/{conftest.py,test_config.py}`

**Produces:** `Config.load(path) -> Config`; `NasaMlError`, `DataError`, `ModelNotLoadedError`, `PredictionError`.

- [ ] Write `pyproject.toml` with `src/` layout, deps `numpy pandas scikit-learn matplotlib joblib httpx pyyaml fastapi "uvicorn[standard]"`, dev extras `pytest pytest-cov ruff httpx`, and `[project.scripts] nasa-space-ml = "nasa_space_ml.cli:main"`.
- [ ] Write `.gitignore`: `.venv/ __pycache__/ data/raw/* data/processed/* reports/figures/* !.gitkeep .pytest_cache/ .ruff_cache/ *.pyc` — note `models/` is NOT ignored (artifacts ship).
- [ ] Write `configs/settings.yaml` (paths, CAD window `2020-01-01`→`2031-01-01`, `dist_max_au: 0.2`, `page_limit: 5000`, KOI table, random `seed: 42`, hyperparameter grids).
- [ ] Write failing `tests/test_config.py`: loads yaml, exposes `cfg.koi.table == "Q1_Q17_DR25_KOI"`, `cfg.cad.dist_max_au == 0.2`, unknown key raises `ConfigError`.
- [ ] Implement `config.py` with frozen dataclasses + `ConfigError`.
- [ ] Implement `errors.py`: `NasaMlError(Exception)` base with `.code` and `.http_status`; subclasses `DataError`(502), `ModelNotLoadedError`(503), `PredictionError`(400).
- [ ] Run tests → pass. `ruff check .`. Commit `chore: scaffold package, config and error types`.

## Task 2: Atomic cache with sha256 manifest

**Files:** `src/nasa_space_ml/data/cache.py`, `tests/test_cache.py`

**Produces:** `Cache(dir)`, `.write_bytes(name, data) -> Path`, `.read_bytes(name) -> bytes`, `.manifest() -> dict`, `.verify(name) -> bool`.

- [ ] Failing tests: write then read round-trips; manifest records `sha256` + `bytes` + `utc`; `verify()` false after the file is hand-edited; reading a missing name raises `DataError`; write is atomic (a second write to the same name fully replaces, never appends).
- [ ] Implement using `tempfile.NamedTemporaryFile(dir=..., delete=False)` + `os.replace`, `hashlib.sha256`, `manifest.json` updated on each write.
- [ ] Commit `feat: content-addressed download cache`.

## Task3: Exoplanet TAP client

**Files:** `src/nasa_space_ml/data/exoplanet.py`, `tests/test_data_exoplanet.py`

**Produces:**
- `KOI_COLUMNS: tuple[str, ...]` — 22 export columns incl. `kepoi_name, koi_disposition, hostname, koi_period, koi_duration, koi_depth, koi_prad, koi_impact, koi_eccen, koi_incl, koi_sma, koi_teq, koi_steff, koi_srad, koi_smass, koi_slogg, koi_num_transits, koi_model_snr, koi_max_sngle_ev, koi_kepmag, koi_gmag, koi_time0bk`
- `build_query(columns, table) -> str`
- `fetch_koi(cfg, cache, client) -> pd.DataFrame`

- [ ] Failing tests using `httpx.MockTransport`: query contains no `limit` keyword and no `group by`; CSV parses to a DataFrame with `koi_disposition` present; an HTTP 500 raises `DataError` after ≤3 retries; a cached artifact short-circuits the client (transport call count == 0); the DataFrame is written to cache exactly once.
- [ ] Implement `build_query` (plain `select a,b,c from T`), `fetch_koi` with `client.get(url, params={"query":..., "format":"csv"})`, `df = pd.read_csv(io.StringIO(text))`, retry loop with backoff, cache-first.
- [ ] Commit `feat: NASA Exoplanet Archive TAP client`.

## Task 4: Asteroid CAD client with pagination and PHA labels

**Files:** `src/nasa_space_ml/data/asteroid.py`, `tests/test_data_asteroid.py`

**Produces:**
- `CAD_FIELDS: tuple[str, ...]`
- `fetch_cad_events(cfg, cache, client) -> pd.DataFrame` (all events, paginated)
- `fetch_pha_designations(cfg, cache, client) -> frozenset[str]`
- `load_events(cfg, cache) -> pd.DataFrame`, `load_events_with_labels(...) -> pd.DataFrame`

**Design note to encode in the docstring:** `pha=1` filters *objects*, so PHA status is object-level. We fetch all events once, fetch the PHA event list once, take the distinct `des` values, and label events by membership. Never infer the label from `dist`/`h` inside the fetch layer.

- [ ] Failing tests with MockTransport returning synthetic pages: request 1 uses `limit=5000&limit-from=1`, request 2 uses `limit-from=5001`; stops when `limit-from + len(page) > total`; `total` is read from the payload; `signature.version != "1.5"` raises `DataError` mentioning the version; `pha=1` is present only on the designation query; a `des` appearing in the PHA page yields `pha_flag=True` for **all** of that object's events; `t_sigma_f` strings like `"< 00:01"`, `"13:02"`, `"2_09:08"` all parse to float seconds.
- [ ] Implement, including `parse_t_sigma(seconds) -> float | None`.
- [ ] Commit `feat: JPL CAD pagination and object-level PHA labels`.

## Task 5: Feature engineering (pure, no sklearn yet)

**Files:** `src/nasa_space_ml/ml/features.py`, `tests/test_features.py`

**Produces:**
- `FEATURES_KOI: list[str]`, `LEAK_FORBIDDEN: frozenset[str] = {"koi_pdisposition", "keoi_disposition", "koi_count", "koi_vet_stat"}`
- `featurize_koi(df) -> tuple[pd.DataFrame, pd.Series, pd.Series]` (X, y, host-group)
- `FEATURES_CAD: list[str]`
- `featurize_cad(df) -> tuple[pd.DataFrame, pd.Series, pd.Series]` (object-level, grouped by `des`)
- `assert_no_leakage(cols)`

**Engineered exoplanet features (with the reason in the docstring):**
`log_period`, `log_duration`, `log_depth`, `log_prad`, `depth_ratio = sqrt(depth_ppm*1e-6)`, `rp_over_rs = depth_ratio`, `prad_from_depth_er = rp_over_rs * srad * 9.736` (R☉→R⊕), `prad_inconsistency = |koi_prad − prad_from_depth_er|`, `dur_over_per`, `impact`, `n_transits`, `model_snr`, `stellar_density = smass/srad**3`, `teq`, `slogg`.

**Engineered asteroid features:** per-`des` aggregation → `n_approaches`, `min_dist_au`, `min_dist_min_au` (MOID proxy), `median_v_rel`, `max_v_inf`, `h_mag`, `diameter_km = 1329/sqrt(0.14) * 10**(-h/5)` (standard magnitude-to-diameter relation; document the albedo assumption), `log_diameter`, `dist_over_v`, `season_sin/season_cos` from `jd` month, `mean_t_sigma_s`, `earliest_year`.

- [ ] Failing tests: hand-computed `prad_from_depth_er` for `depth=7900 ppm, srad=0.9` ≈ 7900→`sqrt(7.9e-3)=0.0889`, `×0.9×9.736 ≈ 0.778 R⊕` (assert within 1%); `diameter_km(h=22.193) ≈ 139 m` within 5%; `assert_no_leakage` raises if `koi_pdisposition` is in the list; NaN-heavy synthetic frame still returns all-numeric X of the right width; output index/columns are stable across two calls on the same input; the CAD featurizer returns **one row per `des`** and never mixes PHA labels within a `des`.
- [ ] Implement with `np.errstate` guards and `errors="raise"` on width mismatch.
- [ ] Commit `feat: physics-grounded feature engineering for both datasets`.

## Task 6: Grouped stratified splits + leakage guards

**Files:** `src/nasa_space_ml/ml/splits.py`, `tests/test_splits.py`

**Produces:** `split_indices(X, y, groups, seed) -> Split(train_idx, val_idx, test_idx)`, `cv_iter(X, y, groups, n_splits)`.

- [ ] Failing tests: `train/val/test` are pairwise disjoint; **no group appears in two splits** (the headline test — feed a synthetic frame where `des0000001` appears in 4 rows and assert it lands in exactly one split); class proportions within ±10 points across splits; `cv_iter` never places two rows of the same group in the same fold; the same seed reproduces identical indices.
- [ ] Implement with `GroupShuffleSplit` → `GroupShuffleSplit` chain for the 3-way, `StratifiedGroupKFold(n_splits=5)` for CV. Guard: raise `ValueError` if `n_splits > number of distinct groups`.
- [ ] Commit `test+feat: grouped splits that make leakage impossible`.

## Task 7: Metrics that survive review

**Files:** `src/nasa_space_ml/ml/metrics.py`, `tests/test_metrics.py`

**Produces:** `classification_report(y_true, y_pred, y_proba, labels) -> dict` returning `accuracy, balanced_accuracy, macro_f1, per_class{precision,recall,f1}, confusion_matrix, roc_auc_ovr, average_precision_ovr` and, for binary, `average_precision, roc_auc, brier`.

- [ ] Failing tests on hand-built tiny inputs: macro-F1 of a perfect classifier == 1.0; a random 50/50 binary classifier's ROC-AUC ≈ 0.5 but its **average precision ≈ base rate** — assert the gap is visible (this is the teaching test); Brier of a perfectly confident correct model == 0.0; confusion matrix row sums equal class support; a balanced-3-class perfect-recall-per-class case yields macro-F1 1.0 while accuracy is also 1.0.
- [ ] Implement with sklearn metric calls only; never import matplotlib here.
- [ ] Commit `feat: metric suite incl. PR-AUC and calibration`.

## Task 8: Baselines — majority class and NASA's actual PHA rule

**Files:** `src/nasa_space_ml/ml/baselines.py`, `tests/test_baselines.py`

**Produces:** `MajorityBaseline` (`.fit/.predict/.predict_proba`), `pha_rule(moid_au, h_mag) -> np.ndarray[bool]` implementing MOID ≤ 0.05 au **AND** H ≤ 22.0.

- [ ] Failing tests: majority baseline predicts the single most common label for every row; `pha_rule` truth table over all four corners (0.04/21 → True; 0.04/23 → False; 0.06/21 → False; 0.06/23 → False); **strict** `<=` boundary at exactly 0.05 and 22.0 is True; NaN in either input → False with a documented "unknown is not hazardous" policy.
- [ ] Implement.
- [ ] Commit `feat: majority and NASA PHA rule baselines`.

## Task 9: Model zoo + cross-validated selection

**Files:** `src/nasa_space_ml/ml/train.py`, `tests/test_train.py`

**Produces:** `Candidate` dataclass, `candidates(task) -> list[Candidate]`, `cross_validate(...) -> list[CvResult]`, `train_task(cfg, X, staff, groups, task) -> TrainedModel`.

- [ ] Candidates: `logreg` (SimpleImputer median + StandardScaler + LogisticRegression(max_iter=2000, class_weight="balanced")), `rf` (RandomForest 300 trees, balanced_subsample), `hgb` (HistGradientBoostingClassifier, `class_weight="balanced"`, `early_stopping=False`, `max_iter=300`), plus baselines. **Do not pass `multinomial` to anything** — it was removed from sklearn 1.7+.
- [ ] Failing tests on synthetic data: `train_task` returns a fitted estimator whose `.predict` runs on the input columns; the selected candidate is the one with the best CV score for the task's headline metric (`macro_f1` for exoplanet, `average_precision` for asteroid); every candidate's metrics are retained in the result, including losers; a `RandomizedSearchCV`-free implementation (fixed grids only, deterministic under `seed`).
- [ ] Implement: `StratifiedGroupKFold` CV loop, `np.nan`-safe scoring, refit winner on train+val, evaluate once on the untouched test split.
- [ ] Commit `feat: cross-validated model selection with grouped folds`.

## Task 10: Registry — artifacts that refuse to load wrongly

**Files:** `src/nasa_space_ml/ml/registry.py`, `tests/test_registry.py`, `models/.gitkeep`

**Produces:** `save_model(dir, task, model, *, feature_names, labels, metrics, dataset_sha) -> Path`, `load_model(dir, task) -> LoadedModel(model, feature_names, labels, metrics, meta)`, `Registry.load(task)`, `.status()`.

**Metadata written to `models/<task>.json`:** `schema_version, task, sklearn_version, feature_names, labels, n_train_rows, trained_at_utc, dataset_sha256, metrics`.

- [ ] Failing tests: save→load round-trips predictions bit-for-bit; `load` raises `ModelNotLoadedError` when the file is absent, with a message containing the exact CLI command that fixes it (`python -m nasa_space_ml train`); `load` raises `ModelNotLoadedError` when the artifact's `feature_names` differ from the currently-declared list (train/serve skew guard); a mismatched `sklearn_version` raises with the two versions printed.
- [ ] Implement with `joblib.dump(compress=3)`.
- [ ] Commit `feat: versioned model registry with skew guard`.

## Task 11: Reports and figures

**Files:** `src/nasa_space_ml/ml/reports.py`, `tests/test_reports.py`

**Produces:** `write_metrics_json(dir, payload) -> Path`, `plot_confusion(matrix, labels, path)`, `plot_pr_curve(y_true, proba, path)`, `plot_feature_importance(model, names, path)`.

- [ ] Failing tests: `matplotlib.use("Agg")` is set **before** any pyplot import; each plot writes a non-empty `.png`; `metrics.json` is valid JSON and round-trips; a 3-class and a 2-class confusion matrix both render without raising; `plot_feature_importance` handles `HistGradientBoosting` via `permutation_importance` (HGB has no `feature_importances_`).
- [ ] Implement; figures land in `reports/figures/`.
- [ ] Commit `feat: metric json and figure generation`.

## Task 12: CLI pipeline

**Files:** `src/nasa_space_ml/cli.py`, `src/nasa_space_ml/__main__.py`, `tests/test_cli.py`

**Produces:** argparse commands `fetch`, `train`, `all`, `serve`, `--config`, `--offline`.

- [ ] Failing tests: `--offline` makes `fetch` skip and `train` raise `DataError` naming the missing cached artifact; `all` runs fetch→train→report in order and writes both `models/*.json` (assert on a monkeypatched fetcher producing synthetic frames); `serve` builds the command list `["uvicorn", "nasa_space_ml.api.app:app", ...]` without actually starting it (test with `--dry-run`).
- [ ] Implement with `main(argv=None) -> int` returning exit codes; no `sys.exit` inside helpers.
- [ ] Commit `feat: cli pipeline (fetch/train/all/serve)`.

## Task 13: API schemas and problem-json errors

**Files:** `src/nasa_space_ml/api/schemas.py`, `src/nasa_space_ml/api/errors.py`, `tests/test_api_schemas.py`

**Produces:** ExoplanetRequest` / `AsteroidRequest` (physical units in field names, `Field(ge=, le=, description=, examples=[...])`), `PredictionResponse`, `ModelInfo`, `HealthResponse`; `register_error_handlers(app)`.

- [ ] Failing tests: a request with `koi_period_days=0` → pydantic validation error listing that field; an unknown extra field → `ValidationError` because `model_config = ConfigDict(extra="forbid")`; the request model exposes `to_frame()` producing exactly the artifact's `feature_names` order; `PredictionError` maps to HTTP 400 with `{"error":{"code","message","request_id"}}`; an unhandled exception maps to 500 **without** leaking the exception text.
- [ ] Implement. Bounds come from the data dictionary: `koi_prad_er` 0–100 R⊕, `h_mag` 5–35, `moid_au` 0–5.
- [ ] Commit `feat: pydantic request contracts and error envelope`.

## Task 14: API app, middleware, meta routes

**Files:** `src/nasa_space_ml/api/deps.py`, `api/middleware.py`, `api/app.py`, `api/routers/meta.py`, `tests/test_api_meta.py`

**Produces:** `create_app(cfg, registry=None) -> FastAPI`, `Registry` stored on `app.state`, `X-Request-Id` middleware (echo or generate), `/health`, `/v1/models`, `/v1/metrics/{task}`, `/` → `web/`.

- [ ] Failing tests: with an empty registry `/health` returns `status="degraded"`, `models_loaded=false` and **HTTP 200** (the process is alive); `/v1/exoplanet/predict` returns **503** with `code="model_not_loaded"`; `/v1/models` lists both tasks with metrics after fixtures are registered; every response carries `x-request-id`; an inbound `x-request-id` is echoed unchanged; static files are mounted so `/index.html` is HTTP 200.
- [ ] Implement; register routers with `prefix="/v1"`.
- [ ] Commit `feat: app factory, request-id middleware, meta routes`.

## Task 15: Prediction routers

**Files:** `api/routers/exoplanet.py`, `api/routers/asteroid.py`, `tests/test_api_exoplanet.py`, `tests/test_api_asteroid.py`

**Produces:** `POST /v1/exoplanet/predict`, `POST /v1/exoplanet/predict/batch` (cap 500), `GET /v1/exoplanet/sample?n=1`, and the asteroid equivalents.

- [ ] Failing tests against a toy model registered in `app.state`: single predict returns `predicted_label` + all class probabilities summing to 1.0 (±1e-6) + `model_version` + `elapsed_ms`; batch of 3 returns 3 results in order; batch of 501 → 422; `sample` returns real rows that validate against the request model (this is the round-trip guard that keeps the front-end honest); an out-of-distribution input is still scored but flagged in `warnings`.
- [ ] Implement. The handler resolves the model through `deps`, calls `model.predict_proba`, and never touches the network or the filesystem.
- [ ] Commit `feat: prediction endpoints for both models`.

## Task 16: Front-end

**Files:** `web/index.html`, `web/styles.css`, `web/app.js`

**Produces:** a static page consumed by the mounted route. No bundler, no CDN.

- [ ] Build the page: header with service status from `/health`; two tabs; each tab renders its form from `GET /v1/{task}/sample` prefilled with real NASA values and an "edit" mode; submit → `POST`; result panel shows the label, per-class probability bars, and the metrics block fetched from `/v1/metrics/{task}`; a "why" panel listing that model's top permutation importances.
- [ ] Styling: instrument panel — dark neutral background, monospace numeric readouts, thin rules, dense two-column layout, no gradient hero, no card shadows. Use `system-ui` + `ui-monospace`.
- [ ] JS: `fetch` with an `AbortController`, one `renderError(problem)` that prints `error.code` + `message`, no `innerHTML` for NASA-sourced strings (use `textContent` — XSS hygiene).
- [ ] Test: `tests/test_web_assets.py` asserts the three files exist, the HTML references both endpoint paths it actually calls, and no external `http://` origin is referenced.
- [ ] Commit `feat: static instrument front-end`.

## Task 17: End-to-end pipeline test on synthetic data

**Files:** `tests/test_e2e_pipeline.py`

- [ ] One test that: synthesizes ~300 KOI rows and ~300 CAD events with a known planted signal (deterministic seed), runs featurize → split → train → registry save → `create_app` → POST a row derived from a real training row, and asserts the response is 200 and the label is in the declared label set. This is the test that proves the whole stack is wired.
- [ ] Commit `test: end-to-end pipeline contract`.

## Task 18: CI, Docker, README, learning guide

**Files:** `.github/workflows/ci.yml`, `Dockerfile`, `README.md`, `docs/LEARNING_GUIDE.md`, `docs/DATA_DICTIONARY.md`, `docs/MODEL_CARD_exoplanet.md`, `docs/MODEL_CARD_asteroid.md`

- [ ] CI: `uv sync` → `ruff check` → `pytest --cov` on `python 3.12` (matrix 3.11/3.12 if runtime allows); `--offline`-safe so CI never calls NASA.
- [ ] Docker: multi-stage, `uv` builder + slim runtime, `HEALTHCHECK` hitting `/health`, non-root user, `EXPOSE 8000`.
- [ ] README: one-paragraph what/why, quickstart (3 commands), the metrics table copied from the real `reports/metrics.json`, an architecture diagram in ASCII, and an explicit "known limitations" section.
- [ ] `docs/LEARNING_GUIDE.md`: the 15 chapters listed in §11 of the spec, each with a "why this matters" opener and a pointer to the exact file:line that implements it. This is the user-facing teach-me-everything deliverable.
- [ ] Commit `docs: readme, learning guide, model cards` + `ci: workflows and container`.

## Task 19: Real data, real training, publish

**Files:** `data/raw/*`, `models/*`, `reports/*`

- [ ] `uv run nasa-space-ml fetch` against live NASA endpoints; verify row counts match the table above (KOI 8,054; CAD ≈32,358 events; PHA ≈1,634). If counts are off, stop and re-check the endpoint, don't adjust the test.
- [ ] `uv run nasa-space-ml train`; capture the real metrics into `reports/metrics.json`.
- [ ] Write the results honestly into the README and model cards — including the rule-vs-ML comparison and whichever one actually won.
- [ ] `uv run pytest -q` and `uv run ruff check .` green locally.
- [ ] `git add -A && git commit`, then `git checkout -b main && git add <deliverables>`.
- [ ] Publish: `gh repo create nasa-space-ml --public --source=. --remote=origin --push` under `Yuvraj0967`. Confirm with the user before this step.
- [ ] Verify live: `gh repo view --json url,name,isPrivate`, then `curl -s https://api.github.com/repos/Yuvraj0967/nasa-space-ml | jq '.default_branch, .size'`.

---

## Plan self-review

**Spec coverage:** §2 data → tasks 3,4. §3 layout → task 1. §4 features → task 5. §5 models → tasks 6–10. §6 API → tasks 13–15. §7 front-end → task 16. §8 error handling → tasks 2,4,10,13,14. §9 tests → every task + 17. §10 publishing → task 19. §11 learning guide → task 18. No gaps.

**Deliberate deviations from the spec, and why:**
1. The spec said data snapshots ship in the repo. **Changed:** raw NASA snapshots are gitignored. Rationale — 1.7 MB CSV + ~15 MB CAD is a permanent repo tax, reviewers can regenerate with one command, and it makes CI deterministic. The *trained models and metrics* do ship, so the numbers stay verifiable without any download.
2. The spec's `20/10/70` split became a grouped `train/val/test` chain in task 6. Same intent, cleaner sklearn API.
3. Added: the `sample?n=1` endpoints (task 15) are load-bearing for the front-end's "always show a real NASA row" honesty rule in §7, which the spec required but never assigned to a task.

**Naming consistency check:** `DataError`/`ModelNotLoadedError`/`PredictionError` are defined in task 1 and referenced by tasks 10, 12, 13, 14. `featurize_koi`/`featurize_cad` (task 5) are consumed by tasks 9, 12, 17. `pha_rule` (task 8) is consumed by tasks 9, 15. `create_app(cfg, registry)` (task 14) is consumed by tasks 12, 16, 17. `Registry.load/status` (task 10) is consumed by tasks 14, 15.

**Known risk:** the CAD volume (≈32,000 events → ≈8k objects) makes per-`des` aggregation the slow step, and `HistGradientBoosting` + `RandomForest` on 32k rows is seconds, but `permutation_importance` on 30k rows is not. Task 11 caps it with `n_repeats=10` and `random_state` to keep `train` under a couple of minutes.
