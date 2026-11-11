# NASA Space Objects — Design Specification

Date: 2026-11-11
Status: approved by owner (design presented in-session; owner said "no [changes], finish up")
Repo target: https://github.com/Yuvraj0967/nasa-space-ml

## 1. Goal

One repository containing two complete, independently reproducible machine-learning
projects built on live NASA data, wrapped in a production-grade FastAPI service with an
HTML/JS front-end, tests, CI, a container, and an in-repo learning guide that teaches
every concept the code depends on.

The owner is presenting this as a capstone. Honesty about limitations is a feature, not a
weakness: every reported number must be reproducible from the committed artifacts.

## 2. Datasets (verified live on 2026-11-11)

### 2.1 Exoplanets — NASA Exoplanet Archive KOI table

- Source: `https://exoplanetarchive.ipac.caltech.edu/TAP/sync`, TAP/ADQL, `format=csv`, no API key.
- Table: `Q1_Q17_DR25_KOI` (Kepler Objects of Interest, DR25), 153 columns.
- Verified label distribution on `koi_disposition`:
  FALSE POSITIVE 3,965 / CONFIRMED 2,731 / CANDIDATE 1,358 — 8,054 rows total.
- Verified missingness on the selected columns: 503 empty `koi_zmag`, 132 empty `koi_imag`,
  59 empty on transit parameters. Missingness is therefore modest and mostly column-local.
- Task: 3-class classification of a Kepler candidate's validation status.
- Query is written with `select <cols> from Q1_Q17_DR25_KOI` (plain, no `limit`, no
  `group by` in the download path — `limit` is rejected by this TAP endpoint; `where rownum <= n`
  is the working row cap and is used only by tests/samples).

### 2.2 Asteroids — JPL SBDB Close Approach Data API

- Source: `https://ssd-api.jpl.nasa.gov/cad.api.json`, version 1.5, no API key.
- Verified parameters (from the official doc page and live probes):
  `date-min`, `date-max`, `dist-max`, `pha` (boolean filter), `diameter` (bool, adds
  `diameter`/`diameter_sigma`), `fullname` (bool), `sort`, `limit` (max verified 5,000/request),
  `limit-from` (paging, requires `limit`), `total-only` (bool).
  `count` and `rows_count` are NOT valid and return HTTP 400.
- Verified record counts for 2020-01-01 → 2031-01-01, `dist-max=0.2`:
  all close approaches **32,358**; with `pha=1` → **1,634**. Positive rate ≈ 5.0%.
- Returned fields: `des, orbit_id, jd, cd, dist, dist_min, dist_max, v_rel, v_inf,
  t_sigma_f, h` plus optional `diameter, diameter_sigma, fullname`.
- Task: binary classification — is this object a Potentially Hazardous Object.

### 2.3 Design consequence: labels are object-level, not event-level

`pha=1` filters objects, and PHA status is a property of an object's orbit, while CAD returns
one row per close-approach *event*. Multiple events share one `des`. Therefore the asteroid
unit of analysis is the **object (`des`)**, built by aggregating its events. A naive event-level
split would place the same asteroid in train and test and inflate every metric. The same trap
exists on the exoplanet side as multiple planets per host star; both are handled in §5.

Neither dataset needs a NASA API key, so the repository contains no secrets.

## 3. Architecture

Option 1 approved: one FastAPI service hosting both models, with a static front-end served by
the same process. src-layout package so the CLI, API and tests import one namespace.

```
nasa-space-ml/
├── pyproject.toml
├── README.md
├── LICENSE                          # MIT
├── .gitignore
├── .github/workflows/ci.yml
├── Dockerfile
├── configs/default.toml             # paths, date windows, feature lists, hyperparams
├── data/
│   ├── raw/                         # NASA snapshots + checksum.json  (gitignored)
│   └── processed/                   # pipeline output                  (gitignored)
├── models/                          # trained artifacts + metadata     (committed)
├── reports/
│   ├── metrics.json                 # committed as evidence
│   └── figures/*.png                # committed
├── src/nasa_space_ml/
│   ├── __init__.py
│   ├── __main__.py                  # python -m nasa_space_ml <cmd>
│   ├── cli.py                       # download | features | train | evaluate | all
│   ├── config.py                    # typed Settings, loads configs/default.toml
│   ├── logging_setup.py
│   ├── exceptions.py                # domain exception hierarchy
│   ├── data/
│   │   ├── exoplanet_client.py      # TAP CSV download
│   │   ├── asteroid_client.py       # CAD paged JSON download, label pass
│   │   ├── checksums.py             # sha256 manifest
│   │   └── snapshot.py              # load raw snapshot into DataFrame
│   ├── features/
│   │   ├── common.py                # impute/scale builders, log transforms
│   │   ├── exoplanet.py             # KOI feature engineering
│   │   └── asteroid.py              # event→object aggregation, physics features
│   ├── models/
│   │   ├── exoplanet.py             # baseline + logreg + RF + HistGB
│   │   ├── asteroid.py              # rule baseline + logreg + HistGB
│   │   ├── baselines.py             # dummy, NASA PHA rule
│   │   ├── registry.py              # save/load, metadata, version guard
│   │   └── selection.py             # CV comparison, pick winner
│   ├── evaluation/
│   │   ├── metrics.py               # macro-F1, PR-AUC, Brier, calibration
│   │   ├── splits.py                # grouped stratified splits
│   │   └── plots.py                 # confusion, PR curve, importances
│   └── api/
│       ├── main.py                  # create_app(), lifespan, static mount
│       ├── deps.py                  # app-state accessors
│       ├── schemas.py               # pydantic v2 request/response
│       ├── errors.py                # exception handlers → JSON envelope
│       ├── middleware.py            # request id, timing, access log
│       └── routers/
│           ├── meta.py              # /api/v1/health, /models, /version
│           ├── exoplanets.py        # /predict, /predict/batch, /schema, /samples
│           └── asteroids.py
├── web/
│   ├── index.html
│   ├── app.js                       # fetch, schema-driven forms, results
│   └── styles.css                   # instrument-panel styling
├── tests/
│   ├── conftest.py                  # tiny fixture data, no network
│   ├── fixtures/*.csv|*.json
│   ├── unit/test_features_*.py
│   ├── unit/test_baselines.py
│   ├── unit/test_clients.py         # mocked transport
│   ├── integration/test_api_*.py    # TestClient
│   ├── integration/test_pipeline.py
│   └── regression/test_golden_metrics.py
└── docs/
    ├── LEARNING_GUIDE.md            # the teaching deliverable
    ├── DATA_DICTIONARY.md
    ├── MODEL_CARD_exoplanets.md
    ├── MODEL_CARD_asteroids.md
    ├── ARCHITECTURE.md
    └── superpowers/specs/*.md       # this file
```

### 3.1 Isolation contracts

Every unit answers "what does it do / how do I call it / what does it depend on".

- `data/*` — only module that touches the network. Depends: `httpx`, `config`. Output:
  `data/raw/<name>.csv|jsonl` plus `checksum.json`. No pandas transforms live here.
- `features/*` — pure functions `DataFrame -> DataFrame`. No I/O, no model knowledge.
  Deterministic and unit-testable row-for-row.
- `models/*` — consumes the feature-frame contract, produces in-memory estimators. No I/O.
- `models/registry.py` — the only place that reads/writes `models/` and `reports/`.
  Contract: `save(name, bundle, metadata)` / `load(name) -> Bundle` and it refuses to load an
  artifact whose recorded feature list disagrees with `configs/default.toml`.
- `api/*` — depends only on `registry` + `schemas` + `config`. Never imports training code.
- `web/*` — depends only on the public HTTP contract, documented in `docs/ARCHITECTURE.md`.
- `cli.py` — the only orchestrator; each stage is separately runnable and idempotent.

A file that grows past roughly 250 lines is split; feature engineering for one dataset never
lives in the other dataset's module.

## 4. Feature engineering

### 4.1 Exoplanets

Numeric base columns (all verified present):
`koi_period, koi_duration, koi_depth, koi_prad, koi_impact, koi_eccen, koi_sma, koi_incl,
koi_teq, koi_steff, koi_srad, koi_smass, koi_slogg, koi_num_transits, koi_model_snr,
koi_max_sngle_ev, koi_kepmag`.

Engineered, with the physical reason each exists:

- `depth_sqrt = sqrt(koi_depth)` and `radius_ratio = 1e-3 * depth_sqrt` — transit depth is
  `(Rp/Rs)^2`, so this recovers the planet-to-star radius ratio directly.
- `prad_implied = radius_ratio * koi_srad` — the planet radius the *depth* implies, in stellar
  radii units. `prad_mismatch = |koi_prad − prad_implied|` is the single most valuable feature:
  blends and eclipsing binaries systematically report a radius that does not match their depth.
- `log_period`, `log_depth`, `log_duration`, `log_prad` — these quantities span orders of
  magnitude; log makes them near-linear and keeps a tree model from spending splits on scale.
- `duration_over_period` — grazing geometries and long-period binaries separate here.
- `stellar_density_proxy` from `koi_smass / koi_srad^3` — used by the validation community to
  reject false positives.
- `n_koi_in_system` (from `koi_count`) — kept **and explicitly discussed**: multi-Kepler systems
  were statistically easier to validate, so this encodes a discovery-pipeline advantage rather
  than physics. Including it is a confound the learning guide calls out by name.

Dropped: `koi_pdisposition`, `koi_vet_stat`, `koi_quarters`, `koi_delivname`, error bars
(`*_err1/_err2` are used only to flag unreliable rows, not as predictors), all magnitude
columns except `koi_kepmag`, and every `*_str`/`*_reflink` display field.

**Leakage rule (hard requirement, tested):** no column derived from the disposition strings may
appear in any training matrix. `koi_pdisposition` is a near-copy of the label and would yield
~0.99 accuracy. A unit test asserts the intersection of the feature list with a forbidden-name
set is empty.

### 4.2 Asteroids

Step 1 — event aggregation by `des`:
`n_approaches`, `min_dist_au`, `mean_dist_au`, `min_dist_min_au` (closest 3σ lower bound),
`max_dist_max_au`, `mean_v_rel`, `max_v_rel`, `mean_v_inf`, `min_v_inf`, `jd_first`,
`jd_last`, `timespan_days`, `year_first`, `month_first`.

Step 2 — physics features:
- `moid_proxy_au = min_dist_min_au`, with the guide stating plainly that true MOID is an
  orbital element JPL computes from the full ephemeris, and this is a *proxy* computed from
  published approach windows. The gap between proxy and truth is the reason the model is
  allowed to beat the rule.
- `diameter_est_km = 1329 / sqrt(p_v) * 10^(-h/5)` with geometric albedo `p_v = 0.14` — the
  standard Harris–Tompkins relation, implemented explicitly rather than trusting the sparse
  optional `diameter` field. Taught as "derive, don't borrow".
- `t_sigma_hours` — parsed from the `d_hh:mm` / `hh:mm` string format of `t_sigma_f`.
- `log10_dist_au`, `log10_diameter_m` — scale compression.
- `h` and `pha_threshold_margin_h = h − 22.0` — the object's signed distance from NASA's
  brightness cut, so the model can express "how close am I to the rule boundary".
- `moon_distances_min = min_dist_au * 6460000 / 384400` approach distance in lunar distances —
  chosen because it is the unit the public and the press actually use.

## 5. Modelling

### 5.1 Splits

- 70/15/15 train/validation/test, **grouped** so no group repeats across sets:
  exoplanets grouped by `hostname`, asteroids by `des`. Implemented with a GroupShuffleSplit
  chain, then `StratifiedGroupKFold(n_splits=5)` inside the training set for model comparison.
- Grouping is applied before any scaling/imputation: the fitted preprocessor exists only on
  train and is `transform`-ed onto val/test. A test asserts val/test rows never influenced
  fitted statistics.

### 5.2 Candidate models

Exoplanets:
1. `DummyClassifier(strategy="most_frequent")` — the floor.
2. Logistic regression, multinomial: median impute + `StandardScaler`, `class_weight="balanced"`.
3. `RandomForestClassifier` (200 trees, `class_weight="balanced_subsample"`).
4. `HistGradientBoostingClassifier` (native NaN handling; the `class_weight` option is set to
   `"balanced"` directly - verified present in the installed scikit-learn 1.9, so no
   manual sample-weight arithmetic is needed), with early stopping on the validation set.

Asteroids:
1. `DummyClassifier` — floor.
2. **NASA PHA rule baseline**: `moid_proxy_au <= 0.05 AND h <= 22.0`. Implemented in
   `models/baselines.py` as a pure function and scored with the identical metric code.
3. Logistic regression with the same preprocessing.
4. `HistGradientBoostingClassifier` with `class_weight="balanced"`.

Selection metric: macro-F1 for exoplanets, average precision (PR-AUC) for asteroids. Winner is
refit on train+val and evaluated once on the untouched test set.

### 5.3 Expected numbers and the honesty rule

Targets, recorded as expectations rather than promises:
- exoplanets: dummy ≈ 0.44 macro-F1-ish floor; HistGB expected 0.70–0.85 macro-F1; logistic
  regression 0.55–0.70.
- asteroids: the PHA rule is expected to score very high because the label is defined by a
  rule. Average precision for the learned model is expected to be 0.90+ and only *marginally*
  above the rule. If the learned model does not beat the rule, that result is reported as the
  result. The write-up then pivots to the genuinely interesting output: the disagreement set —
  objects the rule calls hazardous that the model rejects, and the reverse.

`reports/metrics.json` records every number for every candidate, including the ones that lost.
The README shows all candidates, not the winner alone.

### 5.4 Calibration and probability quality

Because hazard assessment is about probabilities, not just labels, the winning asteroid model
gets a calibration curve, Brier score, and reliability plot. A near-saturated model that is
badly calibrated is a reportable defect, so this is computed rather than assumed.

## 6. API design

- `create_app(settings)` factory; `lifespan` loads both bundles into `app.state`. If artifacts
  are missing the app starts but `/api/v1/health` reports `models_loaded: false` and prediction
  endpoints return 503 with a message telling the operator to run `python -m nasa_space_ml train`.
  Refusing to start entirely would make the learning story harder to demo.
- Versioned prefix `/api/v1`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | front-end (StaticFiles) |
| GET | `/api/v1/health` | liveness, model load state, artifact versions |
| GET | `/api/v1/models` | metadata + metrics for both bundles |
| GET | `/api/v1/exoplanets/schema` | feature list, units, bounds — drives the form |
| POST | `/api/v1/exoplanets/predict` | one row → class + per-class probabilities |
| POST | `/api/v1/exoplanets/predict/batch` | up to 500 rows |
| GET | `/api/v1/exoplanets/samples?n=3` | real held-out rows for the demo |
| GET/POST | `/api/v1/asteroids/*` | mirrored set for asteroids |

- Pydantic v2 schemas carry units in field names and `ge`/`le` bounds from the physics
  (period 0.01–1e6 days, depth 0.1–1e6 ppm, H 8–35 mag, distance 0–0.5 au). Out-of-range input
  is a 422 with `details` naming the field and the acceptable interval. This is contract-first
  design and it is taught as such.
- Response payload includes `predicted_label`, `probabilities`, `model_version`, `explanation`
  (top-5 feature contributions from the fitted booster where available, else the standardized
  deviation from the training median), and `elapsed_ms`.
- Errors: one envelope everywhere — `{"error": {"code", "message", "details", "request_id"}}`.
  Domain exceptions (`DataUnavailable`, `ModelNotLoaded`, `InvalidFeatures`, `UpstreamNASAError`)
  map to 503/422/502; unexpected exceptions map to 500 and are logged with traceback while the
  response body stays generic. No internal message ever leaks.
- Middleware: monotonic request id propagated into logs and the error envelope; access log with
  path, status, duration.
- `/docs` and `/openapi.json` stay enabled — they are a demo surface, not an accident.

## 7. Front-end

Single page, two tabs, vanilla JS, no bundler and no CDN dependency (works offline once the API
is up). Forms are rendered from `/schema`, so the UI cannot drift from the model contract.

Visual direction follows the owner's standing preference: a credible field-instrument / mission
console, not a generic AI landing page. Concretely — dark low-saturation panel, monospaced
numeric readouts, hairline rules, tabular data, signal colours limited to two states, no
gradient hero, no rounded-card bento, real labels ("MOID proxy (AU)") instead of marketing copy.
Every number on screen is either a dataset fact, a model output, or a metric that exists in
`reports/metrics.json`.

Each tab carries: a "load real example" button backed by `/samples`, the probability readout,
the explanation block, and a metrics panel that renders the committed figures from `reports/`.

## 8. Error handling outside the API

- HTTP clients: explicit timeouts, bounded retry with exponential backoff and jitter on 429/5xx
  and connection errors, `raise_for_status()` translated into `UpstreamNASAError`, and a
  `User-Agent` identifying the project. Paging loops terminate on a documented condition
  (`limit-from + limit >= total`) and refuse to exceed a configured maximum request count so a
  broken page cannot spin forever.
- CAD responses are validated against `signature.version` — if the API version changes under us,
  we stop loudly instead of silently mis-parsing columns.
- Raw downloads are sha256-summed into `data/raw/checksum.json`. Later stages verify the file
  they read is the file that was downloaded.
- `ValueError` with the offending column name on any NaN/inf that survives into a training
  matrix — never a silent imputation.

## 9. Testing

- Unit: the physical derivations (radius ratio from depth, diameter from H, t_sigma parsing)
  are asserted against hand-computed values — these are the parts most likely to be silently
  wrong.
- Leakage guard test: forbidden-name intersection must be empty.
- Split integrity test: no `des` and no `hostname` appears in more than one split.
- Client tests with mocked transport: paging math, retry-on-429, version-mismatch abort.
- Integration: FastAPI `TestClient` against a small model trained in-session from fixtures
  (~200 rows), covering 200/422/503 paths and schema/batch endpoints.
- Regression: `test_golden_metrics.py` compares retrained metrics to `reports/metrics.json`
  within tolerance and fails on a >2-point drop. This is the "fitness function" that keeps the
  README honest six months from now.
- The suite never touches the network. CI is hermetic; `download` is opt-in locally.
- Lint/format: `ruff check` + `ruff format --check`. CI matrix: Python 3.11 and 3.12, Ubuntu.

## 10. Delivery

1. Local git repo in the session working directory, first commit is this spec.
2. `gh repo create Yuvraj0967/nasa-space-ml --public --source=. --push`.
3. Committed: source, tests, configs, CI, Dockerfile, docs, `models/` artifacts and
   `reports/` (metrics + figures). Ignored: `data/raw`, `data/processed`, `.venv`, caches.
   Repo stays small (~a few MB) while remaining reproducible end-to-end via `download`.
4. README quickstart must work from a clean clone:
   `uv venv && uv pip install -e ".[dev]" && python -m nasa_space_ml all && uvicorn ...`.

## 11. Teaching layer (equal in priority to the code)

`docs/LEARNING_GUIDE.md` is written as chapters, each tied to code that actually exists in this
repo, each ending with "try this / what you should see":

1. What NASA publishes and why no API key is needed; TAP/ADQL vs a REST paging API.
2. Reading a data dictionary: what `koi_depth` really measures, what `H` really means.
3. Downloading responsibly: checksums, rate limits, version signatures, caching.
4. Data cleaning and the meaning of an empty cell in a survey catalogue.
5. Class imbalance, and why accuracy is the wrong first question.
6. Leakage: the `koi_pdisposition` trap, and how it was caught by a test.
7. Grouped splitting: multiple planets per star, multiple events per asteroid.
8. Feature engineering from physics, not from a recipe book.
9. From logistic regression to gradient boosting: why each step helps.
10. Metrics that survive review: macro-F1, PR-AUC vs ROC-AUC, calibration, Brier.
11. Model cards and artifact metadata: reproducibility as an engineering object.
12. Serving a model: pydantic contracts, error envelopes, lifespan, versioned routes.
13. Front-end without a framework: schema-driven forms and honest numbers.
14. Tests, CI, containers, and how you would deploy this for real.
15. Limitations and what you would do next (MOID truth, drift, re-validation cycles).

Explanations also happen in chat as each stage is built, per the owner's choice.

## 12. Out of scope

No GPU/deep learning, no light-curve pixel data, no database, no auth, no real cloud deploy,
no React/Vue build chain, no third-party model hub. The API is a single process with in-memory
model state; the front-end calls it directly. These are named so the owner can defend the
boundary when asked.

## 13. Definition of done

- `python -m nasa_space_ml all` reproduces raw → features → models → reports from a clean clone.
- `pytest` green with no network access; `ruff check .` clean; CI green on the pushed repo.
- `uvicorn nasa_space_ml.api.main:app` serves both prediction endpoints and the front-end;
  `/docs` renders.
- Every metric in the README traces to `reports/metrics.json`.
- Repo is public at `Yuvraj0967/nasa-space-ml` and loads without a 404.
- Learning guide has all 15 chapters written against real code paths.
