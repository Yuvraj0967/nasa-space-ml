# Architecture

This document describes the module boundaries, the request lifecycle, and the reasons the
seams are where they are. For the concepts themselves, see
[LEARNING_GUIDE.md](LEARNING_GUIDE.md); for column-level detail, see
[DATA_DICTIONARY.md](DATA_DICTIONARY.md).

## The one diagram that matters

```
        ┌────────────── NASA endpoints (no keys, public) ─────────────┐
        │  exoplanetarchive.ipac.caltech.edu/TAP/sync   (ADQL → CSV)  │
        │  ssd-api.jpl.nasa.gov/cad.api.json            (JSON, paged) │
        └───────────────┬─────────────────────────────────────────────┘
                        │  data/http.py      retries, backoff, timeout, UA
                        ▼
                   data/exoplanets.py  ·  data/asteroids.py     ← parse + validate
                        │
                        ▼
                   data/cache.py       content-addressed snapshot  →  data/raw/
                        │                                          + checksum.json
                        ▼
                   features/{exoplanets,asteroids}.py   physics features, leakage blacklist
                        │                               FEATURE_COLUMNS = the contract
                        ▼
                   models/splits.py    StratifiedGroupKFold + greedy group-disjoint holdout
                        │
                        ▼
                   models/train.py     candidate table → CV → select → refit → holdout score
                        │                                        → reports/*/metrics.json
                        ▼
                   models/registry.py  models/<name>.joblib + models/<name>.json (sidecar)
                        │
        ┌───────────────┴───────────────────┐
        ▼                                   ▼
   api/deps.py  AppState             cli.py (fetch | train | status | serve | all)
        │        feature-contract guard
        ▼        asteroid table memo
   api/routers/{exoplanets,asteroids,meta}.py
        │        api/schemas.py (Pydantic bounds, extra="forbid")
        ▼        api/problem.py (RFC 9457) · api/middleware.py (X-Request-Id)
   web/index.html + app.js + styles.css      no framework, no CDN, no external origin
```

There is exactly one direction of dependency: downward toward the data layer. `features/`
does not know `models/` exists; `models/` does not know `api/` exists; nothing in the
package knows `web/` exists. The only module that reaches across the boundary is
`api/deps.py`, which is where that is allowed and is therefore where the interesting
validation lives.

## Layer rules

| Layer | May import | May never |
| --- | --- | --- |
| `data/` | `config`, `errors`, `http`, `cache` | know what a feature is |
| `features/` | `config`, `errors`, `data` | know what a model is |
| `models/` | `config`, `features`, `data`, `evaluation` | know what HTTP is |
| `evaluation/` | `models`, `features` | mutate anything |
| `api/` | everything | hold state that outlives the process |
| `cli.py` | everything | be the only way in (every command is also a function) |

Two rules that are not visible in the import graph but are load-bearing:

**A number is only allowed to exist in one place.** Thresholds, URLs, label orders,
column lists, and the random seed come from `configs/settings.yaml`. `features/*.py` exports
`FEATURE_COLUMNS`, and both `models/train.py` and `api/deps.py` read that same tuple —
which is the structural reason train/serve skew has to be *introduced* by a developer
rather than occurring by drift.

**Nothing about a published score may depend on a file that is not in git.** `models/` and
`reports/` are committed; `data/raw/` is not. The sidecar carries the dataset fingerprint,
so the shipped artifacts describe bytes the reader can regenerate and compare, without the
repo redistributing NASA's tables.

## Configuration

`config.py` loads YAML into frozen dataclasses — `Config`, `PathsConfig`,
`ExoplanetSettings`, `AsteroidSettings`, `SplitSettings` — and computes the repository root
from the file's location, so `paths.raw_dir` is `data/raw` *relative to the checkout*, not
to the current working directory. A typo'd or incomplete settings file raises
`ConfigError` at load with the offending key named, rather than falling back to a default:
an application that quietly substituted a default threshold would keep returning 200s, and
those 200s would be wrong.

Resolution order for the config path, implemented in `api/app.py:resolve_config_path`:
explicit argument → `NSM_CONFIG` → search from the package's location. That ordering exists
because a container installs the package into site-packages, where the repository-relative
candidate no longer points at anything, and because a test that passes a path must not be
overridden by an inherited environment variable.

`--offline` is a process-wide switch consumed by the data layer. With it set, any code path
that would open a socket raises instead, which is what allows CI to run the whole suite with
network egress disabled and not merely hope that no test reaches out.

## The cache

`data/cache.py` stores each artifact plus `checksum.json`, a manifest mapping filename to
byte count, sha256, UTC write time, and the URL it came from. Writes go to a temp file and
`os.replace` — a signal mid-download cannot leave a truncated CSV that *looks* valid, which
matters because the next stage of the pipeline trusts its input's existence more than its
contents.

`dataset_fingerprint()` is a sha256 over the sorted manifest of the artifacts the models
were actually built from. It is stamped into every sidecar, exposed by `/health` and
`/meta`, and printed by `status`. Two machines agree on the bytes or the fingerprint says so
immediately. It is also the cache key for the memo in `deps.py`, for the same reason: a
re-fetch must invalidate the derived tables or the API serves features computed from a
snapshot the model was not trained on.

## Splits, training, artifacts

`splits.py` produces train/val/test by **group**, with `StratifiedGroupKFold` for CV and a
greedy quota assignment for the holdout: groups are allocated to whichever split still
needs rows, class counts tracked, so no `kepid` or `des` appears twice and the label ratios
survive. The split design is a documented decision, not an implementation detail —
`/api/v1/meta` returns it, because a reviewer should not have to read the code to know what
"test set" meant here.

`train.py` holds the candidate table. Each candidate is `(name, kind, estimator)`; both rule
baselines and learned models sit in the same list and are scored by the same loop, because
a baseline that is evaluated by different code than the model is not a baseline. The
selection metric is fixed per task before any score is computed — macro-F1 for exoplanets,
average precision for asteroids — and recorded in `reports/*/metrics.json` as
`selection_metric` alongside every candidate's fold scores. The winner is refit on
train+val and evaluated on the untouched holdout exactly once.

`registry.py` saves the estimator (`joblib`) and a JSON sidecar in the same directory. The
sidecar answers, without loading the model: what algorithm, which features and in what
order (`feature_names`, `feature_hash`), what ranges those features had in training
(`feature_ranges`, used to flag extrapolation), which labels, which fingerprint, which
versions (`environment`), when, and what the holdout said (`metrics`, `train_metrics`,
`baseline`, `cv`). `load_artifact` then *refuses*: a feature list that does not match the
live contract, or a scikit-learn version joblib cannot be trusted with, raises rather than
predicting. Joblib survives minor releases and not majors, and `"unknown"` fails closed.

## Serving

`create_app(config_path=None)` is a factory returning an isolated `FastAPI` instance;
`app = create_app()` exists only so `uvicorn nasa_space_ml.api.app:app` works. The factory
matters for tests, which build a second app against a temp repository, and for containers,
where the config arrives via `NSM_CONFIG`. State lives on `app.state.nsm`, so two apps in
one process cannot cross-contaminate.

The lifespan opens the cache and loads artifacts. A model that fails to load does **not**
prevent startup: it is recorded in `app.state.nsm.model_errors`, logged, and the routes that
need it answer 503. The alternative — crash at boot — turns one bad artifact into an outage
for endpoints that never touched it.

`AppState` holds config, cache, loaded models, and errors. It owns no database, no
scheduler, and no global singletons. Two things in it are worth naming:

- **The feature-contract guard** (`_ordered_matrix`): a request row is passed through the
  same `derive()` the training pipeline used, then matched against the artifact's
  `feature_names`. Any mismatch raises `InvalidFeaturesError` with `missing_in_model`,
  `extra_in_model`, and both hashes — the only practical defence against a developer changing
  `FEATURE_COLUMNS` and redeploying the old artifact.
- **The asteroid table memo**: `asteroid_tables()` memoises the 32,390-event aggregation
  against `dataset_fingerprint`. The measured cost of not doing it was ~13 s per
  `/api/v1/asteroids/sample`, which made the console feel broken. Events and objects are
  cached together, because they are always needed as a pair.

Routers are three: `exoplanets`, `asteroids`, `meta`. They contain no arithmetic. Each
defines a Pydantic request model whose bounds come from the physics of the column — a
transit depth cannot exceed 100 % of a star's light, a month is 1–12 — with `extra="forbid"`
so an unexpected key is a 422 instead of something quietly ignored, and nothing required:
`POST {}` scores an all-missing row and returns 200, because "is this endpoint alive"
should be answerable before reading the schema. Label columns are absent from the request
models by construction, which is the last line of the leakage defence.

Errors are RFC 9457 `application/problem+json` produced in one function
(`problem.py:from_internal_error`), carrying `code`, `status`, `detail`, `instance`,
`request_id`, and a structured `errors` block. No traceback crosses the boundary. Every
response carries `X-Request-Id`, echoed if the caller supplied a usable one, generated
otherwise, and included in error bodies so a screenshot identifies a log line. Full table:
[errors.md](errors.md).

## Front end

Three files, no build step, no framework, no CDN, no external origin — a
`re.findall` test asserts the served HTML contains zero `src=`/`href=` pointing at a
different machine. That is not purism: it is the difference between a demo that survives a
review room with no wifi and one that does not.

The console is a screening instrument, not a landing page: fetch a real row, run the
screening, show the probability, the features that mattered, and the provenance block
(model, algorithm, sklearn version, training timestamp, dataset fingerprint, request id)
underneath every answer. It calls `/sample` then `/predict`; it never recomputes a
prediction in JavaScript, so there is exactly one implementation of the model's behaviour.

## Deployment shapes

| Shape | How paths resolve | Notes |
| --- | --- | --- |
| editable install from a checkout | package-relative `parents[3]` | `uvicorn`/`nasa-space-ml serve`; data cache present if fetched |
| wheel/site-packages install | `NSM_CONFIG`, `NSM_WEB_DIR` | the Dockerfile's route; `web/` must be copied or `/` 404s while the API keeps working |
| container | `pip install .` + `NSM_*` env | non-root uid, healthcheck on `/api/v1/health`, `data/raw` intentionally empty |
| CI | checkout, editable, `--offline` for `status` | `docs/superpowers/…` design + plan committed; no network in the suite |

## Testing strategy as an architectural choice

The HTTP layer is `httpx.Client` injected with a `transport`, so tests swap in
`httpx.MockTransport` and exercise pagination, retry, 5xx, malformed CSV, and a version
mismatch without leaving the process. That is not a testing convenience; it is why the
retry policy can be *correct* rather than merely plausible, and why CI is honest about
touching nothing.

Tests that need real cached data skip with the exact command to run
(`pytest.skip("...run nasa-space-ml fetch")`) instead of failing red on a fresh clone,
because a test that cannot pass without 6 MB of gitignored bytes is not a regression test —
it is a fetch check that got lost. `warnings.simplefilter("error")` turns DeprecationWarnings
into failures, which is how the pandas 3.0 migration debt stays visible; the single ignore in
`pyproject.toml` names the library that emits it.

## What is deliberately absent

No database — the dataset is 6 MB of CSV and a second storage engine would add a failure
mode without adding a capability. No message queue or job scheduler; training is a command
you run, not a thing that happens to you. No auth, no rate limiting, no multi-tenancy: a
laptop-hosted capstone with no user accounts, and theatre-grade security would misrepresent
what has been tested. No async route handlers — `httpx` calls happen at fetch time, not
request time, so the request path is CPU-bound and synchronous. No experiment tracker — the
sidecar and the report JSON are the record, and both are in git.
