# nasa-space-ml

**Two machine learning problems built from live NASA data, served by one FastAPI
application with a screening console in front of it.**

| | Problem 1 | Problem 2 |
| --- | --- | --- |
| Question | Will the Kepler Objects of Interest team validate this candidate? | Is this near-Earth object potentially hazardous? |
| Source | [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) TAP/ADQL | [JPL Small-Body Database](https://ssd-api.jpl.nasa.gov/doc/cad.html) close-approach API |
| Table | `Q1_Q17_DR25_KOI` — 8,054 rows, 6,923 host stars | 32,390 close approaches of 22,581 objects, 2020–2031 |
| Target | `koi_disposition` — CONFIRMED / CANDIDATE / FALSE POSITIVE | `PHA` / `NOT_PHA`, 5.3 % positive |
| Model | `HistGradientBoostingClassifier` | `HistGradientBoostingClassifier` |
| Selection metric | macro-F1 **0.7687** | average precision **0.7261** |
| Baseline it beats | majority-class dummy, macro-F1 0.2229 | NASA's own published PHA rule, AP 0.2575 |

Both models, both report suites, and every number below were produced on a CPU-only
laptop from public NASA endpoints that require no API key. Nothing in this repository is
a mock, a placeholder, or a synthetic dataset.

---

## Run it in sixty seconds

```bash
git clone https://github.com/Yuvraj0967/nasa-space-ml
cd nasa-space-ml
uv venv && uv pip install -e ".[dev]"     # or: python -m venv .venv && .venv/bin/pip install -e ".[dev]"

.venv/bin/nasa-space-ml serve             # http://127.0.0.1:8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The console screens a set of values
through the shipped model, shows the probabilities, the features that moved the answer, and
the provenance block underneath.

**No training required.** The two artifacts under `models/` and the figures under
`reports/` are committed, so `/predict` is useful the moment the service boots — which is
the whole point of shipping them. The data cache under `data/raw/` is *not* committed (NASA's
bytes belong to NASA, and 6 MB of CSV regenerates in a couple of minutes), so the routes that
read real rows — `/api/v1/*/sample`, which the console's **FETCH ROW** button uses — answer
502 with the command to run, and `/api/v1/meta` reports an empty snapshot whose fingerprint
plainly disagrees with the artifacts'. Fetch once and both go away:

```bash
.venv/bin/nasa-space-ml fetch             # ~2 min, ~9 HTTPS requests, no key needed
```

On Windows the paths are `.venv\Scripts\nasa-space-ml.exe`, and `uv` works the same way.

### Or run it in a container

```bash
docker build -t nasa-space-ml .
docker run -p 8000:8000 nasa-space-ml
```

The image is built from wheels the builder stage downloaded, so the runtime layer needs
no index access, and it runs as an unprivileged uid. Models and reports are baked in; the
data cache is deliberately not.

---

## What is in here

```
configs/settings.yaml      the single source of truth: URLs, columns, labels, thresholds, seed
src/nasa_space_ml/
  data/                    TAP + CAD clients, retry/backoff HTTP layer, content-addressed cache
  features/                physics-derived features, the leakage blacklist, the feature contract
  models/                  grouped CV splits, candidate definitions, NASA-rule baseline, registry
  evaluation/              metrics, operating points, calibration, matplotlib reports
  api/                     FastAPI app factory, Pydantic contracts, problem+json errors, routers
  cli.py                   nasa-space-ml fetch | train | status | serve | all
web/                       the console: index.html + styles.css + app.js, no framework, no CDN
tests/                     371 tests, none of which touch the network
docs/                      the guide below
models/ reports/           trained artifacts and their figures — committed on purpose
data/                      gitignored; regenerate with `nasa-space-ml fetch`
.github/workflows/ci.yml   lint, test (3.11 + 3.12), boot-the-service smoke job, docker build
```

### Documentation

The docs are ordered by how long a reviewer has.

| File | Read it when |
| --- | --- |
| this file | you want to know what the thing is and run it |
| [docs/LEARNING_GUIDE.md](docs/LEARNING_GUIDE.md) | you want to understand every concept used, chapter by chapter, with the code that implements it |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | you want the module boundaries, the request lifecycle, and why the seams are where they are |
| [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) | you want every column, its units, its missingness, and its role |
| [docs/model_cards/](docs/model_cards/) | you want the per-model facts: data, metrics, limits, intended use |
| [docs/errors.md](docs/errors.md) | you got a 4xx/5xx and want to know what the `type` URI means |
| [docs/superpowers/](docs/superpowers/) | you want the original design spec and implementation plan this was built from |

---

## The two models, honestly

### 1. Exoplanet candidate validation

The Kepler mission produced thousands of transit signals; a "candidate" (a KOI) becomes a
"confirmed planet" only after a validation study rules out eclipsing binaries, blend
targets, and instrumental artifacts. That study is human labour measured in person-months,
and its verdict is published in `koi_disposition` — which makes it a label worth learning
from.

21 columns are pulled by an ADQL `select`, turned into 21 model features by
`features/exoplanets.py` — log transforms on the quantities spanning orders of magnitude,
plus two that come from physics rather than from the table: `duration_over_period` and
`prad_inconsistency`, the latter comparing the radius the catalogue reports against the
radius implied by `(Rp/Rs)² ≈ depth`. Splitting is grouped by `kepid`, the host star, so a
star never appears on both sides of an evaluation.

Test-set results, held out and untouched until the final line of `models/train.py`:

| | value |
| --- | --- |
| macro-F1 | **0.7687** |
| accuracy | 0.8074 |
| ROC-AUC (macro / one-vs-rest) | 0.9226 |
| Brier score | 0.0940 |
| per-class F1 | CANDIDATE 0.6031 · CONFIRMED 0.8375 · FALSE POSITIVE 0.8656 |
| majority-class dummy, same metric | 0.2229 |
| CV mean ± std across 5 grouped folds | 0.7659 ± 0.0145 |

The weakest class is `CANDIDATE` — which is the correct thing to be weak at. "Still being
studied" is not a physical category; it is a queue position, and the features do not
determine it.

### 2. Potentially Hazardous Object classification

NASA's published definition is a rule: a minimum orbit intersection distance (MOID) of at
most 0.05 au **and** an absolute magnitude `H` of at most 22.0. Here that rule is implemented literally as a scikit-learn estimator
(`models/baselines.py::PHARuleClassifier`) and it is the baseline every learned candidate
has to beat.

The CAD API returns close-approach *events*; `features/asteroids.py` aggregates them to one
row per object: 32,390 events → 22,581 objects, with `moid_proxy_au = min(dist_min)` over
each object's recorded approaches, an implied diameter from the Harris–Tumbioli relation
`D = 1329/√p · 10^(-H/5)` at an assumed albedo of `p = 0.14`, and that assumption written
into the sidecar rather than hidden in a notebook.

| | value |
| --- | --- |
| average precision | **0.7261** |
| ROC-AUC | 0.9778 |
| recall at threshold 0.5 | 0.9336 |
| precision at threshold 0.5 | 0.4787 |
| NASA rule, same dataset, same split | AP 0.2575, recall 0.2199 |
| prevalence of PHA | 5.34 % |
| CV mean ± std across 5 grouped folds | 0.7386 ± 0.0249 |

Confusion matrix on the 4,516-object test holdout (rows are truth):

| | predicted NOT_PHA | predicted PHA |
| --- | ---: | ---: |
| **NOT_PHA** (4,275) | 4,030 | 245 |
| **PHA** (241) | 16 | 225 |

Six missed PHAs out of 241 is the number that matters, and 245 false alarms is the price
paid for it. `reports/asteroids/metrics.json` records precision-conditioned operating
points so you can choose a different price: holding 75 % precision pushes the threshold to
≈0.941 and recall down to ≈0.43.

**Neither model is a hazard assessment.** This predicts membership of a catalogue, from
two-dimensional summary statistics, for objects whose orbits were fitted from a few
thousand observations. Real NEO risk determination is done by NASA's Scout and Sentry
systems over full covariance-propagated ephemerides, and nothing here competes with that.

---

## Commands

```
nasa-space-ml [--config PATH] [--offline] fetch                 download and cache both sources
nasa-space-ml … train [--task exoplanets|asteroids|both] [--no-importance]
                          grouped CV, select the winner, refit, score the holdout, write artifacts
nasa-space-ml … status                       what is cached, what is trained, and the fingerprints
nasa-space-ml … serve [--host --port --reload]
nasa-space-ml … all                          fetch, then train
python -m nasa_space_ml …                    the same thing, uninstalled
```

`--offline` makes every network path raise rather than reach out, which is what lets the
test suite and CI run with a laptop radio switched off.

## API

Eight JSON endpoints under `/api/v1`, documented at
[`/docs`](http://127.0.0.1:8000/docs) once running, plus the console at `/`.

| Route | What it does |
| --- | --- |
| `GET /api/v1/health` | model load state, versions, dataset fingerprint |
| `GET /api/v1/meta` | sources, feature contract, leakage guard, split design, cache state |
| `GET /api/v1/exoplanets/schema` | request fields with units, bounds, and derived-feature notes |
| `GET /api/v1/exoplanets/sample[?index&disposition&limit]` | a real row, straight from the cache |
| `POST /api/v1/exoplanets/predict` | 3-class probabilities, per-feature explanation, provenance |
| `GET /api/v1/asteroids/schema` | same contract for the PHA model |
| `GET /api/v1/asteroids/sample[?index&pha&limit]` | a real aggregated object |
| `POST /api/v1/asteroids/predict` | PHA probability, NASA-rule comparison, provenance |

A `POST` with an empty body returns 200 with a prediction on an all-missing row rather
than a validation error — gradient boosting tolerates `NaN` natively, and "does this
endpoint work" should be answerable before you have read the schema. It is not a useful
prediction: `FALSE POSITIVE` at 93.6 % is what a model reduced to its prior says. Every
request gets
an `X-Request-Id`, echoed in the header and repeated inside any error body. Errors follow
[RFC 9457](https://datatracker.ietf.org/doc/html/rfc9457) with `application/problem+json`;
see [docs/errors.md](docs/errors.md).

## Tests and CI

```bash
.venv/bin/pytest                                   # 371 tests, ~25 s, no network
.venv/bin/pytest -m network                        # 3 opt-in probes against the real endpoints
.venv/bin/ruff check src tests scripts && .venv/bin/ruff format --check src tests scripts
```

Warnings are errors. Tests that need the data cache skip with the exact command to run
rather than failing red on a machine that has not fetched anything. GitHub Actions runs
three jobs on every push: `lint`, `test` across Python 3.11 and 3.12 — including an
assertion that the committed artifacts' feature lists still match the code that builds
them — and `smoke`, which boots the service, scores a real Kepler candidate over HTTP,
loads the console, and then builds and runs the Docker image.

## Environment

Developed and trained on Windows 11 / ARM64, Python 3.12.10, CPU only. Pinned versions
recorded in each sidecar's `environment` block and used for the numbers above:

`numpy 2.5.0 · pandas 3.0.3 · scikit-learn 1.9.0 · matplotlib 3.11.0 · joblib 1.5.3 ·
httpx 0.28.1 · fastapi 0.141.1 · pydantic 2.13.5`

Joblib artifacts round-trip across scikit-learn *minor* versions but not majors; the
registry checks the stamped version and refuses to serve a model it cannot vouch for,
which is why `scikit_learn: "unknown"` fails closed rather than open.

## What this project is not

Worth stating before a reviewer has to ask:

- It is not novel. Gradient boosting on tabular astrometry has been done many times; the
  contribution here is the end-to-end shape — real API, real labels, real holdout, real
  errors, real CI — not a state-of-the-art score.
- It is not a predictor of habitability, and says nothing about life.
- The asteroid model's `moid_proxy_au` is the minimum of the *reported* close-approach
  distances, not a computed MOID from orbital elements. It cannot see an approach NASA has
  not catalogued, and it cannot see one that a longer arc would reveal. That is a real
  limitation of the CAD feed, and it is why this is a catalogue-membership model.
- 0.4787 precision means roughly half of the objects this calls PHA are not. On the test
  holdout that is 245 wrong alarms to catch 225 real ones. Whether that trade is right is
  an operational question this repository deliberately does not answer for you.
- The front end has not been audited for accessibility beyond semantic markup and a
  keyboard-reachable console.
- No secrets, no keys, no paid services. If a `.env` ever appears here, that is a bug to
  report, not a feature to configure.

## License and attribution

Code, models, and documentation: MIT — see [LICENSE](LICENSE).

Data: NASA's. The Exoplanet Archive and the JPL Small-Body Database are US government
works with their own citation and usage terms
([archive notice](https://exoplanetarchive.ipac.caltech.edu/docs/notice.html),
[JPL terms](https://www.jpl.nasa.gov/terms/)). This project is not endorsed by or affiliated
with NASA, JPL, or Caltech, and no NASA bytes are redistributed here — the cache is
gitignored and every fetch is a fresh call to the public endpoint.

If you cite the numbers, cite the dataset fingerprint with them:
`f732fe257e31929eae5bc08409ce59168df76df429911c30a0b949bca38eedbc`, snapshot
`2026-11-12T13:33:35+00:00`. NASA re-publishes these tables; a score without a fingerprint
is a score that cannot be re-checked.
