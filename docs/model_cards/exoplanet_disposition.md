# Model card · `exoplanet_disposition`

Generated from `models/exoplanet_disposition.json` and `reports/exoplanets/metrics.json`.
Every figure here is readable in one of those two files; nothing is estimated.

## Summary

| | |
| --- | --- |
| Name | `exoplanet_disposition` |
| Task | 3-class classification of Kepler DR25 Objects of Interest |
| Labels | `CONFIRMED`, `CANDIDATE`, `FALSE POSITIVE` (order as configured, sklearn sorts them at inference) |
| Algorithm | `learned:hist_gradient_boosting` — `sklearn.ensemble.HistGradientBoostingClassifier` |
| Features | 21, contract hash `41939f75dc829393` |
| Rows / groups | 8,054 rows across 6,923 host-star groups |
| Split | 5,273 train / 1,208 val / 1,573 test, group-disjoint by `kepid` |
| Selection metric | macro-F1, declared before scoring |
| Test macro-F1 | **0.7687** |
| Baseline | majority-class dummy, macro-F1 0.2229 |
| Trained | 2026-11-12T13:44:29+00:00, Python 3.12.10, scikit-learn 1.9.0, Windows 11 ARM64, CPU |
| Dataset fingerprint | `f732fe257e31929eae5bc08409ce59168df76df429911c30a0b949bca38eedbc` |

## What it predicts, precisely

The `koi_disposition` string the Kepler Objects of Interest team published for a signal —
that is, the outcome of a *validation study*, not a fact about the universe. The label
encodes "what peer-reviewed vetting concluded given the data available to that team at that
time". A `CANDIDATE` row is not a planet that has been shown to be unreal, and a
`CONFIRMED` row is not proof of a planet; both are statements about a published catalogue.
The model predicts the catalogue.

## Intended use

Screening and teaching. The console uses it to show what a validation study would look at
and how strongly the physical quantities point. Reasonable uses: prioritising which
candidates a human should read next, exploring which observables drive validation, and
demonstrating an end-to-end ML pipeline on real archive data.

## Out-of-scope uses

Any claim about a specific object's existence. Any use as the validation itself. Anything
outside the DR25 KOI population — the features are Kepler-specific (`koi_kepmag`,
`koi_num_transits`, the Q1–Q17 quarterly coverage), so a TESS or ground-based survey object
has no meaningful input row. Habitability, atmosphere, or biosignature inference; equilibrium
temperature is one of the features and it is not a habitability test. Automated catalogue
updates: a 0.77 macro-F1 model would move real verdicts in the wrong direction often enough
to matter.

## Training data

`https://exoplanetarchive.ipac.caltech.edu/TAP/sync`, table `Q1_Q17_DR25_KOI`, an ADQL
`select` of 21 named columns (no `LIMIT`, which that endpoint rejects). Cached to
`data/raw/koi_q1_q17_dr25.csv`, 1,183,069 bytes, sha256
`180a77a7b9f43cfef5d5259e3804859c30b88a35c31774ce40466abe391e15ec`.

`data/exoplanets.load` drops rows whose `koi_disposition` is null — none in this snapshot —
and `models/train.py` keeps only the three configured label values, which is all of them:
FALSE POSITIVE 3,965 · CONFIRMED 2,731 · CANDIDATE 1,358.

Full column semantics, units, and missingness: [DATA_DICTIONARY.md](DATA_DICTIONARY.md).

### Leakage controls

`data/exoplanets.py::LEAKAGE_FORBIDDEN` lists the columns that encode the verdict rather
than the observation — `koi_pdisposition` (the pipeline's own preliminary disposition, in
the code's words "effectively a copy of the label"), `koi_vet_stat`, `koi_quarters`,
`koi_delivname`. `build_query` refuses to compose an ADQL select that contains any of them,
so leakage is blocked at the point the HTTP request is built rather than at the point the
matrix is assembled — a blacklist enforced this early cannot be bypassed by a later stage
that forgot about it. The list is published by `/api/v1/meta` as
`exoplanets_forbidden_columns`, and `api/schemas.py` omits the label and disposition columns
from the request model entirely with `extra="forbid"`, so posting one is a 422 rather than a
feature. The guard is asserted in `tests/test_data_exoplanets.py`.

Why this matters more here than in most tabular problems: the disposition columns would
predict the label with near certainty and produce a CV score around 1.0 that means nothing.
The failure would be invisible in the metrics and visible only in the archive.

## Features

21 columns in this exact order, from `FEATURE_COLUMNS`:

`log_period`, `log_duration`, `log_depth`, `log_prad`, `prad_inconsistency`,
`duration_over_period`, `impact`, `eccentricity`, `inclination`, `log_sma`,
`equilibrium_temp`, `stellar_teff`, `log_stellar_radius`, `stellar_mass`, `stellar_logg`,
`stellar_density`, `n_transits`, `model_snr`, `max_single_event_snr`, `kepler_mag`,
`koi_count`

Two are derived from physics rather than passed through. `duration_over_period` is the
transit geometry ratio. `prad_inconsistency` compares the catalogue radius against the one
implied by `(Rp/Rs)² ≈ depth`: a large disagreement is what a blended eclipsing binary or a
wrong stellar radius looks like in the numbers, and it is the fourth most important feature
here.

`models/splits.py` guarantees that all of a star's candidates land in the same split. Kepler
reports up to several KOIs per `kepid`, and stellar parameters are shared between them; a
row-level split would let the model meet the same star twice.

## Evaluation

Holdout: 1,573 rows / 1,385 groups, seen exactly once after candidate selection and refit.

| Metric | Value |
| --- | --- |
| macro-F1 (selection) | **0.7687** |
| accuracy | 0.8074 |
| weighted-F1 | 0.8115 |
| ROC-AUC (macro, one-vs-rest) | 0.9226 |
| Brier | 0.0940 |
| log-loss | 0.4966 |
| CV macro-F1, 5 grouped folds | 0.7659 ± 0.0145 |
| majority-class dummy on the same holdout | macro-F1 0.2229, accuracy 0.5022 |

Per class:

| Label | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| CANDIDATE | 0.557 | 0.658 | 0.603 | 269 |
| CONFIRMED | 0.833 | 0.842 | 0.838 | 514 |
| FALSE POSITIVE | 0.898 | 0.835 | 0.866 | 790 |

Confusion (rows = truth):

| | pred CANDIDATE | pred CONFIRMED | pred FALSE POSITIVE |
| --- | ---: | ---: | ---: |
| **CANDIDATE** | 177 | 39 | 53 |
| **CONFIRMED** | 59 | 433 | 22 |
| **FALSE POSITIVE** | 82 | 48 | 660 |

Candidate comparison from `reports/exoplanets/candidate_comparison.png`:

| Candidate | Kind | CV mean macro-F1 | CV std | Fit time |
| --- | --- | ---: | ---: | ---: |
| `hist_gradient_boosting` | learned | **0.7659** | 0.0145 | 14.15 s |
| `random_forest_200` | learned | 0.7602 | 0.0138 | 3.70 s |
| `logistic_regression` | learned | 0.6632 | 0.0107 | 0.72 s |
| `dummy_majority` | rule | 0.2192 | 0.0001 | 0.01 s |

Random forest is within one standard deviation of the winner. That gap is not interesting
enough to claim gradient boosting is *better* here, only that it is better by the
pre-declared metric, and the honest reading is that this dataset is learnable by several
reasonable tree ensembles with a large margin over a linear model.

## Calibration

No reliability curve is recorded for this model (`reports/exoplanets/` holds confusion,
importance, and candidate figures; the calibration figure is only generated for the
imbalanced asteroid task). Brier 0.0940 and log-loss 0.4966 on a 3-class problem with a
0.33-class prior are both consistent with useful probabilities, but that is weaker evidence
than the calibration plot the other model has. Treat the probabilities as *ordering*
information rather than as frequencies. See
[LEARNING_GUIDE.md](LEARNING_GUIDE.md) chapter 12.

## Interpretation

Permutation importance on the holdout, 8 repeats, `f1_macro` scorer
(`reports/exoplanets/permutation_importance.png`):

| Feature | Mean decrease in macro-F1 |
| --- | ---: |
| `model_snr` | 0.1426 |
| `duration_over_period` | 0.0887 |
| `log_prad` | 0.0766 |
| `prad_inconsistency` | 0.0338 |
| `max_single_event_snr` | 0.0290 |
| `log_duration` | 0.0276 |
| `equilibrium_temp` | 0.0179 |
| `koi_count` | 0.0175 |

Signal-to-noise and transit geometry dominating is the physically sensible outcome:
validation is largely a question of "is this signal real and does it look like a planet
transiting a star of this size".

`stellar_logg` (−0.0026) and `log_stellar_radius` (−0.0017) report *negative* importance,
which is not a bug — permuting a redundant feature can accidentally help a shallow model.
It means those two columns carry signal the ensemble already gets from elsewhere
(`stellar_density` is built from both).

## Limitations

Trained on DR25 with the 21 columns this project requested; the archive publishes 153, and
several of the excluded ones (multi-quarter photometry, centroid shifts, difference-imaging
flags) carry real discriminating power that this model cannot see. A score of 0.77 macro-F1
should be read as "what a modest, documented subset of the archive supports", not as a
ceiling for the problem.

`CANDIDATE` is the weakest class and the one with the fewest examples, which is the
expected direction: a candidate's defining property is that it has not been studied enough,
and no feature set can encode that.

Extrapolation is flagged rather than prevented. `feature_ranges` records the training min
and max for every column, and the API annotates a prediction whose inputs fall outside
them. A `koi_prad` of 50,000 Earth radii is inside no Kepler catalogue, and the model will
still hand back probabilities for it.

Snapshot-dependent. NASA re-publishes the KOI table; a re-fetch will change the fingerprint
and invalidate these numbers. Re-fetch, retrain, and compare — the fingerprint in
`/api/v1/health` tells you whether you are looking at the same data.

## Recommendations for monitoring

The feature-contract guard raises `InvalidFeaturesError` rather than serving silently
mismatched predictions. Beyond that: re-check the fingerprint after any re-fetch; re-examine
these metrics if `configs/settings.yaml` labels, thresholds, or column lists change; treat
`nasa-space-ml train` output as authoritative over this card, since the card is a snapshot.

## Use in the API

```http
POST /api/v1/exoplanets/predict
{"koi_period": 9.4888, "koi_depth": 6225.0, "koi_prad": 17.3,
 "koi_srad": 1.79, "koi_steff": 5848.0, "koi_model_snr": 51.2}
```

Returns probabilities for all three classes, argmax prediction, the raw and derived feature
echo, per-feature extrapolation flags, provenance (model, algorithm, sklearn version,
training timestamp, dataset fingerprint), and a `request_id`.
`GET /api/v1/exoplanets/sample` serves a real row, so the request above can be built by the
client rather than by hand.

## Caveats

This is a student capstone, not a pipeline used by a survey team. Not peer-reviewed.
Not endorsed by NASA, the Exoplanet Archive, Caltech, or the Kepler mission.
