# Model card · `asteroid_pha`

Generated from `models/asteroid_pha.json` and `reports/asteroids/metrics.json`. Every figure
here is readable in one of those two files; nothing is estimated.

## Summary

| | |
| --- | --- |
| Name | `asteroid_pha` |
| Task | binary classification — is this near-Earth object a Potentially Hazardous Object |
| Labels | `NOT_PHA`, `PHA` (positive class `PHA`, prevalence 5.34 % on the holdout) |
| Algorithm | `learned:hist_gradient_boosting` — `sklearn.ensemble.HistGradientBoostingClassifier` |
| Features | 19, contract hash begins `b1…` (see the sidecar) |
| Rows / groups | 22,581 objects aggregated from 32,390 close-approach events |
| Split | 14,678 train / 3,387 val / 4,516 test, group-disjoint by `des` |
| Selection metric | average precision, declared before scoring |
| Test average precision | **0.7261** |
| Baseline | NASA's published PHA rule implemented literally, AP 0.2575 |
| Trained | 2026-11-12T13:49:02+00:00, Python 3.12.10, scikit-learn 1.9.0, Windows 11 ARM64, CPU |
| Dataset fingerprint | `f732fe257e31929eae5bc08409ce59168df76df429911c30a0b949bca38eedbc` |

## What it predicts, precisely

Membership of the PHA category as NASA's definition would apply it to this object, given the
close-approach record the CAD API publishes. The definition is a rule with two thresholds:
MOID ≤ 0.05 au and absolute magnitude `H` ≤ 22.0. `configs/settings.yaml` carries both
numbers (`pha_distance_au: 0.05`, `pha_absolute_magnitude: 22.0`) and
`models/baselines.py::PHARuleClassifier` implements them literally, as an sklearn estimator
with the same `fit`/`predict_proba` interface as the learned candidates, so it is scored by
the identical evaluation loop.

It does **not** predict whether an object is dangerous. Impact risk is a function of an
orbit propagated with its covariance against a specific Earth ephemeris over decades, which
is what NASA's Sentry and Scout systems do. This model sees 19 summary statistics of a
catalogue of past and catalogued-future approaches and predicts a label that is largely a
function of two of them.

## Intended use

Teaching an imbalanced-learning pipeline end to end, and demonstrating what the gap between
"the published rule" and "a learned model on the same rule's inputs" looks like when it is
measured honestly. Concretely: a triage aid for sorting which catalogued NEOs deserve a
closer look, at a chosen operating point, by someone who has the real orbit to check
against.

## Out-of-scope uses

Any hazard or impact statement. Any decision about a specific object. Any object with no
close-approach record in CAD — it has no input row, which is the correct answer to give and
the reason `/api/v1/asteroids/predict` requires MOID and `H` rather than inventing them.
Newly discovered objects, whose arcs are days old and whose uncertainty regions are
enormous; `observation_span_years` is a feature here and a 3-day arc is far outside the
training distribution, and the API flags it as extrapolation rather than hiding it.
Physical-property inference: diameter here is *derived from* `H` under an assumed albedo, so
it cannot be evidence about the object's size.

## Training data

`https://ssd-api.jpl.nasa.gov/cad.api.json`, version 1.5 (asserted against
`signature.version`; a different version raises rather than guessing), parameters
`date-min=2020-01-01`, `date-max=2031-01-01`, `dist-max=0.2` au, paged at `limit=5000` up to
20 pages. Cached to `data/raw/cad_approaches.csv` (4,943,794 bytes, sha256
`9323d71963e86b29ef4e4e72dade7574a8ac6cd2f0cbdea875a90b20e6493400`) plus
`data/raw/cad_pha_des.txt` (9,951 bytes, 1,206 designations) — the second artifact is the
label source, and reading it as a designation list rather than inferring labels from the
event fields is what makes this a real supervised problem instead of a restatement of the
rule.

**The API returns events, not objects.** 32,390 rows → 22,581 objects, aggregated by
`features/asteroids.py::aggregate_events`: one row per designation, with min-distance
statistics, a closest-approach descriptor set, counts, and derived physical estimates. Full
column semantics and missingness rates: [DATA_DICTIONARY.md](DATA_DICTIONARY.md).

### Why the label is not the rule

The temptation in this problem is to compute `MOID ≤ 0.05 and H ≤ 22.0` and call that the
target — which gives a 1.000 AP model that has learned nothing. The label here comes from
JPL's own PHA list (`cad_pha_designations`), which is maintained by humans applying that
definition to *properly computed* MOIDs from orbital elements. The gap between the published
label and what these two thresholds would say about `min(dist_min)` is exactly the signal the
model has to find, and it is why the rule baseline scores 0.2575 rather than 1.0.

## Features

19 columns in this exact order, from `FEATURE_COLUMNS`:

`moid_proxy_au`, `log_moid`, `moid_lunar_distances`, `n_approaches`,
`n_approaches_within_lunar`, `v_rel_at_closest`, `v_inf_at_closest`,
`uncertainty_ratio_at_closest`, `absolute_magnitude_h`, `diameter_from_h_km`,
`log_diameter_from_h`, `diameter_reported_km`, `diameter_inconsistency_km`,
`implied_albedo`, `n_orbit_solutions`, `t_sigma_seconds_at_closest`,
`observation_span_years`, `closest_month_sin`, `closest_month_cos`

`moid_proxy_au = min(dist_min)` over an object's recorded approaches. It is a *proxy*: the
true MOID is computed from orbital elements, and this is the smallest catalogued close
approach distance instead. An object whose only catalogued approaches are 0.15 au but whose
elements imply a 0.02 au crossing in 2038 looks non-hazardous here. This is the single most
important limitation of the model and it is a property of the feed, not of the learning
algorithm.

`diameter_from_h_km` is the Harris–Tumbioli relation `D = 1329/√p · 10^(-H/5)` evaluated at
the assumed geometric albedo `p = 0.14` (`default_geometric_albedo` in settings). `p = 0.14`
is the conventional choice for an unknown-composition NEO; it is a documented assumption
carried in the sidecar, not a measurement. `implied_albedo` inverts the same identity using
`diameter_reported_km` when JPL reports one, and `diameter_inconsistency_km` is their
difference — the "is this object's catalogue diameter consistent with its brightness at the
assumed albedo" question.

`closest_month_sin` / `closest_month_cos` encode calendar month on the unit circle, because
month is cyclic and a model that sees January = 1 and December = 12 is being invited to
believe those are 11 units apart.

`log_moid`, `moid_lunar_distances`, `n_approaches_within_lunar`, `log_diameter_from_h` are
monotone recombinations of features already in the list. They are kept because they make the
physical statement legible in the console — the Moon's distance is 0.00257 au, and
"1.4 lunar distances" communicates to a human in a way `0.0036` does not — and the model
pays for them, not the reviewer: their permutation importances come out at exactly 0.0.

## Evaluation

Holdout: 4,516 objects (241 PHA), seen once after selection and refit.

| Metric | Value |
| --- | --- |
| average precision (selection) | **0.7261** |
| ROC-AUC | 0.9778 |
| prevalence | 0.0534 |
| recall at 0.5 | 0.9336 |
| precision at 0.5 | 0.4787 |
| F1 (positive) | 0.6329 |
| macro-F1 | 0.8008 |
| Brier | 0.0425 |
| CV average precision, 5 grouped folds | 0.7386 ± 0.0249 |

Confusion at threshold 0.5 (rows = truth):

| | predicted NOT_PHA | predicted PHA |
| --- | ---: | ---: |
| **NOT_PHA** (4,275) | 4,030 | 245 |
| **PHA** (241) | 16 | 225 |

Against the baselines on the same holdout:

| Model | AP | ROC-AUC | PHA recall | PHA precision |
| --- | ---: | ---: | ---: | ---: |
| `hist_gradient_boosting` | **0.7261** | 0.9778 | 0.9336 | 0.4787 |
| `nasa_pha_rule` | 0.2575 | 0.6098 | 0.2199 | 0.9815 |
| `logistic_regression` (CV) | 0.5266 | 0.9667 | — | — |
| `dummy_majority` (CV) | 0.0534 | 0.5000 | 0.0000 | — |

The rule's numbers are the interesting part of this card. Applied to these inputs it flags
54 objects, 53 of which are genuinely PHA — 98.1 % precision — but it finds only 22 % of the
241 positives, because `min(dist_min)` is a poor stand-in for a computed MOID and because
`H` is a catalogue field with gaps. ΔAP over the rule is **+0.4686**. The learned model is
not better at *being* the rule; it is better at guessing which objects JPL's humans ended up
listing, using the same two physical quantities plus seventeen others.

`reports/asteroids/metrics.json` records `pha_rule_unknowns: 0` — on this holdout every
object had both MOID and `H` available, so the rule's third outcome ("cannot evaluate") never
fired. On a fresher snapshot with unmeasured magnitudes it will.

## Calibration

`reports/asteroids/calibration.png`, 10 uniform bins over 4,516 holdout rows, with the
underlying numbers embedded in `metrics.json`:

| Binned mean predicted | Observed PHA fraction |
| ---: | ---: |
| 0.0049 | 0.0008 |
| 0.1370 | 0.0606 |
| 0.2493 | 0.0909 |
| 0.3409 | 0.1852 |
| 0.4414 | 0.1600 |
| 0.5582 | 0.2143 |
| 0.6561 | 0.3000 |
| 0.7535 | 0.3594 |
| 0.8587 | 0.4096 |
| 0.9489 | 0.5882 |

The ordering is right and the magnitudes are not. Every bin over-predicts relative to what
actually occurs — the most-confident 10 % of rows are 58.8 % PHA, not 94.9 % — which is the
signature of a boosting model trained on a 5 % positive class: cross-entropy pressure makes
it sharper than the data supports. It is *informative* mis-calibration, not random noise:
probabilities rank correctly, so they are safe for prioritising and unsafe as frequencies.
Do not quote "94.9 % chance this is hazardous" to anyone. Quote the rank, or quote the
precision at the threshold you chose, which is what the operating points below give.

## Operating points

From the same file — thresholds chosen to *achieve* a target precision on the holdout, which
is the honest way to state an operating point for an imbalanced problem:

| Target precision | Threshold | Achieved precision | Recall |
| ---: | ---: | ---: | ---: |
| 0.10 | 0.0053 | 0.3604 | 0.9959 |
| 0.25 | 0.0053 | 0.3604 | 0.9959 |
| 0.50 | 0.6359 | 0.5000 | 0.8963 |
| 0.75 | 0.9412 | 0.7536 | 0.4315 |

The first two rows landing on the same threshold is not a bug: the model cannot be made to
deliver 10 % precision, because its lowest non-trivial threshold already yields 36 % — the
precision floor is set by prevalence and by how much of the positive class the model can see
at all.

At 75 % precision the model catches 104 of 241 real PHAs and misses 137. That is the trade,
stated plainly. The default 0.5 threshold catches 225 and accepts 245 false alarms. Which is
right is an operational question about the cost of a wasted follow-up observation versus a
missed one, and this repository deliberately does not answer it — it gives you the curve and
the numbers and leaves the decision where it belongs.

## Interpretation

Permutation importance, 8 repeats, `average_precision` scorer with `pos_label="PHA"`
(`reports/asteroids/permutation_importance.png`):

| Feature | Mean decrease in AP |
| --- | ---: |
| `absolute_magnitude_h` | 0.4982 |
| `moid_proxy_au` | 0.2870 |
| `diameter_from_h_km` | 0.1196 |
| `uncertainty_ratio_at_closest` | 0.0180 |
| `diameter_inconsistency_km` | 0.0123 |
| `t_sigma_seconds_at_closest` | 0.0089 |
| `v_rel_at_closest` | 0.0037 |
| `closest_month_sin` | 0.0029 |

Two features account for 0.785 of the roughly 0.957 total importance mass — `H` and the
MOID proxy, which are the two terms in NASA's definition. A model that has learned the rule's
*inputs* well is not automatically smart, so the interesting part is the remaining 0.17.

`diameter_from_h_km` at 0.1196 needs a careful reading rather than a triumphant one: it is a
deterministic monotone function of `absolute_magnitude_h`, and no tree can gain information
from a monotone re-expression of a feature it already has. What the number actually measures
is redundancy. The ensemble drew some trees on `H` and others on its diameter transform;
permuting one degrades only the trees that chose it, so importance gets *split* between two
columns carrying identical signal. The honest combined statement is that brightness is worth
about 0.62 and geometry about 0.29 — and the fact that `log_moid` and `log_diameter_from_h`
score exactly 0.0 is the same phenomenon in its pure form, which is why the monotone
recombinations in this feature list are documentation for humans rather than evidence about
the world. The genuinely new information, the part the published rule cannot see at all,
lives in `uncertainty_ratio_at_closest` (0.0180) and `t_sigma_seconds_at_closest` (0.0089) —
how well the orbit is known — which is exactly where JPL's list-makers depart from a
two-threshold test.

Six features report importance of exactly 0.0 (`log_moid`, `moid_lunar_distances`,
`n_approaches_within_lunar`, `log_diameter_from_h`, `n_orbit_solutions`) or slightly negative
(`implied_albedo`, −0.0001). These are the redundant recombinations and, for albedo, a
quantity the model can already reconstruct from `diameter_reported_km` and `H`. They cost
inference time and buy nothing; they are retained for legibility in the console, and this
card says so rather than presenting them as insight.

## Limitations

The MOID proxy. It cannot see an approach that has not been catalogued, and the closest
recorded approach is not the minimum orbit intersection distance. This is the reason the rule
baseline scores 0.26 rather than 1.0 and the reason a high probability from this model is a
suggestion to look at the orbit, not a conclusion about it.

Precision 0.4787 at the default threshold. Roughly half the objects this calls PHA are not
on JPL's list. On the test holdout that is 245 wrong alarms to catch 225 real ones.

Label is a snapshot of a curated list. JPL adds and removes objects as arcs lengthen; the
1,206 designations cached here were the list on 2026-11-12, and an object that has since
acquired a better orbit may have changed category. The fingerprint is the defence: rerun
`nasa-space-ml fetch && nasa-space-ml train --task asteroids` and compare.

Missingness. `diameter_reported_km` is present for roughly 1.5 % of objects (~98.5 %
missing), so `implied_albedo` and `diameter_inconsistency_km` are mostly `NaN` — handled
natively by the boosting model's sparse splits, but they are features the model rarely gets
to use.

Time window. Approaches from 2020-01-01 to 2031-01-01 out to 0.2 au. Objects whose next
close approach is in 2040 contribute nothing here.

No physical risk model: no mass, no composition, no impact-energy, no population-based
Torino/Scales contribution. This is a catalogue-membership classifier wearing a more
evocative problem statement.

## Recommendations for monitoring

The feature-contract guard raises `InvalidFeaturesError` if `FEATURE_COLUMNS` and the
artifact disagree, and refuses to serve an artifact whose sklearn version it cannot vouch
for. Beyond that: compare the fingerprint on every re-fetch — the CAD table updates weekly;
watch `PHA recall` on new designations specifically, since it is the failure mode with
consequences; treat `nasa-space-ml train --task asteroids` output as authoritative over this
card.

## Use in the API

```http
POST /api/v1/asteroids/predict
{"moid_proxy_au": 0.0312, "absolute_magnitude_h": 21.4,
 "diameter_reported_km": 0.28, "uncertainty_ratio_at_closest": 0.017,
 "observation_span_years": 12.3, "closest_approach_month": 3}
```

All fields optional, physical bounds enforced, unknown keys rejected. The response carries
both class probabilities, the argmax label, the raw and derived feature echo, a provenance
block, and an explicit note comparing the answer to what NASA's published rule would have
said at 0.05 au / H 22.0 — because a prediction that hides the rule it was measured against
is the easiest way to fool a reviewer. `GET /api/v1/asteroids/sample` serves a real
aggregated object.

## Caveats

A student capstone. Not peer-reviewed, not operated by any survey programme, and not
connected to NASA's Scout or Sentry risk assessment in any way. Not endorsed by NASA, JPL,
or Caltech. If you are looking for real information about a specific asteroid, use
[`https://cneos.jpl.nasa.gov/`](https://cneos.jpl.nasa.gov/).
