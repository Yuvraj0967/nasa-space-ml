# Data Dictionary

This document is the contract for the data that produced the shipped models. It records the
NASA sources, the exact columns each request pulls, how those columns become features, the
label conventions, and the physical bounds the API enforces. Every count, byte size, hash and
formula below was read from this repository's files or computed from its snapshots; a reviewer
can check them against `configs/settings.yaml`, the `src/nasa_space_ml/**` modules, the two
`models/*.json` sidecars and the bytes under `data/raw/`.

**Snapshot identity.** Both model sidecars (`models/exoplanet_disposition.json`,
`models/asteroid_pha.json`) are stamped with the same
`dataset_fingerprint = f732fe257e31929eae5bc08409ce59168df76df429911c30a0b949bca38eedbc`.
That hash is a content-addressed digest of the three cached artifacts described below, so it
tells you whether your own `nasa-space-ml fetch` produced the same bytes this document
describes. If your fingerprint differs, NASA has re-published a table or a close-approach page
since `2026-11-12T13:33:35+00:00` (the newest `written_utc` in `data/raw/checksum.json`) and
the numbers here are no longer a match for your data.

| Artifact | Bytes | sha256 (from `checksum.json`) |
| --- | ---: | --- |
| `data/raw/koi_q1_q17_dr25.csv` | 1,183,069 | `180a77a7b9f43cfef5d5259e3804859c30b88a35c31774ce40466abe391e15ec` |
| `data/raw/cad_approaches.csv` | 4,943,794 | `9323d71963e86b29ef4e4e72dade7574a8ac6cd2f0cbdea875a90b20e6493400` |
| `data/raw/cad_pha_des.txt` | 9,951 | `fb35e4334b937912ef3e7ec13bde1354ed76dfe91f7fc3e0c1dc1fedbdaa9d1f` |

---

## Source 1 - NASA Exoplanet Archive (TAP/ADQL)

- **Endpoint:** `https://exoplanetarchive.ipac.caltech.edu/TAP/sync` (`exoplanets.source`).
- **Table:** `Q1_Q17_DR25_KOI` (`exoplanets.table`) - the Kepler DR25 Objects of Interest for
  quarters Q1-Q17. The archive table has 153 columns; this project requests 21 of them.
- **Query shape** (built by `data/exoplanets.build_query`, `limit` is deliberately never used
  because that endpoint rejects it with ORA-00933):

  ```sql
  select kepoi_name, kepid, koi_disposition, koi_period, koi_duration, koi_depth, koi_prad,
         koi_impact, koi_eccen, koi_incl, koi_sma, koi_teq, koi_steff, koi_srad, koi_smass,
         koi_slogg, koi_num_transits, koi_model_snr, koi_max_sngle_ev, koi_kepmag, koi_count
  from Q1_Q17_DR25_KOI
  ```

  `download()` sends it as HTTP params `{"query": <adql>, "format": "csv"}`, validates the CSV
  (`parse_csv`) before committing, and writes it to `data/raw/koi_q1_q17_dr25.csv`.

- **Raw file:** `data/raw/koi_q1_q17_dr25.csv`, **1,183,069 bytes**, **8,054 data rows** (21
  columns + header).
- **How `load()` filters rows:** `data/exoplanets.load` reads the snapshot, coerces the numeric
  columns, then keeps only rows where the label column `koi_disposition` is not null
  (`frame[frame[label].notna()]`) and resets the index. In this shipped snapshot no row has a
  null disposition, so **8,054 rows survive** `load()` unchanged; training (`load_exoplanet_matrices`)
  then keeps only the three configured label values, which are the only ones present
  (FALSE POSITIVE 3,965 / CONFIRMED 2,731 / CANDIDATE 1,358 = 8,054). The sidecar records
  `rows: 8054`, `groups: 6923`.

### Requested KOI columns

Missing-value policy is uniform: `parse_csv` runs `pd.to_numeric(..., errors="coerce")` on every
numeric column, so a blank survey cell - which means "not measured", never zero - becomes `NaN`.
`kepoi_name`, `kepid` and `koi_disposition` are kept as read (text/identifier).

| Column | Type | Units | Meaning | Missing-value policy | Role |
| --- | --- | --- | --- | --- | --- |
| `kepoi_name` | text | - | Candidate id, e.g. `K00106.02` | kept as text | excluded (identifier) |
| `kepid` | integer | - | 8-digit EPIC host-star id | kept as text | **group** (split key) |
| `koi_disposition` | categorical | - | Validation verdict | null -> row dropped by `load()` | **label** |
| `koi_period` | float | days | Orbital period | coerce->NaN | feature (`log_period`) |
| `koi_duration` | float | hours | Transit duration | coerce->NaN | feature (`log_duration`) |
| `koi_depth` | float | ppm | Transit depth (ratio of areas) | coerce->NaN | feature (`log_depth`, feeds `prad_inconsistency`) |
| `koi_prad` | float | Earth radii | Catalogue planet radius | coerce->NaN | feature (`log_prad`, feeds `prad_inconsistency`) |
| `koi_impact` | float | - | Impact parameter b | coerce->NaN | feature (`impact`, passthrough) |
| `koi_eccen` | float | - | Eccentricity | coerce->NaN | feature (`eccentricity`, passthrough) |
| `koi_incl` | float | degrees | Orbital inclination | coerce->NaN | feature (`inclination`, passthrough) |
| `koi_sma` | float | AU | Semi-major axis | coerce->NaN | feature (`log_sma`) |
| `koi_teq` | float | K | Equilibrium temperature | coerce->NaN | feature (`equilibrium_temp`, passthrough) |
| `koi_steff` | float | K | Stellar effective temperature | coerce->NaN | feature (`stellar_teff`, passthrough) |
| `koi_srad` | float | solar radii | Stellar radius | coerce->NaN | feature (`log_stellar_radius`; feeds `prad_inconsistency`, `stellar_density`) |
| `koi_smass` | float | solar masses | Stellar mass | coerce->NaN | feature (`stellar_mass`; feeds `stellar_density`) |
| `koi_slogg` | float | dex (log10 cm/s^2) | Stellar surface gravity | coerce->NaN | feature (`stellar_logg`, passthrough) |
| `koi_num_transits` | float | count | Observed transits | coerce->NaN | feature (`n_transits`, passthrough) |
| `koi_model_snr` | float | SNR | Model signal-to-noise | coerce->NaN | feature (`model_snr`, passthrough) |
| `koi_max_sngle_ev` | float | SNR | Strongest single-event SNR | coerce->NaN | feature (`max_single_event_snr`, passthrough) |
| `koi_kepmag` | float | Kepler mag | Kepler magnitude | coerce->NaN | feature (`kepler_mag`, passthrough) |
| `koi_count` | float | count | Candidates on this star | coerce->NaN | feature (`koi_count`, passthrough; flagged confound) |

### Leakage-forbidden columns

`build_query` refuses to emit any of these (raises `DataError` if requested) because each one
is downstream of, or a copy of, the disposition verdict - training on them yields a model that
scores beautifully and means nothing:

- `koi_pdisposition` - the pipeline's own preliminary verdict; effectively a copy of the label
  (this rationale is stated in the module docstring).
- `koi_vet_stat` - vetting-status flag set during validation, so it encodes that a disposition
  decision has already been made.
- `koi_quarters` - the delivery-quarter signature the disposition was computed for; a verdict-
  dependent bookkeeping field rather than a physical measurement.
- `koi_delivname` - the data-delivery/version identifier assigned alongside the disposition call.

---

## Source 2 - JPL SBDB Close Approach Data (CAD v1.5)

- **Endpoint:** `https://ssd-api.jpl.nasa.gov/cad.api.json` (`asteroids.source`).
- **API version contract:** `asteroids.expected_api_version = "1.5"`. `_validate_signature`
  aborts if `signature.version` differs, because the payload is **positional arrays**: a silent
  field reorder would mis-assign every number.

- **Exact params sent** (`data/asteroids.build_params`, per page):

  | Param | Value | Source |
  | --- | --- | --- |
  | `date-min` | `2020-01-01` | `asteroids.date_min` |
  | `date-max` | `2031-01-01` | `asteroids.date_max` |
  | `dist-max` | `0.2` | `asteroids.dist_max_au` (AU) |
  | `diameter` | `1` | makes the optional `diameter`/`diameter_sigma` columns appear |
  | `limit` | `5000` | `asteroids.page_limit` |
  | `limit-from` | `1`, then advanced by rows seen | offset cursor |
  | `sort` | `date` | fixed |
  | `pha` | `1` | added only for the second (PHA-only) walk |

- **The `fields` header contract.** CAD returns rows as unnamed positional arrays plus a
  `fields` array that names them. `_page` reads `fields` from the first page, refuses to parse a
  page that has rows but no `fields` array, and raises if the order changes between pages. The
  `fields` observed for this request are exactly `EVENT_FIELDS` (13 names), which is also the
  header written to the CSV. Pagination stops on an empty page or a short page (`len(data) < 5000`)
  or when `limit-from` passes the first-page `total`, with a hard cap of `max_pages = 20`.

- **Artifacts written to `data/raw`:**

  | Artifact | Bytes | Contents |
  | --- | ---: | --- |
  | `cad_approaches.csv` | 4,943,794 | **32,390 event rows** (one row per close approach), **22,581 distinct `des` objects** |
  | `cad_pha_des.txt` | 9,951 | **1,206 distinct designations**, newline-separated, sorted set from a second `pha=1` walk |

  `load_events` labels each object by membership: `pha_flag = des in cad_pha_des.txt`. In this
  snapshot 1,634 of the 32,390 events belong to PHA objects, which aggregate up to the 1,206
  PHA objects. The sidecar records `rows: 22581`, `groups: 22581`, `n_approach_events: 32390`.

### CAD event fields (positional)

| Field | Position in row | Type | Meaning |
| --- | ---: | --- | --- |
| `des` | 0 | string | Object designation |
| `orbit_id` | 1 | string | Orbit-solution id for this approach |
| `jd` | 2 | float | Julian date of the close approach |
| `cd` | 3 | string | Calendar date-time, `%Y-%b-%d %H:%M` (e.g. `2020-Jan-01 02:06`), not ISO |
| `dist` | 4 | float | Nominal geocentric approach distance (AU) |
| `dist_min` | 5 | float | Lower 3-sigma bound of the approach distance (AU) |
| `dist_max` | 6 | float | Upper 3-sigma bound of the approach distance (AU) |
| `v_rel` | 7 | float | Relative velocity at approach (km/s) |
| `v_inf` | 8 | float | Velocity at infinity (km/s) |
| `t_sigma_f` | 9 | string | Formatted 3-sigma time uncertainty (`"< 00:01"`, `"13:02"`, `"2_09:08"`) |
| `h` | 10 | float | Absolute magnitude H |
| `diameter` | 11 | float | Diameter estimate (km); present only where JPL has one (~98.5% empty) |
| `diameter_sigma` | 12 | float | 1-sigma diameter uncertainty (km); same sparsity as `diameter` |

`t_sigma_f` is turned into seconds by `parse_t_sigma` (`days*86400 + hours*3600 + minutes*60`,
leading `< ` ignored); unparseable or `"none"` returns `None`, never a guessed zero. The
positional numeric columns (`jd, dist, dist_min, dist_max, v_rel, v_inf, h, diameter,
diameter_sigma`) are coerced to float; `diameter` is 98.521% empty and `diameter_sigma` 98.626%
empty in the raw feed, `v_inf` 0.025% and `h` 0.015% empty.

### Object-level aggregation (`features/asteroids._aggregate_one`)

Each designation's events collapse to one row. The **closest event** is the one with the smallest
numeric `dist_min` (strict `<`, so the earliest such row wins a tie); if no event has a numeric
`dist_min` there is no closest event and the distance-derived fields become `None`. `LABEL_COLUMN`
is `pha_flag = bool(group["pha_flag"].any())`.

| Output column | How it is produced | Semantics | Missing at object level |
| --- | --- | --- | ---: |
| `moid_proxy_au` | **min** numeric `dist_min` over the object's events | closest-approach distance proxy (AU) | 0.00% |
| `n_approaches` | `float(len(group))` | event count | 0.00% |
| `n_approaches_within_lunar` | count of events with numeric `dist_min <= 0.00257` | approaches inside 1 lunar distance | 0.00% |
| `v_rel_at_closest` | `v_rel` of the **closest event** | point value, not an aggregate (km/s) | 0.00% |
| `v_inf_at_closest` | `v_inf` of the **closest event** | point value (km/s) | 0.03% |
| `t_sigma_seconds_at_closest` | `parse_t_sigma(t_sigma_f)` of the **closest event** | timing uncertainty at closest (s) | 0.00% |
| `uncertainty_ratio_at_closest` | `dist_min / dist_max` at the **closest event** (None if `dist_max <= 0` or either NaN) | relative positional uncertainty | 0.00% |
| `closest_approach_month` | month of the closest event's `cd`; else median of all events' months (`_seasonal_fallback`) | 1..12, feeds the seasonal encoding | 0.00% |
| `absolute_magnitude_h` | **median** numeric `h` over events | object brightness proxy | 0.02% |
| `diameter_reported_km` | **median** numeric `diameter` over events | JPL-reported diameter (km) | 98.47% |
| `diameter_from_h_km` | `1329/sqrt(0.14) * 10^(-H_median/5)` from median H | diameter implied by H at assumed albedo | 0.02% |
| `diameter_inconsistency_km` | `abs(diameter_reported_km - diameter_from_h_km)` (None if either missing) | reported-vs-implied disagreement | 98.47% |
| `implied_albedo` | `(1329/diameter_reported_km)^2 * 10^(-0.4 * H_median)` (None if either missing) | albedo implied by reported diameter | 98.47% |
| `n_orbit_solutions` | count of distinct non-null `orbit_id` values | tracking/data-quality signal | 0.00% |
| `observation_span_years` | `(max(jd) - min(jd)) / 365.25` if more than one numeric `jd`, else `0.0` | baseline of the approach record (yr) | 0.00% |

The `~98.5% missing` columns are `diameter_reported_km`, `diameter_inconsistency_km` and
`implied_albedo`. They inherit the sparsity directly from the raw `diameter` event column
(98.521% empty): CAD returns a diameter only for the small subset of objects JPL has a physical
radar/thermal estimate for, the median of an all-empty set is `None`, and the latter two columns
each require that reported diameter, so they go missing together.

---

## Derived feature contract

Both sides (training and serving) import these lists from the same modules, so the model file is
the authority on order. `registry.load_artifact` refuses any model whose recorded feature list
differs from the live one, which is the train/serve-skew guard.

### Exoplanet features (21) - `features/exoplanets.FEATURE_COLUMNS`

Model `feature_hash = 41939f75dc829393`, `n_features = 21`. Constants read from code:
`SOLAR_RADIUS_IN_EARTH_RADII = 109.25`. `_log10_positive` returns `NaN` for non-positive or
blank inputs (a survey blank stays a blank, and gradient boosting uses it).

| Feature | Formula / derivation (verbatim) | Physical meaning | Missing-value behaviour |
| --- | --- | --- | --- |
| `log_period` | `_log10_positive(koi_period)` | log10 orbital period (days) | NaN if `koi_period <= 0` or blank |
| `log_duration` | `_log10_positive(koi_duration)` | log10 transit duration (hours) | NaN if `<= 0` or blank |
| `log_depth` | `_log10_positive(koi_depth)` | log10 transit depth (ppm) | NaN if `<= 0` or blank |
| `log_prad` | `_log10_positive(koi_prad)` | log10 planet radius (Earth radii) | NaN if `<= 0` or blank |
| `prad_inconsistency` | `(koi_prad - sqrt((koi_depth*1e-6).where(koi_depth>0)) * koi_srad * 109.25).abs()` | catalogue radius vs radius implied by depth (Earth radii) | NaN when depth `<= 0` or any input NaN |
| `duration_over_period` | `(koi_duration / koi_period).where(koi_period > 0)` | transit-duration fraction of orbit | NaN when period `<= 0` |
| `impact` | `koi_impact` (passthrough) | impact parameter b | NaN if blank |
| `eccentricity` | `koi_eccen` (passthrough) | orbital eccentricity | NaN if blank |
| `inclination` | `koi_incl` (passthrough) | inclination (degrees) | NaN if blank |
| `log_sma` | `_log10_positive(koi_sma)` | log10 semi-major axis (AU) | NaN if `<= 0` or blank |
| `equilibrium_temp` | `koi_teq` (passthrough) | equilibrium temperature (K) | NaN if blank |
| `stellar_teff` | `koi_steff` (passthrough) | stellar effective temperature (K) | NaN if blank |
| `log_stellar_radius` | `_log10_positive(koi_srad)` | log10 stellar radius (solar radii) | NaN if `<= 0` or blank |
| `stellar_mass` | `koi_smass` (passthrough) | stellar mass (solar masses) | NaN if blank |
| `stellar_logg` | `koi_slogg` (passthrough) | stellar surface gravity (dex) | NaN if blank |
| `stellar_density` | `(koi_smass / koi_srad.pow(3)).where(koi_srad > 0)` | mean stellar density proxy (Msun/Rsun^3) | NaN when radius `<= 0` |
| `n_transits` | `koi_num_transits` (passthrough) | observed transit count | NaN if blank |
| `model_snr` | `koi_model_snr` (passthrough) | model SNR | NaN if blank |
| `max_single_event_snr` | `koi_max_sngle_ev` (passthrough) | strongest single-event SNR | NaN if blank |
| `kepler_mag` | `koi_kepmag` (passthrough) | Kepler magnitude | NaN if blank |
| `koi_count` | `koi_count` (passthrough) | candidates sharing the star (confound, kept) | NaN if blank |

### Asteroid features (19) - `features/asteroids.FEATURE_COLUMNS`

Model `feature_hash = b11f6f011dd5c80d`, `n_features = 19`. Constants read from code:
`LUNAR_DISTANCE_AU = 0.00257`, `DIAMETER_CONSTANT_KM = 1329.0`, `DAYS_PER_YEAR = 365.25`,
`default_geometric_albedo = 0.14`. Five columns are computed in `derive()`
(`DERIVED_COLUMNS = {log_moid, moid_lunar_distances, log_diameter_from_h, closest_month_sin,
closest_month_cos}`); the other fourteen are the `BASE_INPUT_COLUMNS` produced by the
aggregation above.

| Feature | Formula / derivation | Physical meaning | Missing-value behaviour |
| --- | --- | --- | --- |
| `moid_proxy_au` | base: min `dist_min` over events | closest approach in window (AU) | None if no numeric `dist_min` |
| `log_moid` | derived: `_log10_positive(moid_proxy_au)` | log10 MOID proxy | NaN if `moid_proxy_au <= 0` or NaN |
| `moid_lunar_distances` | derived: `moid_proxy_au / 0.00257` | closest approach in lunar distances | NaN propagates |
| `n_approaches` | base: event count | approaches recorded for the object | always present |
| `n_approaches_within_lunar` | base: count `dist_min <= 0.00257` | sub-lunar approaches | always present |
| `v_rel_at_closest` | base: `v_rel` at closest event | relative velocity (km/s) | None if no closest event |
| `v_inf_at_closest` | base: `v_inf` at closest event | velocity at infinity (km/s) | ~0.03% None |
| `uncertainty_ratio_at_closest` | base: `dist_min/dist_max` at closest event | positional uncertainty fraction | None if `dist_max <= 0` or NaN |
| `absolute_magnitude_h` | base: median `h` | object absolute magnitude | ~0.02% None |
| `diameter_from_h_km` | base: `1329.0/sqrt(0.14) * 10^(-H_median/5)` | diameter implied by H at albedo 0.14 (km) | None if H missing |
| `log_diameter_from_h` | derived: `_log10_positive(diameter_from_h_km)` | log10 of the above | NaN if `<= 0` or NaN |
| `diameter_reported_km` | base: median `diameter` | JPL-reported diameter (km) | ~98.5% None (sparse feed) |
| `diameter_inconsistency_km` | base: `abs(diameter_reported_km - diameter_from_h_km)` | reported-vs-implied gap (km) | ~98.5% None |
| `implied_albedo` | base: `(1329/diameter_reported_km)^2 * 10^(-0.4*H_median)` | albedo implied by reported diameter | ~98.5% None |
| `n_orbit_solutions` | base: distinct non-null `orbit_id` count | orbit re-fit count (data quality) | always present |
| `t_sigma_seconds_at_closest` | base: `parse_t_sigma(t_sigma_f)` at closest event | 3-sigma timing uncertainty (s) | None if unparseable |
| `observation_span_years` | base: `(max(jd)-min(jd))/365.25`, else `0.0` | approach-record baseline (yr) | 0.0 for single-event objects |
| `closest_month_sin` | derived: `sin(2*pi*(month-1)/12)` where `1 <= month <= 12` | seasonal (month) encoding, sine | NaN if month out of range |
| `closest_month_cos` | derived: `cos(2*pi*(month-1)/12)` where `1 <= month <= 12` | seasonal (month) encoding, cosine | NaN if month out of range |

---

## Labels and class conventions

- **Exoplanets (3-class, no designated positive).** Contract order is
  `configs/settings.yaml` `exoplanets.labels = [CONFIRMED, CANDIDATE, FALSE POSITIVE]`, and the
  sidecar `labels` field stores that order verbatim. Selection metric is macro-F1, so no single
  class is "positive". At runtime the sklearn estimator's `classes_` array - which orders the
  probability columns and the API's `classes_order` - is the **alphabetical** sort
  `[CANDIDATE, CONFIRMED, FALSE POSITIVE]`. `_score_predictions` scores against `classes_`, not
  the config order; using the config order for the probability matrix would silently mispair
  classes.
- **Asteroids (binary, negative-first).** Contract order
  `asteroids.labels = [NOT_PHA, PHA]`. Index **0 is the negative label** (`NOT_PHA`) and index
  **1 is the positive label** (`PHA`). `features/asteroids.build_matrices` maps `pha_flag True ->
  labels[1]` (PHA), `False -> labels[0]` (NOT_PHA); `PHARuleClassifier` sets
  `classes_ = [negative_label, positive_label] = [NOT_PHA, PHA]`; metrics use
  `positive = labels[1] = "PHA"`, `negative = labels[0] = "NOT_PHA"`. Because sklearn sorts the
  fitted `classes_` alphabetically to `[NOT_PHA, PHA]`, the last probability column is the PHA
  score - the negative-first ordering holds in both the config and the runtime array here.
- **API convention.** `deps.predict` returns `probabilities` and `classes_order` zipped against
  `estimator.classes_` (the trained model's own order). For the binary asteroid task that order
  is negative-first (`NOT_PHA` then `PHA`), matching the config; for the exoplanet task the
  response lists classes in `classes_` order (`CANDIDATE`, `CONFIRMED`, `FALSE POSITIVE`) even
  though the sidecar contract order is the config order.

---

## What is NOT in this repository and why

- **`data/raw/*` and `data/processed/*` are gitignored** (see `.gitignore`), except the `.gitkeep`
  placeholders, because they are regenerable with `nasa-space-ml fetch`. `git ls-files data`
  tracks only `data/raw/.gitkeep` and `data/processed/.gitkeep`; `git check-ignore` confirms the
  three snapshot files and `checksum.json` are excluded. The bytes on disk in this working copy
  are exactly the snapshot this dictionary documents (fingerprints above).
- **Models and reports ARE committed.** `models/*.joblib` and `models/*.json` plus `reports/**`
  ship so every published number (metrics, CV table, feature ranges, `dataset_fingerprint`) stays
  checkable without re-downloading NASA data.
- **No secrets are needed.** Neither source requires an API key: the Exoplanet Archive TAP endpoint
  and the SBDB CAD endpoint are open HTTP GET services. The only outbound credential-like value is
  the courtesy `user_agent` string `nasa-space-ml/0.1 (capstone project; contact via GitHub)`.

---

## Reproducing the snapshot

- **Fetch:** `nasa-space-ml fetch` downloads both sources through `data/exoplanets.download` and
  `data/asteroids.download`, writing `koi_q1_q17_dr25.csv`, `cad_approaches.csv`,
  `cad_pha_des.txt` into `data/raw/` and recording each in `data/raw/checksum.json`. (The
  convenience alias `nasa-space-ml all` fetches then trains.)
- **Offline:** `nasa-space-ml --offline fetch` skips the network and reads only what is already
  cached; `--offline train` trains from the cached snapshot alone. This is how the pipeline runs
  end-to-end in CI without depending on NASA being reachable.
- **`dataset_fingerprint`:** computed by `data/cache.Cache.dataset_fingerprint()`. It is not a hash
  of the raw bytes directly - it is a content-addressed manifest digest. On each atomic write
  (`write_bytes` -> temp file + `os.replace`), `_record` stores the file's `sha256`, byte count and
  `written_utc` in `checksum.json`. The fingerprint then runs `hashlib.sha256` over the manifest
  entries in **sorted name order**, feeding each entry's `name` and its recorded `sha256` hex into
  the digest, and returns the final hexdigest. For this snapshot the three sorted entries
  (`cad_approaches.csv`, `cad_pha_des.txt`, `koi_q1_q17_dr25.csv`) yield
  `f732fe257e31929eae5bc08409ce59168df76df429911c30a0b949bca38eedbc` - the value stamped on both
  sidecars. Regenerating it after a `fetch` and comparing to that string is the fastest way to tell
  whether your data matches the shipped numbers; `verify()` also fails loudly if a cached file was
  hand-edited after it was written.
