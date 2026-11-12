# Error catalogue

Every non-2xx answer from this service is an RFC 9457 problem document. Nothing here is a
bare stack trace: the API layer maps the package's exception hierarchy onto problem
documents in exactly one place, `src/nasa_space_ml/api/problem.py`, so a route cannot forget
to be informative and a caller never has to parse an HTML page.

```http
HTTP/1.1 503 Service Unavailable
content-type: application/problem+json
x-request-id: 7f3c1e9b2a4d80c5

{
  "type": "https://github.com/Yuvraj0967/nasa-space-ml/blob/main/docs/errors.md#not-trained",
  "title": "ModelNotTrained",
  "status": 503,
  "detail": "no trained artifact for 'asteroid_pha': ... run `nasa-space-ml train --task asteroids`",
  "instance": "/api/v1/asteroids/predict",
  "code": "ModelNotTrainedError",
  "request_id": "7f3c1e9b2a4d80c5",
  "errors": {"model": "asteroid_pha"}
}
```

The `type` value is RFC 9457's **document identifier**. It is not an XML namespace and
nothing is required to fetch it — but it is written as a real, clickable URL to this page
on `main`, because a reviewer staring at a 503 should be one click from the explanation
rather than one from a 404. The anchor is the `code` field with `Error` removed and
lower-cased, so `InvalidFeaturesError` → `#invalid-features`. That sentence is not a
promise anyone has to keep by hand: `tests/test_api.py` walks `TYPE_URIS` and asserts
both that each URI is the clickable `/blob/main/` form and that the anchor it names
exists in this file.

`request_id` appears in the `X-Request-Id` response header as well as the body. Send your
own (≤ 64 characters, any token you like) and the service echoes it back, which is what
makes a screenshot from a review session traceable to a line in the server log. A longer or
absent value gets a fresh 16-hex-digit id generated instead.

## Summary

| `code` | HTTP | Meaning | Fix |
| --- | --- | --- | --- |
| `RequestValidationError` | 422 | body fails the Pydantic field contract | read `errors[].field` |
| `InvalidFeaturesError` | 422 | body is valid JSON but not a valid model row | [below](#invalid-features) |
| `ConfigError` | 500 | settings.yaml missing/unreadable/incomplete | [below](#config) |
| `DataError` | 502 | a NASA artifact or cache entry could not be used | [below](#data) |
| `UpstreamNASAError` | 502 | NASA answered with an error status or a surprise shape | [below](#upstream) |
| `ModelNotTrainedError` | 503 | service is up, artifact is not | [below](#not-trained) |
| `NasaSpaceMlError` | 500 | an internal error with no more specific class | report it |
| *(FastAPI default)* | 404 | unknown path, or no console page mounted | check the path |

One deliberate asymmetry: 422 means *you* can fix it by changing what you sent; 5xx means
nothing you change about the request will help. The mapping in `STATUS_BY_CODE` keeps that
promise, and there are no 4xx codes invented for conditions the caller cannot act on.

## <a name="config"></a>`ConfigError` — 500

`configs/settings.yaml` is missing, unreadable, or missing a key the code cannot operate
without. Raised at app construction, not at request time, so in practice you see it as a
startup traceback from `create_app()` rather than an HTTP response — the exception handler
that renders it exists so that a config problem surfacing *late* (a route reading a value
it expected to be there) still produces a problem document.

The usual causes, in rough order of how often they happen:

- `NSM_CONFIG` points at a path that does not exist (containers set this variable; see
  `resolve_config_path` in `api/app.py`).
- The YAML parses but a required block is absent — `paths`, `exoplanets.labels`,
  `asteroids.pha_distance_au`.
- The labels list has the wrong length or the wrong set of values for the task. The
  registry treats a changed label contract as a config error rather than retraining
  silently, because a label set is the one thing that invalidates every stored metric.

The message names the missing key. There is no default value for anything that affects a
score, on purpose: an application that quietly substituted a fallback threshold would
still return 200s, and they would be wrong ones.

## <a name="data"></a>`DataError` — 502

Something that should have been usable was not. Most often the local cache rather than
NASA:

- the raw file for a source is not cached → the detail names the command to run
  (`nasa-space-ml fetch`), and this is the 502 you get on `/api/v1/exoplanets/sample` in a
  fresh clone or a published container, where `data/raw/` is intentionally empty;
- a cached artifact's recorded byte size or sha256 does not match the file on disk;
- a CSV parsed to zero usable rows, or to rows missing the label column;
- a model sidecar whose JSON is not the shape `registry.load_metadata` expects.

It is a 502 and not a 500 because the failure is upstream of the application's own logic —
the data layer is a dependency the request could not reach — but it is not necessarily
NASA's fault. Check `nasa-space-ml status` first: it prints the manifest, the byte counts,
and `verified=` for every artifact.

## <a name="upstream"></a>`UpstreamNASAError` — 502

A subclass of `DataError` reserved for "the remote endpoint actually answered badly": a
non-2xx status after the retry policy was exhausted, a body that is not CSV/JSON, a TAP
error page where data was expected, or a `signature.version` that is not the `1.5` declared
in `configs/settings.yaml`.

The `detail` carries the status code and, for TAP, the first line of the error text —
which is how `ORA-00933` (a `LIMIT` clause that endpoint rejects) and `ORA-00904` (an
unknown column like `hostname`) got diagnosed in the first place without guessing.

Before assuming NASA is down, run:

```bash
.venv/bin/python scripts/probe_live.py     # fetches both sources, builds both matrices
```

Retry policy lives in `configs/settings.yaml` under `http:` — `max_attempts`,
`backoff_base_seconds`, `backoff_cap_seconds`, `timeout_seconds`. A request that fails this
way has already been retried that many times with capped exponential backoff, so retrying
it yourself immediately is usually the wrong instinct.

## <a name="not-trained"></a>`ModelNotTrainedError` — 503

The process is healthy and the endpoint exists; the artifact does not, or failed to load.
`/api/v1/health` will tell you which: it reports per-model state rather than a single
overall green light, because "the web page loads" is a useless answer when only one of two
models is broken.

Three distinct situations:

1. **Never trained.** `models/<name>.joblib` does not exist. The detail string is literally
   the command: `nasa-space-ml train --task exoplanets` (or `asteroids`, or `both`). This is
   the expected state of a fresh clone that has not run training; the committed artifacts
   are in git, so it only appears if you deleted them or set `paths.model_dir` elsewhere.
2. **Refused at load.** The sidecar's `scikit_learn` version does not match the installed
   one in a way joblib can survive. Artifacts round-trip across minor releases, not major
   ones, and a sidecar that records `"unknown"` fails *closed* — an unverifiable artifact is
   not served. This looks like a 503 with a version message, and the fix is to install the
   pinned version or retrain.
3. **Corrupt binary.** The `.joblib` exists but will not deserialize. Retrain.

A fourth, adjacent case worth naming: a *sidecar* that is valid JSON but whose
`feature_names` disagree with the code's live contract raises `DataError` at startup. The
app still boots, the model is listed in `model_errors`, and the affected routes answer 503
while the rest keep working. That rough edge is documented in
[docs/LEARNING_GUIDE.md](docs/LEARNING_GUIDE.md) chapter 14 and covered by
`tests/test_registry.py`; the mitigation is that `initialise()` catches everything the
model layer can throw, so a bad artifact degrades two routes instead of killing the process.

## <a name="invalid-features"></a>`InvalidFeaturesError` — 422

The request parsed; it cannot be made into a feature vector. Raised by the same guard in
`api/deps.py` on both sides of the trained-artifact boundary, and it is the mechanism that
makes train/serve skew impossible to miss:

```
"detail": "live feature list does not match the artifact 'asteroid_pha' (hash a1b2…):
           missing_in_model=['implied_albedo'], extra_in_model=['implied_eccentricity'].
           Re-run `nasa-space-ml train` to rebuild the artifact."
```

The `errors` block carries `missing_in_model`, `extra_in_model`, `live_hash`, and
`model_hash`. The hash is an order-sensitive digest of the feature list — `feature_list_hash`
in `models/registry.py` — because column order matters to a model that was fit on a matrix,
and a list hash that ignored order would let a silently reordered contract through.

You cannot reach this error by sending a bad body. `RequestValidationError` catches those
first. It is reachable only by *changing the code* without retraining, which is exactly why
it is a 422 rather than a 500: the caller's request is fine, and the repository is in a
state that has to be fixed.

## `RequestValidationError` — 422

FastAPI's own validation failure, re-expressed as a problem document instead of the default
`{"detail": [...]}` so that every error on this API has one shape:

```json
{
  "code": "RequestValidationError",
  "title": "Invalid request body",
  "errors": {
    "fields": [
      {"field": "koi_prad", "in": "body",
       "message": "Input should be less than or equal to 100000",
       "type": "less_than_equal", "input": 50000000}
    ]
  }
}
```

The bounds are not arbitrary. They come from the physical quantities the columns represent —
a transit depth cannot exceed 100 % of a star's light, an inclination is a real angle, a
month is 1–12 — so a value outside them is a units mistake in the caller, not an unusual
observation. `extra="forbid"` on every request model means an unexpected key is also a 422:
a silently ignored `koi_pdisposition` would be a label fed to the model wearing the costume
of a feature, and the schema is not going to permit that by oversight.

## 404 and the mounted console

`/` serves the console only when a `web/` directory is actually found — the env override
`NSM_WEB_DIR`, then the package-relative path, then `./web`. A `pip install` into a
site-packages tree with no page present still serves the whole API, and `/` answers 404
rather than crashing at import. Any unknown path gets FastAPI's normal 404; it is not
converted to a problem document, because it is not an error the application raised.

## What is deliberately not here

No 401/403 — there is no authentication, because this is a capstone served on a laptop and
an unauthenticated demo is an honest demo. No 429 — there is no rate limiter; adding one
would be theatre without a deployment to protect. No 500 with a traceback: `exception
handlers` swallow the stack for the client and the server keeps it, which is the only
ordering that is ever right.
