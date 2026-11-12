# Learning guide

This is the teaching document for `nasa-space-ml`. It is not a summary of the repository and
not a description of the API — those are elsewhere — it is an attempt to explain every
concept the project rests on, in the order a reader meets them, using this code as the worked
example. Each chapter ends with **check yourself**: questions a reviewer might ask, answerable
from what came before them.

Who it is for: someone who can write Python and has met basic statistics, and who has not
yet had to make an ML system that another person can run. The student who built this needs to
defend every line of it out loud, and that is the standard each chapter is written against —
not "what does the docs site say", but "what would I say when asked *why*".

Read it in order. The chapters follow the data pipeline, not difficulty: chapter 4 (caching)
is easier than chapter 9 (metrics), and the reason they are in that order is that the cache
decision constrains everything downstream.

**What this project claims.** Two classifiers trained on public NASA data, with grouped
splits, pre-declared selection metrics, rule baselines that include NASA's own published
definition, honest holdout numbers (macro-F1 0.7687; average precision 0.7261), a service that
serves them with typed contracts and RFC 9457 errors, a test suite of 371 cases that never
touches the network, and CI that boots the thing. **What it does not claim.** That either model
is state of the art; that a PHA prediction is a hazard assessment; that the asteroid MOID
proxy is a MOID; that 0.77 macro-F1 makes a validation scientist redundant. Those limits are
not buried in a concluding paragraph — they are in the chapter that owns the concept.

Conventions: file paths are relative to the repository root. Code blocks marked *quoted* are
copied from this repository; anything else is illustrative. Numbers come from
`reports/exoplanets/metrics.json`, `reports/asteroids/metrics.json`, or the two sidecars in
`models/`, and the filename is named whenever a number appears. Column-level detail — every
field, its units, its missingness — lives in [DATA_DICTIONARY.md](DATA_DICTIONARY.md), which
this guide deliberately does not duplicate.

---

## Chapter 1 — Two NASA questions, one service

### The shape of the problem

Most tutorial ML projects start from a CSV someone else cleaned. That choice quietly deletes
half of the engineering: if the file is on disk, you never have to decide whether the data is
current, whether the download was truncated, whether a column changed name, or whether the
number you are quoting describes bytes anyone else can obtain. This project starts one step
earlier, from two live NASA endpoints, and keeps that step visible in the published result.

The two problems were chosen to be *different in kind*, not just in content:

| | Exoplanets | Asteroids |
| --- | --- | --- |
| Question | Will the KOI team validate this candidate? | Is this object potentially hazardous? |
| Target type | 3-class | binary |
| Class balance | 49 / 34 / 17 % | 94.7 / 5.3 % |
| Natural metric | macro-F1 | average precision |
| Baseline worth beating | majority-class dummy | NASA's published rule |
| Unit of independence | host star (`kepid`) | object (`des`) |
| Raw rows vs units | 8,054 KOIs / 6,923 stars | 32,390 events / 22,581 objects |

Four axes vary — target arity, balance, baseline, and the grain of the data — and everything
that is hard about ML shows up somewhere in that grid. Imbalance forces a metric conversation
(chapter 9). A published rule forces a baseline conversation (chapter 10). Repeated entities
force a splitting conversation (chapter 8). A label that has a look-alike twin in the same
table forces a leakage conversation (chapter 5). Building both in one repository, sharing the
data, feature, training, registry, and API layers, is what makes the shared structure visible
instead of implicit.

Why not three or four problems? Because a second dataset stresses the abstractions and a fifth
only repeats them, and because the honest cost of each additional model is a full training
run, a report suite, a model card, and a set of numbers someone must be able to check. Two is
enough to prove the pipeline generalises and few enough to finish properly.

### What "end to end" means here

Ten stages, each of which can fail in a way the previous stage cannot see:

1. **Fetch** — an ADQL query over HTTPS to the Exoplanet Archive; a paginated walk of JPL's
   close-approach API. (`data/exoplanets.py`, `data/asteroids.py`)
2. **Cache** — content-addressed, checksummed, atomic. (`data/cache.py`)
3. **Feature** — physics-derived columns, a leakage blacklist, one exported `FEATURE_COLUMNS`
   tuple per task. (`features/`)
4. **Split** — group-disjoint, stratified, deterministic. (`models/splits.py`)
5. **Compare** — rules and learned models in one list, scored by one loop, on a metric chosen
   in advance. (`models/train.py`, `evaluation/metrics.py`)
6. **Persist** — a joblib binary plus a JSON sidecar that answers questions without loading
   the binary. (`models/registry.py`)
7. **Serve** — a FastAPI app with typed contracts, provenance in every response, and errors as
   RFC 9457 documents. (`api/`)
8. **Show** — a browser console that calls the API and never recomputes a prediction. (`web/`)
9. **Verify** — 371 tests, mocked transports, warnings-as-errors, and three opt-in live probes
   against NASA. (`tests/`)
10. **Publish** — CI that lints, tests on two Python versions, boots the service, scores a real
    request over HTTP, and builds the container. (`.github/workflows/ci.yml`)

The definition of "end to end" this project uses is that **stage 10 can be run by a stranger
on a stranger machine and produces a prediction from stage 1 data**. A notebook with a good
score is not end to end; it is stage 5 of 10 with the hard parts deleted.

### The repo map

```
configs/settings.yaml     every tunable: URLs, columns, labels, thresholds, seed, retry policy
src/nasa_space_ml/
  config.py               YAML -> frozen dataclasses; refuses a missing key instead of defaulting
  errors.py               the whole exception hierarchy, five classes
  data/                   http.py (retry/backoff), cache.py (checksums), one module per source
  features/               derive(), FEATURE_COLUMNS, LEAKAGE_FORBIDDEN, frame_from_payload()
  models/                 splits.py, baselines.py, train.py, registry.py
  evaluation/             metrics.py (selection + operating points), reports.py (figures)
  api/                    app.py (factory), deps.py (state + contract guard), problem.py, routers
  cli.py                  fetch | train | status | serve | all
web/                      index.html, styles.css, app.js — no framework, no CDN
models/                   *.joblib + *.json sidecars      [committed]
reports/                  metrics.json + PNG figures       [committed]
data/raw/                 NASA bytes + checksum.json       [gitignored]
tests/                    371 cases, one file per module
docs/                     this guide, ARCHITECTURE, DATA_DICTIONARY, errors, model cards
scripts/                  live probes used during development, excluded from the wheel
```

The three directories that are committed against the usual convention — `models/`, `reports/`,
and (in git history) `docs/superpowers/` — each carry a reason. Artifacts ship because a
reviewer with ten minutes should see a prediction, not a training run. Reports ship because a
quoted number that cannot be opened is a number taken on faith. The design spec and
implementation plan ship because "how did you decide this" is a fair question and the answer
should be a document, not a memory.

### The layering rule that makes the rest survivable

Dependencies point one way, toward the data. `features/` does not import `models/`;
`models/` does not import `api/`; nothing imports `web/`. The full table, with the "may never"
column that matters more than the imports, is in
[ARCHITECTURE.md](ARCHITECTURE.md#layer-rules). What it buys concretely: a change to a metric
cannot silently alter what NASA column gets requested, and a change to the front end cannot
alter a score. The single module allowed to know about both halves of the system is
`api/deps.py`, which is exactly why the feature-contract guard lives there.

### Check yourself

*Why two models instead of one?* Because one dataset cannot demonstrate imbalance, metric
choice, rule baselines, and two different units of independence at once.

*What does "end to end" mean in this repository?* That `git clone` + `pip install` + `serve`
produces a live prediction, and that CI proves it.

*Where does a tunable number live?* `configs/settings.yaml`. If it lives in code, it is either
a constant with a unit (like `LUNAR_DISTANCE_AU`) or a bug.

---

## Chapter 2 — Talking to NASA: TAP/ADQL and the SBDB CAD API

### Two APIs, two philosophies

Both sources are public, need no key, and return data rather than a JSON envelope designed for
a mobile app. That is where the similarity ends.

**The Exoplanet Archive exposes a query language.** TAP (Table Access Protocol) is an IVOA
standard: you POST a query written in ADQL — Astronomical Data Query Language, essentially SQL
with a dialect for celestial coordinates — to `/TAP/sync`, and it hands back a table. The
request this project makes is a plain `select` of 21 named columns:

```sql
select kepoi_name, kepid, koi_disposition, koi_period, ... from Q1_Q17_DR25_KOI
```

built by `data/exoplanets.py::build_query`. The archive's DR25 KOI table has 153 columns;
asking for 21 is both a bandwidth decision and a documentation decision — the query is the
feature list, in public.

**JPL's CAD API exposes a resource.** `https://ssd-api.jpl.nasa.gov/cad.api.json?date-min=...
&date-max=...&dist-max=0.2&limit=5000&start=...` returns close-approach *events* as JSON, and
its payload is an array of arrays with a separate `fields` list naming the columns:

```json
{"signature": {"source": "NASA JPL Solar System Dynamics", "version": "1.5"},
 "fields": ["des", "orbit_id", "jd", "cd", "dist", "dist_min", ...],
 "data": [["1950 DA", "26", 2451545.0, "2 Mar 1950 12:07:00", ...], ...]}
```

Zipping `fields` onto each row is the whole parse. It also means the column *order* is not a
contract — the order can change between requests, so a parser that indexes positionally
breaks silently when NASA reorders a response. Everything in `data/asteroids.py` goes through
names.

### Why `version` is asserted rather than ignored

`_validate_signature` reads `signature.version` and compares it to
`asteroids.expected_api_version: "1.5"` in settings. A v2.0 would plausibly add fields, rename
them, or change `dist_min`'s definition — and every one of those changes would produce a
*successful HTTP response* that this pipeline would happily train on. An assertion that costs
one string comparison turns a silent statistical corruption into a loud `UpstreamNASAError`
at fetch time. That is the general principle: **validate the thing that would hurt most if it
drifted quietly.**

### Why the response format matters more than it sounds

TAP can return XML, VOTable, JSON, or CSV. CSV was chosen because pandas reads it in one call,
because a human can open it in a text editor during a debugging session, and because the
failure modes are legible: an error page from that endpoint arrives as HTML or as a text
message beginning `ORA-`, and both are easy to detect. The tradeoff is real — CSV has no type
system, so every column arrives as a string and must be coerced, which is chapter 7's problem,
and a value containing a comma would need quoting discipline NASA does provide but which is
one more thing to trust.

### Two errors that came from the live endpoints, not from guessing

While building the data layer, two ADQL mistakes produced real archive errors:

- `ORA-00933` when a `LIMIT` clause was included. That endpoint rejects it; the module
  docstring of `data/exoplanets.py` records the fact so nobody re-adds it. Row counts are
  controlled by the `select` list, not by a row limit.
- `ORA-00904` for a column named `hostname`, which the KOI table does not have — the
  discovery that forced the group key to be `kepid` rather than a star *name* (chapter 8),
  and which is written into `configs/settings.yaml` as a comment rather than remembered.

Both are the reason "read the documentation" is only half of it: documentation for a
twenty-year-old archive describes what the schema says, and the actual dialect of the running
server is only discoverable by sending a request and reading the reply. Those comments in the
repository are the residue of that process, and they are more valuable than a paragraph of
prose about API care.

### Being a client of a government archive

Neither endpoint publishes a rate limit, which means the polite answer is: ask for what you
need in as few requests as possible, cache aggressively so a second run asks for nothing, page
large queries instead of demanding them whole, identify yourself with a real User-Agent, and
back off when the server says it is busy. All five appear in this repository — the first
because the ADQL query is a single request for the whole table, the rest in chapter 3.

### Check yourself

*What is ADQL?* SQL-like query language for astronomical tables, used over TAP.

*Why assert `signature.version`?* Because a renamed or redefined field returns HTTP 200 and
corrupts training silently; the assertion converts that into a startup failure.

*Why CSV?* One-call parsing, human-readable during debugging, and detectable error bodies —
costing a type system, so every column needs explicit coercion.

*Why `kepid` and not `hostname`?* The KOI table has no `hostname` column; the archive said so
with ORA-00904, and `kepid` is the star-level identifier that survives that constraint.

---

## Chapter 3 — Being a good client: timeouts, retries, backoff, pagination

### Every network call needs a deadline

`configs/settings.yaml` sets `timeout_seconds: 60`, and `data/http.py::make_client` passes it
to `httpx.Client`. Without a timeout a request to a saturated archive does not fail — it
*hangs, indefinitely, holding a database connection or a server worker or a training run
open while nothing happens. The failure mode of no-timeout code is not an error message; it is
a system that appears to be working and is not.

Sixty seconds is a choice, not a constant. The KOI CSV is 1.2 MB; on a laptop connection that
is seconds, so 60 s is generous enough to survive a slow evening and short enough that a hung
socket is detected by a human before they go to lunch. A different number would be needed for
a different payload, and that is the reasoning that produces a number.

### Retrying is about distinguishing two kinds of failure

`_attempt_fetch` (quoted, abridged) is the whole policy:

```python
for attempt in range(1, settings.max_attempts + 1):
    try:
        response = client.get(url, params=params)
    except httpx.HTTPError as exc:
        last_error = f"{type(exc).__name__}: {exc}"
    else:
        if response.status_code in RETRY_STATUS:
            last_error = f"HTTP {response.status_code}"
        elif response.is_error:
            # 4xx other than 429 will not fix itself; do not waste retries.
            raise UpstreamNASAError(...)
        else:
            return on_response(response)

    if attempt < settings.max_attempts:
        backoff = min(
            settings.backoff_cap_seconds,
            settings.backoff_base_seconds * (2 ** (attempt - 1)),
        )
        _sleep(backoff * (1.0 + random.random() * 0.25), sleeper)
```

Read it as a classification of failure into three buckets:

1. **Transport errors** (`ConnectError`, `ReadTimeout`, DNS) — the network is unreliable, the
   request probably never reached the server, retry.
2. **Retryable status codes** (`RETRY_STATUS` = 429 plus 5xx) — the server is alive and telling
   us to come back. Retry.
3. **Other 4xx** — the server understood us and refused. Our query is wrong, our column list
   is wrong, or we are not allowed. **Retry changes nothing**, so this raises immediately
   rather than burning four attempts and thirty seconds on a mistake that will not fix itself.

That third branch is where most hand-rolled retry logic goes wrong: retrying a 400 turns a
clear, immediate, actionable error into a vague slow one.

### Backoff, cap, and jitter

`backoff_base_seconds: 0.5` doubling gives 0.5, 1, 2, capped at `backoff_cap_seconds: 12`
across `max_attempts: 4` — so the worst-case sleep is about 3.5 s and then a hard failure. Two
details are worth being able to explain:

- **The cap** exists because pure exponential backoff on a 5-attempt policy would sleep 16 s,
  32 s, 64 s: longer than the outage, longer than the patience of anyone watching a terminal.
  A cap bounds the worst case, which is the property a reviewer asks about.
- **The jitter** — `1.0 + random.random() * 0.25` — is a 0–25 % multiplicative spread. The
  textbook reason for jitter is a *thundering herd*: many clients that failed simultaneously
  retry simultaneously, re-create the outage, and fail again in lockstep. With one client this
  is irrelevant. It is here anyway, cheaply, because "the retry policy is correct for the
  general case" is a better property than "correct for my laptop today", and 25 % full jitter
  costs nothing.

`Retry-After` is *not* honoured, and that is a known gap rather than an oversight: the
response header would say precisely how long to wait, and this code ignores it in favour of
its own schedule. For two endpoints that rarely emit 429, the simplification is defensible.
It is the first thing that would need fixing under real load, and chapter 15 lists it among
the things not tested.

### The sleeper is injectable, and that is the interesting part

`_attempt_fetch` receives `sleeper` and `data/http.py` defaults it to `time.sleep`. Every test
passes a function that records the durations and returns instantly. So the suite can assert
"four attempts, sleeping 0.5 s, 1.0 s, 2.0 s" in milliseconds rather than waiting four
seconds, and can assert that a 400 raises *without* sleeping at all. Backoff is one of those
behaviours you cannot test honestly without a seam — and if it is untested, it is very likely
wrong, because off-by-one errors in `2 ** (attempt - 1)` are the norm.

`tests/test_http.py` covers: success on attempt one; recovery after two failures; exhaustion
after four; a 400 raising immediately; a malformed body not being cached; the jitter staying
inside its stated bounds; and the exact sleep sequence.

### Pagination is a client-side loop over a stateless server

`data/asteroids.py::_page` walks the CAD endpoint with `limit=5000` and an advancing `start`
offset, up to `max_pages: 20`. The window asked for (2020-01-01 to 2031-01-01, `dist_max`
0.2 au) yields 32,390 events, so about seven pages.

Why not ask for everything? Because the server's default is small and an unbounded request is
both slower to fail and more likely to be refused. Why stop at 20? Because a loop that reads
"keep going until the server returns nothing" and has an off-by-one in the termination test
becomes an infinite loop that fills the disk. `max_pages` converts that class of bug into a
`DataError` with a number in it. It is a safety rail, not a feature: hitting it means the
window or the API changed, and the right response is a loud failure.

### `--offline` is a switch, not a hope

`config` and the data layer cooperate so that with `--offline` set, any path that would open a
socket raises instead. CI runs `python -m nasa_space_ml --offline status` as an assertion that
the package is importable and its configuration readable on a machine with no egress. The
value of that flag is architectural: it makes "the test suite cannot touch the network"
*enforceable* rather than a promise that decays as tests are added.

### Check yourself

*What are the three failure classes and what does each do?* Transport → retry; 429/5xx →
retry with backoff; other 4xx → raise now.

*Why cap exponential backoff?* Unbounded doubling sleeps longer than the outage and longer than
the user. The cap bounds worst-case latency.

*Why is `max_pages` there?* To turn a termination bug into a bounded, explicit failure. Twenty
is far above the seven the current window needs.

*How do you test a backoff schedule without waiting for it?* Inject the sleeper; assert the
durations.

---

## Chapter 4 — Cache, snapshot, fingerprint: making a number checkable

### The problem this solves

Twelve months from now, someone — possibly a teacher, possibly a future version of the author —
reads "test macro-F1 0.7687" in the README and wants to know if it is still true. The
information needed to answer that is: which bytes were the input, and which code turned them
into a model. A repository that cannot answer the first question has published an
unfalsifiable claim.

### A cache that knows what it contains

`data/cache.py` keeps three artifacts plus a manifest. The interesting part is the write path:

```python
handle = tempfile.NamedTemporaryFile(dir=self.directory, delete=False)
try:
    handle.write(payload)
    handle.flush()
    os.fsync(handle.fileno())
finally:
    handle.close()
os.replace(handle.name, target)
self._record(name, payload)
```

Four deliberate steps. Write to a *temporary file in the same directory* (so `os.replace`
stays within one filesystem and is therefore atomic). `flush` and `fsync` so the bytes are on
the platter before the rename, not in an OS buffer that a crash can eat. `os.replace` to
publish — a POSIX atomic rename, so any reader at any instant sees either the complete old
file or the complete new one, never a half. Then record the checksum.

Why is this worth four syscalls? Because the alternative failure mode is a truncated CSV that
*exists*, that `path.exists()` returns True for, that pandas happily reads — and that produces
a training set of 3,000 rows instead of 8,054 with no error anywhere. A pipeline that trusts
"the file is there" as evidence of "the file is complete" will eventually train on a partial
download and report a number derived from half the data. Atomic rename makes that state
unrepresentable.

`_record` writes `checksum.json`: per artifact a `sha256`, a `bytes` count, and a `written_utc`
timestamp. `read_bytes` then verifies before use:

```python
if not self.verify(name):
    raise DataError(
        f"cached artifact {name!r} does not match its recorded checksum; "
        "delete it and re-run `nasa-space-ml fetch`",
        details={"path": str(target)},
    )
```

so a hand-edit, a partial copy from a USB stick, or a disk hiccup becomes an error with a
command in it rather than a feature matrix.

### The fingerprint

```python
def dataset_fingerprint(self) -> str:
    entries = self.manifest()
    digest = hashlib.sha256()
    for name in sorted(entries):
        digest.update(name.encode("utf-8"))
        digest.update(entries[name]["sha256"].encode("utf-8"))
    return digest.hexdigest()
```

A hash over sorted (filename, content-hash) pairs. The sorted filenames matter: without them
two caches with the same set of content hashes under different names would collide, and the
names are part of what is being claimed. Sorting matters for the same reason the split tie-break
in chapter 8 does — determinism independent of dictionary iteration order.

This snapshot's value is
`f732fe257e31929eae5bc08409ce59168df76df429911c30a0b949bca38eedbc`. It appears in both
sidecars, in both `reports/*/metrics.json`, in `/api/v1/health`, in `/api/v1/meta`, and in the
README's citation block.

Two properties make it useful rather than merely clever. First, **it is cheap to compare**: one
hex string answers "did you get the same bytes" without shipping 6 MB. Second, **it is honest
when empty**: an empty cache hashes to `e3b0c44298fc1c14...` — the well-known SHA-256 of the
empty string — so a fresh clone with no data displays a fingerprint that obviously does not
match the artifacts, rather than a blank field a reviewer might read as "not applicable".

### What ships, what does not

`data/raw/*` is gitignored; `models/` and `reports/` are not. The reasoning is asymmetric on
purpose. NASA's bytes are not ours to redistribute, they regenerate in a couple of minutes
with a documented command, and committing them would make the repository's diff noise drown
the code. The *artifacts*, by contrast, cannot be regenerated in any meaningful sense — a
retrain on a later snapshot yields a different model, so the only way to make "here is the
model these numbers describe" true is to commit the model. The fingerprint is what lets the
reader check that the committed model and their own fresh fetch do or do not describe the same
world.

### Check yourself

*Why does the cache store checksums?* Because existence is not evidence of completeness, and
a truncated CSV trains silently on less data.

*What is a dataset fingerprint for?* Turning "is your number still reproducible" into a string
comparison, and stamping every published score with the bytes it came from.

*Why is `data/raw` gitignored while `models/` is committed?* One is redistributable and
regenerable; the other is neither. The fingerprint bridges the gap.

---

---

## Chapter 5 — Labels, and the art of leaking yourself into a great score

### What a label is, and where the answer is hiding

`koi_disposition` takes three values used by this project: `CONFIRMED`, `CANDIDATE`, `FALSE
POSITIVE`. It is the *outcome of a validation study* — the published verdict of a body of
work done over several years using photometry, centroiding, adaptive optics, and speckle
imaging. Learning to predict it is legitimate, useful, and the whole point of the model.

The KOI table has 153 columns, and some of them are not observations. They are *the vetting
pipeline's own intermediate conclusions*:

```python
LEAKAGE_FORBIDDEN: frozenset[str] = frozenset(
    {"koi_pdisposition", "koi_vet_stat", "koi_quarters", "koi_delivname"}
)
```

Quoted from `data/exoplanets.py`. Each is a piece of the answer wearing a different name:

| Column | What it actually encodes |
| --- | --- |
| `koi_pdisposition` | the pipeline's *preliminary* disposition — near-identical to the final label |
| `koi_vet_stat` | whether the row has been vetted at all |
| `koi_quarters` | which data releases the candidate appears in — a proxy for how long it was studied |
| `koi_delivname` | which delivery of the KOI table the row came from, i.e. its vintage |

Any one of them would produce a model with a CV score around 0.95 that has learned nothing
about planets. It would have learned the archive's bookkeeping.

### Where the guard lives, and why there

```python
def build_query(columns: tuple[str, ...] = BASE_COLUMNS, *, table: str) -> str:
    """Compose the ADQL select. Deliberately avoids `limit` - see module docstring."""
    if not columns:
        raise DataError("build_query needs at least one column")
    overlap = set(columns) & LEAKAGE_FORBIDDEN
    if overlap:
        raise DataError(
            f"refusing to request leakage-prone columns: {sorted(overlap)}",
            details={"columns": sorted(overlap)},
        )
```

The check is at the *query builder*, one level above feature engineering. This placement is
the entire design decision: a blacklist enforced in the feature layer protects the training
matrix and nothing else, while a blacklist enforced where the HTTP request is composed
protects every downstream consumer, including an exploratory notebook that calls `download()`
and gets a CSV that structurally cannot contain the poisoned columns. You cannot leak what was
never downloaded.

Three more layers, each catching a different mistake:

- `features/exoplanets.py::RAW_INPUT_COLUMNS` is an explicit allowlist; a column not listed
  never reaches `derive()`.
- `build_matrices` raises `DataError` naming missing columns rather than dropping them
  silently — a renamed NASA field is a contract break, not a NaN.
- `api/schemas.py` omits the label and every disposition column from the request model, and
  sets `extra="forbid"`. So `{"koi_pdisposition": "CONFIRMED"}` posted to `/predict` is a 422
  with `extra_forbidden`. The API *refuses to be told the answer*.

That last one is not decorative. A reviewer who types a real row's disposition into the
console and gets a confident prediction would otherwise conclude the model "works" while the
prediction was an echo.

### Why leakage is invisible in the score

This is the concept worth being able to say out loud. A leak is not an error; it is a feature
that correlates with the label because it *is* the label, or is derived from it. Cross-validation
detects overfitting — a model that memorised noise fails on held-out data. It cannot detect a
leak, because the leak is present in the held-out data too. Every metric goes up: accuracy,
AUC, Brier, calibration. The score is real. It describes a model that predicts the archive
from the archive.

The only defences are structural, and they all reduce to the same sentence: **never let a
feature be derived from anything you obtain after you know the answer.** Not from the label
column. Not from a table joined on the label. Not from a "cleaned" version of the dataset
someone produced after vetting. Not from the pipeline that assigns the labels.

And the reason it is worth this much machinery: leakage is the failure mode that survives
code review. It does not throw, does not log, does not make a plot look wrong. It makes the
plot look *good*.

### The asteroid analogue, which is subtler

`asteroids/` has no forbidden columns, because its label is not in the event table at all —
it comes from a separate artifact, the PHA designation list (`cad_pha_des.txt`), joined on
`des` in `load_events`. But there is a different trap, and it is the one that makes this
project's second model interesting.

If you build the label as `MOID <= 0.05 and H <= 22.0` and then feed `moid_proxy_au` and
`absolute_magnitude_h` as features, you get average precision 1.000 and a model that is a
wasted round trip — it re-implements the rule with extra steps. The label here is instead
what JPL's humans actually published, which depends on properly computed MOIDs from orbital
elements, on review, and on things these two columns do not capture. That gap between "the
rule on proxy inputs" and "the list JPL maintains" is the only reason the learned model has
anything to learn, and it is why the NASA rule baseline scores AP 0.2575 rather than 1.0
(chapter 10).

### Check yourself

*What is label leakage?* Using, as an input, information that only exists because the answer
is already known.

*Why does CV not catch it?* Because the leak is in the held-out fold too. Every metric rises.

*Why is the blacklist checked in `build_query` rather than in `derive`?* One level higher means
the forbidden columns never enter the repository, protecting every consumer, not just the
training matrix.

*What would the asteroid model be learning if its label were computed from MOID and H?*
Nothing — it would be an inefficient restatement of the rule. The label must come from the
independent published list.

---

## Chapter 6 — Features from physics, not from a feature store

### The principle

Adding columns because a library makes it easy is how projects end up with forty features, no
idea which matter, and a reviewer asking "why does this one exist?". Every engineered feature
in this repository answers a question an astronomer would recognise, and the docstring in
`features/exoplanets.py` states the standard: *"each one answers a question a validation
scientist actually asks."*

### Transit depth: the identity that does the heavy lifting

When a planet crosses its star, the fractional drop in brightness is the ratio of the two
disks' areas:

```
depth = (Rp / Rs)²
```

`koi_depth` is that quantity in parts per million; `koi_prad` is the catalogue's quoted planet
radius in Earth radii; `koi_srad` is the stellar radius in solar radii. So you can invert the
identity and ask a second, independent question about the same object:

```python
# Transit depth is a ratio of areas: depth = (Rp / Rs)^2, quoted in parts per million.
depth_ppm = pd.to_numeric(out["koi_depth"], errors="coerce")
stellar_radius = pd.to_numeric(out["koi_srad"], errors="coerce")
radius_ratio = (depth_ppm * 1e-6).where(depth_ppm > 0) ** 0.5
prad_from_depth = radius_ratio * stellar_radius * SOLAR_RADIUS_IN_EARTH_RADII
quoted_prad = pd.to_numeric(out["koi_prad"], errors="coerce")
out["prad_inconsistency"] = (quoted_prad - prad_from_depth).abs()
```

`SOLAR_RADIUS_IN_EARTH_RADII = 109.25` is the unit conversion the two radii need before they
can be compared — the kind of line that looks trivial and is where a factor-of-109 bug would
live. Note `.where(depth_ppm > 0)`: a zero or negative depth is not a small planet, it is a
missing or impossible measurement, and taking the square root of it must yield `NaN` rather
than a warning or a complex number.

The result is a *disagreement* feature. If the catalogued radius and the radius implied by the
depth differ substantially, the transit interpretation is wrong — which is precisely how an
eclipsing binary, a background blend, or a badly estimated stellar radius announces itself.
It is the fourth most important feature in the model: mean macro-F1 decrease 0.0338 across 8
permutation repeats (`reports/exoplanets/metrics.json`). A model given only the raw columns
would have to learn the identity `(Rp/Rs)² = depth` from data; given this feature, it is told.

`duration_over_period` (importance 0.0887, second-highest) is the other geometry ratio: how
long the transit lasts relative to how long the orbit takes. It is scale-free, and it is what
separates a grazing or long-duration event from a short, central one.

`stellar_density` = `smass / srad³` completes the set. It is derivable from Kepler's third law
given period and semi-major axis, and — more importantly for validation — a *wrong* stellar
density is one of the classic signatures of a blend, because the host star's classification
drives everything else.

### Log transforms, and why not just feed the raw numbers

```python
out["log_period"] = _log10_positive(out["koi_period"])
```

with a helper that turns non-positive or blank into `NaN` rather than raising. `koi_period`
runs from under a day to well over 1000; `koi_depth` spans orders of magnitude. Three
reasons to log them: a linear model can express a power-law relation as a straight line; the
residual structure becomes multiplicative, which matches how measurement error behaves on
these quantities; and a tree choosing a split at "one order of magnitude out" is a more
transferable decision than one at "3.7 days out". For `HistGradientBoostingClassifier` the
first two reasons do not apply — trees are invariant to monotone transforms of a single
feature, which is exactly why `log_moid` scores 0.0 importance in chapter 13 — but the
transforms stay because the *linear baseline* needs them to be a fair opponent, and a
comparison where one model is handicapped by scaling is not a comparison.

### The Harris–Tumbioli relation, and an assumption written down

For the asteroid side, the physical quantity that matters is size, and what the survey
actually measures is brightness. The standard conversion, used by JPL on the same pages this
project fetches, is

```
D_km = 1329 / sqrt(p_v) * 10^(-H/5)
```

implemented literally:

```python
DIAMETER_CONSTANT_KM = 1329.0


def diameter_from_absolute_magnitude(h: Any, albedo: float) -> float | None:
    """NASA's standard H -> diameter relation, D_km = 1329/sqrt(p) * 10^(-H/5)."""
    if albedo <= 0:
        raise DataError("geometric albedo must be positive", details={"albedo": albedo})
    ...
    return DIAMETER_CONSTANT_KM / math.sqrt(albedo) * 10.0 ** (-h_value / 5.0)
```

`p_v`, the geometric albedo, is *not measured* for most objects — it requires knowing the
diameter independently, which is the thing we are trying to estimate. So it must be assumed.
This project assumes 0.14, the conventional value for an unknown-composition NEO, and the
assumption is stated three times rather than once: as `default_geometric_albedo: 0.14` in
settings, as a comment above it ("the assumption is documented, not hidden"), and in the model
card's feature list. A hidden assumption is a bug that has not been found yet; a written one is
a quantity a reviewer can argue with.

The code distinguishes two kinds of bad input in a way worth noticing. An unparseable
magnitude returns `None` — one malformed cell in a 32,390-row download or a partial API
request must not take down a training run. A non-positive *albedo* raises: that is not a
measurement problem, it is a physically incoherent parameter, and continuing would produce
diameters that are silently nonsense.

`implied_albedo` inverts the same equation using JPL's *reported* diameter when there is one,
and `diameter_inconsistency_km` is the gap between the reported diameter and the
assumed-albedo diameter. Together they turn the assumption into a question: for the ~1.5 % of
objects with a measured diameter, does the assumed albedo hold?

### MOID, and the honest version of a proxy

Minimum orbit intersection distance is a real orbital-mechanical quantity: you take two
ellipses and find their minimum separation. It is computed from orbital elements with a
propagator, and this project has no elements and no propagator. It has a table of catalogued
close approaches. So:

```python
record["moid_proxy_au"] = best_distance  # min over the object's recorded dist_min
```

The module docstring refuses to let the name do more work than it can:

> `moid_proxy_au` - the smallest `dist_min` across the window. Real MOID is an orbital
> ... about, so it is a lower bound on nothing and an upper bound on the true MOID only if ...

Read plainly: it is neither a bound nor an approximation of MOID in the general case. An
object whose catalogued approaches are all 0.15 au could have an uncatalogued 0.02 au
crossing; an object with a 0.03 au approach recorded for 2027 has a genuinely small MOID but
this feature knows about it only because JPL computed and published that approach. Renaming it
`moid_au` would have been a lie in a datatype. `moid_proxy_au`, documented next to its
formula, is the truth in a longer name — and the model card's limitations section repeats it.

Two thresholds give this physical scale to a human: `LUNAR_DISTANCE_AU = 0.00257` (the Moon's
distance, the unit astronomers actually reach for) so `moid_lunar_distances = moid / 0.00257`
reads as "1.4 lunar distances", and `n_approaches_within_lunar` counting approaches inside
that radius.

### Cyclic variables: month as a point on a circle

```python
out["closest_month_sin"] = ...  # sin(2*pi*month/12)
out["closest_month_cos"] = ...  # cos(2*pi*month/12)
```

December and January are adjacent in time and 11 units apart as integers. Feed a model the
integer and you are asserting a metric on a quantity that has none — a linear model will fit a
trend toward 12, and a tree will happily split at 6.5 and call it a boundary. Encoding on the
unit circle makes adjacency the geometry: the distance between two months is now the arc
between them, and January is near December. It is two columns instead of one, and it is the
standard treatment for hours, days-of-year, and any other variable that wraps. The seasonal
signal in NEO approaches is genuine — survey cadence and detection geometry vary through the
year — and `closest_month_sin` earns 0.0029 AP importance for it: small, but nonzero, and
correctly represented.

`_month_from_cd` deserves a mention for the way it handles the real data: JPL's `cd`
(close-approach date) strings look like `"2 Mar 1950 12:07:00"`, so the month is parsed from
the middle token, with a numeric fallback, and anything that fails to land in 1–12 returns
`None`. When the closest approach has no parseable month at all, `_seasonal_fallback` uses the
median event instead. That is what defensive parsing of someone else's date format looks like
when you have 32,390 rows and no control over any of them.

### Check yourself

*Why is `prad_inconsistency` the most valuable feature?* Because it is the direct physical
signature of a wrong transit interpretation — the catalogued radius against the radius the
depth implies — and validation is largely the business of finding exactly that.

*Why log-transform features a tree model cannot benefit from?* So the linear baseline is a
fair comparison, and because the transforms are what the *relationship* is called in
astronomy. A comparison is only meaningful if both models see the same problem.

*What is assumed in `diameter_from_h_km`, and where is it written?* Geometric albedo 0.14, in
settings, in a code comment, and in the model card.

*Why not call `moid_proxy_au` just `moid_au`?* Because it is the minimum of catalogued
approach distances, not a computed orbit-intersection minimum. The name is documentation, and
a datatype is where documentation goes when nobody reads the comments.

---

## Chapter 7 — Missing values are statements, not zeroes

### Four ways to write "unknown", three of them wrong

| Representation | Meaning | In this project |
| --- | --- | --- |
| `NaN` (float) | not measured | ✅ the one true missing marker |
| `None` | absent in Python | converted to `NaN` at the frame boundary |
| `0.0` | measured, and the value is zero | ❌ never used for absence |
| `""` | empty string | ❌ would poison a numeric column's dtype |

The distinction that matters is semantic: a blank cell in a NASA table is a *claim about the
world* — the survey did not determine this quantity — and `0.0` is a different claim about the
world. Impute zero and you have asserted that an object with no reported diameter has zero
metres of diameter, that a KOI with no quoted eccentricity is on a perfectly circular orbit,
and that an asteroid with no `H` measurement is infinitely bright. Each of those is a
fabricated measurement, and models do not know the difference between a fact and a
hallucination they were handed.

### `to_numeric(errors="coerce")`: useful, and dangerous in the same breath

Every numeric column in both feature modules goes through `pd.to_numeric(..., errors="coerce")`
(or `_float`, which does the same per-value). Coercion turns anything unparseable into `NaN`,
which is exactly right for the archive's sentinel values and empty cells.

It is also silently wrong for a class of real bugs: a column that has changed *format* rather
than becoming missing — `dist_min` suddenly carrying `"1.2e-03 au"` with a unit suffix, or a
locale that writes `"0,05"`, or a header row embedded mid-file — coerces to all-`NaN` without
one word of complaint. The pipeline continues; the feature becomes uninformative; the score
drops and is blamed on the model.

Two mitigations exist here, and both are the general pattern for the problem:

- **Assertions at the frame boundary.** `build_matrices` refuses to proceed if any feature
  column is not float-typed, raising `DataError(f"non-numeric feature columns: {bad}")`. It
  also requires every named column to be present, so a schema change is a loud failure rather
  than a coercion to nothing.
- **A checksum over the whole artifact.** The cache fingerprint (chapter 4) changes if the
  bytes change, so a silent format shift at NASA is at least *detectable after the fact* by
  comparing the fingerprint against the sidecar. The data dictionary records per-column
  missingness rates, which is the third mitigation: `diameter_reported_km` ~98.5 % missing is
  written down, so a future 100 % is a visible change rather than a mystery.

What is *not* done, and should be named as a gap: there is no per-column missingness assertion
in the fetch path. A column that becomes 100 % missing is caught only downstream, when
`HistGradientBoostingClassifier` refuses to bin an all-`NaN` column — which is a real error,
but it is an error raised by a library rather than by an intention.

### Why `HistGradientBoosting` makes this tractable

Most sklearn estimators cannot see `NaN`. `SimpleImputer(strategy="median")` is required first,
and that is what the logistic-regression pipeline does:

```python
linear = Pipeline(
    [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
    ]
)
```

Median imputation is a reasonable, boring, defensible choice — but note what it costs: the
information *that the value was missing* is destroyed, replaced by a plausible number. A tree
model with native `NaN` support keeps it. `HistGradientBoostingClassifier` learns, for each
split, which branch missing values should take; missingness then becomes a usable signal, and
in this dataset it is one — an object with no reported diameter is a fainter, less-studied
object, which correlates with the label.

The one thing that breaks it: a column that is missing in **100 %** of rows has no
non-missing data for the histogram to bin, and the estimator raises. That is not a quirk to
work around silently; it is a useful canary, and this repository handles it by dropping
all-`NaN` columns before fitting rather than by imputing a constant, because a constant column
carries no information and would otherwise appear in the importance table as a liar.

### `NaN` and the API

`frame_from_payload` (quoted, with its docstring's key sentence):

```python
"""Build the one-row raw frame the API needs before calling `derive`.

Absent keys become NaN, which is the same treatment a blank catalogue cell gets, so
a partial prediction request behaves exactly like a partially-measured object.
"""

row = {column: payload.get(column, np.nan) for column in RAW_INPUT_COLUMNS}
```

That line is why every request field is optional and why `POST {}` returns 200: an
under-specified request is treated as a partially-measured object, through the identical code
path. The prediction on an all-`NaN` row is `FALSE POSITIVE` at 93.6 % — the model reduced to
its prior — which is a correct answer to "is this endpoint alive" and a meaningless answer to
any real question. The console never shows it, because the console always sends a full row.

### Check yourself

*Why not fill missing with zero?* Because zero is a measurement. Filling with it teaches the
model a relationship between "unmeasured" and whatever zero means in that unit.

*When is median imputation required, and what does it cost?* For estimators that cannot
handle `NaN`, like logistic regression. It costs the missingness signal itself.

*What does a fully-`NaN` column do to `HistGradientBoostingClassifier`?* It refuses to bin it
and raises. Which is why the training path drops such columns explicitly instead of letting
the library be the one to notice.

*What is the danger of `errors="coerce"`?* A format change becomes silent mass-missing. The
mitigations are column-presence and dtype assertions plus the artifact checksum, and the real
gap is no per-column missingness bound at fetch time.

---

## Chapter 8 — Splitting data that has repeated entities

### The assumption that breaks first

`train_test_split` shuffles rows and hands 20 % to the test set. That is correct when rows are
independent draws — and the assumption is not that rows are *different objects*, it is that no
row's information helps predict another row's label. Astronomical catalogues violate it
constantly, in two directions this project has to handle separately.

**Exoplanets.** Kepler reports several candidates per host star. DR25 has 8,054 KOI rows
across 6,923 distinct `kepid` stars; some stars carry four or five candidates. Every KOI for a
given star shares `koi_steff`, `koi_srad`, `koi_smass`, `koi_slogg`, `koi_kepmag` — the stellar
parameters, which are among the most informative inputs, and which are effectively a
fingerprint of the star. Split rows and the same star appears in train and test; the model
memorises "star X's parameters go with CONFIRMED" and then is asked about star X again. Nothing
about the *planet* is being generalised; it is a lookup.

**Asteroids.** The CAD API returns 32,390 close-approach *events* for 22,581 objects. This
project aggregates to one row per object before splitting — so within the model matrix, groups
and rows coincide (22,581 each). The grouping matters *earlier*, in the aggregation itself:
minima, medians, and counts across an object's events must never mix events from different
objects, and the events that were collapsed into `moid_proxy_au` are the same events that
would otherwise leak into three rows.

The lesson generalises to any catalogue with repeated entities — patients with several visits,
sensors with several readings, users with several sessions. **Determine the unit of
independence, then split on it.** For this project: `kepid` and `des`, configured in
`configs/settings.yaml` as `group_column`, with a comment explaining that `kepid` is the star
key because `hostname` does not exist in this table.

### Two mechanisms for two different jobs

The module docstring in `models/splits.py` states the split purpose better than any textbook:

> Two mechanisms, used for different jobs:
>
> * `grouped_holdout` - one deterministic 70/15/15 split by group, used for the final numbers
>   we publish. Class balance is preserved by distributing groups *per plurality class*
>   round-robin, so the rare PHA class does not vanish from the test fold.
> * `grouped_cv_splits` - `StratifiedGroupKFold` for model selection, so no group is ever
>   simultaneously trained and validated, and no fold is missing a class.

Why two, when sklearn has `StratifiedGroupKFold` and that is enough? Because they answer
different questions with different error profiles.

**Cross-validation** (5 folds) is for *choosing between candidates*. Its virtue is that every
row is out-of-fold exactly once, so the estimate uses all the data and has small variance:
0.7659 ± 0.0145 for exoplanets, 0.7386 ± 0.0249 for asteroids. Its weakness is that the same
data participated in selection, so the CV number is optimistic about the *choice*, even
though it is honest about the *model*.

**A held-out set** is for *reporting*. It plays no part in selection, hyperparameters, or
feature decisions, and it is touched exactly once, at the end. It is a single number with a
wider confidence interval and no selection bias, and it is the number quoted in the README.

The order matters and is a discipline rather than a technicality: metric declared, candidates
compared by CV, winner refit on train+val, holdout evaluated once. Do it in a different order
and the holdout has been used, and the honest number you are quoting is no longer honest.

### How the holdout is actually built

`grouped_holdout` operates on groups, never rows:

1. Compute each group's **plurality class** — the most common label among its rows. `_plurality_per_group`
   breaks ties on the label *string*, not dictionary order, with the comment
   *"so the split is identical across machines and Python hash seeds"*. That is a
   reproducibility fix, not a style preference: Python's string hashing is randomised per
   process unless `PYTHONHASHSEED` is pinned, so a tie broken by iteration order would give a
   different split on a different machine, and different published numbers.
2. Bucket groups by that class and shuffle each bucket with `np.random.default_rng(config.seed)`
   — the single seeded generator, `random_seed: 20261111` from settings.
3. Assign groups to folds with a greedy quota, `_round_robin_assign`:

```python
for group in ordered_groups:
    best_gap = -np.inf
    for name in fold_names:
        gap = quotas[name] - taken[name]
        if gap > best_gap:
            best_gap, best_name = gap, name
    assignment[group] = best_name
    taken[best_name] += 1
```

Each group goes to whichever fold is furthest below its share so far. Deterministic given
input order; balanced on cumulative counts; and stable when the catalogue grows by a few rows
— which is what makes these numbers re-runnable next month on a bigger archive.

Why not `train_test_split` twice? Because with groups and rare classes, repeated random splits
can leave the test fold with *zero* PHAs, and an average precision on an empty positive class
is undefined, not zero. The plurality bucketing is what guarantees the rare class appears in
all three folds: the test fold got 241 PHAs out of 1,025 positives in the population.

The resulting sizes (`reports/asteroids/metrics.json`, `folds`): train 14,678 rows/14,678
groups, val 3,387/3,387, test 4,516/4,516 — 5,273/1,208/1,573 for exoplanets, where the group
counts are smaller than the row counts by construction.

### The assertion that makes it a guarantee rather than a habit

```python
assert_no_group_leakage(...)
```

runs inside the training pipeline and in tests, checking that no group appears in two
splits. The docstring is blunt about its purpose: *"it is not decoration ... it is the
cheapest possible defence against a future refactor quietly reintroducing the bug."*

That sentence is the whole argument for verification-as-code. A design decision that lives
only in a docstring is a wish; a design decision that raises when violated is a property of
the program. Every invariant you actually care about should be this kind of assertion, and
"no group in two splits" is the one that, if broken, invalidates every number in the
repository.

### Check yourself

*Why is row-level splitting wrong here?* Because entities repeat. Kepler reports several
candidates per star and shares stellar parameters between them, so a row split lets the model
meet the same star twice.

*What is the difference between the CV number and the published number?* CV (0.7659) selects;
it is optimistic about the *choice*. The holdout (0.7687) is touched once, after selection, and
is what gets quoted.

*Why does the tie-break in `_plurality_per_group` use the label string?* Because Python's hash
randomisation would otherwise make the split machine-dependent, and a machine-dependent split
means unreproducible published metrics.

*How do you know the split still holds after a refactor?* `assert_no_group_leakage` runs in the
pipeline and in tests. It raises rather than trusting the reader.

---

## Chapter 9 — Choosing the metric before the model runs

### Accuracy is a description of the majority class

The asteroid task is 5.34 % positive (`prevalence: 0.0534` in
`reports/asteroids/metrics.json`). A model that answers `NOT_PHA` for every object is 94.66 %
accurate and has identified zero hazardous objects. This is the standard warning, but it is
worth stating precisely because the trap is not only at 100 % majority: exoplanets' largest
class is 49 %, so a majority-only model still scores 0.5022 accuracy while its macro-F1 is
0.2229 — respectable accuracy attached to a useless classifier.

Precision, recall, and F1 condition on the positive class, which is why they are the vocabulary
for imbalance:

```
precision = TP / (TP + FP)     of the things we flagged, how many were right
recall    = TP / (TP + FN)     of the things that exist, how many did we find
F1        = harmonic mean      punishes being bad at either
```

The harmonic mean is the point: a classifier that flags everything gets recall 1.0 and
precision near prevalence; the arithmetic mean of those two would look fine, and the harmonic
mean collapses.

### macro-F1 for three classes, and what averaging buys

`reports/exoplanets/metrics.json` reports per-class F1: CANDIDATE 0.6031, CONFIRMED 0.8375,
FALSE POSITIVE 0.8656.

- **macro-F1** averages those three equally: **0.7687**. Each class is a customer with one
  vote, so the rare `CANDIDATE` class cannot be sacrificed to feed the majority.
- **weighted-F1** weights by support: 0.8115. It is the metric to quote if your cost is
  per-row and classes are equally costly to get wrong.

The difference between them is diagnostic: 0.7687 vs 0.8115 says the weakest class is the
smallest class, and macro chosen as the selection metric means the model cannot pay for
CONFIRMED/FALSE POSITIVE accuracy by degrading CANDIDATE. Choosing the averaging scheme is
choosing which mistakes you are willing to make, and that is a decision to be made before
looking at any score.

### Why ROC-AUC flatters an imbalanced problem, and why it is reported anyway

ROC plots false-positive *rate* against true-positive rate — both normalised by the class's own
total. When the negative class is 95 % of the data, an FPR of 0.05 means thousands of false
alarms, and the curve barely moves: this model scores **ROC-AUC 0.9778** while its precision at
threshold 0.5 is **0.4787**. Those are both true statements about the same model, and the
245 false alarms in the confusion matrix are what reconciles them.

Precision-recall plots the quantities that actually hurt at 5 % prevalence. **Average
precision** — the area under the PR curve, computed by interpolation over thresholds — is
0.7261, against a floor of 0.0534 (equal to prevalence, which is what a ranking-noise model
scores). That gap, 0.7261 vs 0.0534, is the honest measure of what was learned, which is why
AP is the selection metric and ROC-AUC is reported as context.

Note the ordering trap: a model can have excellent ROC-AUC and terrible AP on imbalanced data,
and the difference is entirely about the negative class's size. If a reviewer asks "why is your
AUC higher than your precision?", the answer is that the two curves normalise over different
denominators.

### Probability metrics, reported but never used for selection

- **Brier score** — mean squared error between the one-hot label and the probability vector.
  Proper scoring rule: minimised by reporting the true probability. 0.0940 for exoplanets
  (3-class, so the "no information" baseline is ~0.667 for a uniform guess over three
  classes), 0.0425 for the asteroid task.
- **Log-loss** — harsher on confident mistakes; 0.4966 for exoplanets. The dummy baseline's
  log-loss is 17.94, because a majority-class predictor assigns probability 1.0 to the wrong
  class two thirds of the time and `-log(0)` is a very large number once you clip it.

`evaluation/metrics.py` states the discipline explicitly: Brier and the calibration curve *"are
reported but never used for selection: a model can be well-calibrated and useless"*. A
classifier that outputs the base rate for everything is perfectly calibrated at 5.34 % and
identifies nothing. Calibration is a property of the probabilities; discrimination is a
property of the ranking; a model needs both, and selecting on one is how you get a beautiful
number attached to a useless model.

### Two sklearn behaviours that will bite without being warned about

Both were discovered here at runtime, not from documentation, and both are pinned by tests in
`tests/test_metrics.py`.

**1. `average_precision` needs `pos_label` when labels are strings.** `make_scorer` passes
whatever binary function it wraps, and `average_precision_score` decides which column of
`predict_proba` is "positive" by looking for `pos_label`, whose default is `1`. With labels
`"NOT_PHA"`/`"PHA"` the default finds nothing and raises. The fix is `build_scorer`:

```python
def build_scorer(scorer_name: str, *, pos_label: str | None = None) -> Any:
```

and the asteroid pipeline passes `pos_label="PHA"`. The same subtlety is why
`reports/asteroids/metrics.json` records `"pos_label": "PHA"` inside the importance block —
a score without its positive-class convention is not reproducible.

**2. `predict_proba` columns follow *sorted* `classes_`, not settings order.**
`configs/settings.yaml` lists exoplanet labels as `CONFIRMED, CANDIDATE, FALSE POSITIVE`,
which is a natural reading order. sklearn sorts: `['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']`.
Any code that zips the configured label list against the probability columns will produce
answers that are wrong in a way no test about "does it return three probabilities" would
catch. `tests/test_api.py` pins this directly:

```python
assert [item["label"] for item in body["probabilities"]] == sorted(...)
```

and the sidecar keeps the config order as documentation of intent while the API returns the
sorted order as the reality of the artifact.

### Declaring the metric first

`models/train.py` sets the selection metric per task — macro-F1 for exoplanets, average
precision for asteroids — and it is used for candidate ranking before the holdout is opened
once. This is a research-process discipline, not a code requirement; nothing in sklearn stops
you from computing fourteen metrics and reporting the best.

But the project reports one number as its headline in the README and both model cards, and the
credibility of that number is entirely dependent on it not having been chosen after seeing it.
So the metric is written down in code and in `reports/*/metrics.json` under
`selection_metric`, before results exist, and everything else in the metrics block is
explicitly secondary. If a model's F1 looks bad and its AP looks good, the honest answer is
that AP was never the deal.

### Check yourself

*Why not accuracy here?* Because 94.7 % of the asteroid objects are `NOT_PHA`, so a classifier
that does nothing scores 0.946. Accuracy is only informative when classes are balanced.

*Why macro rather than weighted F1?* Weighted lets the model trade a rare class away for the
majority. Macro makes each class count equally, and CANDIDATE is exactly the class the task
should not be allowed to sacrifice.

*Why report ROC-AUC 0.9778 but select on AP 0.7261?* ROC normalises by class size, so a huge
negative class flatters it. PR curves do not. Both are true, one is honest, and the gap
between them is precisely the 245 false alarms.

*What two sklearn behaviours cost real debugging time here?* `average_precision` needing
`pos_label` for string labels, and `predict_proba` columns following *sorted* `classes_` rather
than the configured label order.

---

---

## Chapter 10 — Baselines: the numbers a model has to beat

### A score means nothing on its own

"Average precision 0.7261" is not a claim until you know what the alternatives scored. A
baseline is the answer to the question a reviewer always asks first: *how much better than the
trivial thing is this?* So every model in this project is reported next to at least one
baseline evaluated by the identical code path, on the identical split.

`models/train.py` keeps them in one dictionary:

```python
return {
    "dummy_majority": ("rule", MajorityRuleClassifier()),
    "nasa_pha_rule": ("rule", rule),
    "logistic_regression": ("learned", linear),
    "hist_gradient_boosting": ("learned", boosted),
}
```

Rules and learned models sit in the *same list* with the same interface and are cross-validated
by the same loop. That is a design decision with a reason: a baseline evaluated by different
code than the model is not a baseline, it is an anecdote. If the rule were scored on a
different split, or with a different `pos_label`, the comparison would be meaningless and no
test would notice.

### The floor: what a class-frequency guess gets you

`MajorityRuleClassifier` predicts the most common training label for every input. Exoplanets:
test macro-F1 **0.2229**, accuracy 0.5022 — the accuracy is respectable precisely because the
classifier is doing nothing but reciting the class shares (49 % FALSE POSITIVE). Asteroids: CV
average precision **0.0534**, which equals the prevalence, because a constant score ranks
positives no better than chance.

A floor is worth more than a percentage-point margin over a good model, because it establishes
that the task has a solution at all and where "no skill" sits. When the headline number is
close to the floor, the honest reading is that the task is hard, not that the reporting is
generous.

### The interesting baseline: NASA's published definition, implemented literally

A PHA is an object with MOID ≤ 0.05 au **and** absolute magnitude H ≤ 22.0. That is a rule
published by NASA, so it is the correct thing to beat, and it is implemented as an sklearn
estimator in `models/baselines.py::PHARuleClassifier`. The rule itself, quoted:

```python
def hazardous_mask(self, X: Any) -> np.ndarray:
    """The rule itself: MOID <= threshold AND H <= threshold, NaN fails the AND."""
    frame = _require_columns(X, (MOID_COLUMN, H_COLUMN))
    moid = pd.to_numeric(frame[MOID_COLUMN], errors="coerce").to_numpy(dtype=float)
    magnitude = pd.to_numeric(frame[H_COLUMN], errors="coerce").to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        return (moid <= self.moid_threshold_au) & (magnitude <= self.h_threshold)
```

```python
def predict_proba(self, X: Any) -> np.ndarray:
    check_is_fitted(self, "classes_")
    hazardous = self.hazardous_mask(X)
    probabilities = np.zeros((len(hazardous), 2), dtype=float)
    probabilities[:, 1] = hazardous.astype(float)
    probabilities[:, 0] = 1.0 - probabilities[:, 1]
    return probabilities
```

Three details are worth defending out loud.

**`fit` validates the label contract and sets `classes_` as `[negative, positive]`** — the
sklearn convention that column 0 is negative and column 1 positive, so `predict_proba` is
indexable by every consumer that assumes it. It raises `ValueError` if either label is absent
from the training data, which is the difference between a rule that silently never fires and a
rule that tells you its inputs are wrong.

**`np.errstate(invalid="ignore")` is not sloppiness.** A `NaN <= 0.05` comparison evaluates to
False and emits a warning; inside `&` that is exactly the desired semantics — an object with no
MOID cannot satisfy the definition — so the warning is suppressed for a *decided* behaviour
rather than allowed to fail the build (warnings are errors here, chapter 15).

**Missing data is a stated policy, and counted.** From the class docstring:

> Missing data policy, stated rather than assumed: an object with no MOID or no H cannot
> satisfy "MOID <= 0.05 AND H <= 22.0", so the rule calls it NOT_PHA. That is the literal
> reading of the definition, and `unknown_count()` reports how many objects the policy touches
> so the number is visible instead of buried in a NaN comparison.

`unknown_count()` is exposed and its result is written into the report as `pha_rule_unknowns:
0` — on this snapshot every holdout object had both numbers, so the policy touched nobody. On a
fresher snapshot with unmeasured magnitudes it will not be zero, and because it is a field in a
committed JSON file, the day it changes is visible.

### What the comparison says

| | AP | ROC-AUC | PHA recall | PHA precision |
| --- | ---: | ---: | ---: | ---: |
| `hist_gradient_boosting` | **0.7261** | 0.9778 | 0.9336 | 0.4787 |
| `nasa_pha_rule` | 0.2575 | 0.6098 | 0.2199 | 0.9815 |
| `dummy_majority` | 0.0534 | 0.5000 | 0.0000 | — |

The rule is *extremely precise and barely sensitive*. On the holdout it flags 54 objects; 53 of
them are real PHAs (98.1 % precision) but it finds only 22 % of the positives. The learned model
finds 93 % and accepts 245 false alarms.

ΔAP = **+0.4686** over NASA's own definition, which is the most interesting claim in the second
model. And note what it is *not* a claim about: it is not "the model is smarter than the rule".
The rule is correct by construction when its inputs are correct. What the gap measures is the
distance between `{MOID, H}` as *published quantities* and JPL's actual list — the model is
exploiting the other seventeen features, above all the orbit-quality ones, to predict which
objects survived that gap.

The API surfaces this comparison in the response itself, in `decision_threshold_note`:

> argmax at 0.5; NASA's published rule on the same inputs says PHA (MOID <= 0.05 au AND H <=
> 22.0). Where the two disagree, the rule is the reference — that disagreement is what the
> model was trained to predict.

A prediction that hides the rule it was measured against is the easiest way to fool a
reviewer, and the note is short enough to fit in the payload.

### Check yourself

*Why implement a rule as an sklearn estimator?* So it is scored by the same CV loop, on the
same splits, with the same `predict_proba` contract. Otherwise the comparison is not a
comparison.

*What does a ΔAP of +0.47 over the NASA rule mean, and not mean?* It means the model ranks
JPL's published list far better than the rule applied to proxy inputs does. It does not mean it
is a better hazard definition — the rule is exact when its inputs are exact.

*Why `np.errstate(invalid="ignore")` there?* Because `NaN <= x` being False is the intended
missing-data policy, and the build treats warnings as errors.

---

## Chapter 11 — Three models, and why gradient boosting won

### The candidate set, and why a linear model is in it

Exoplanets compares four, asteroids three. From `models/train.py`:

| Candidate | Kind | What it can represent |
| --- | --- | --- |
| `logistic_regression` | learned | a linear decision boundary in feature space |
| `random_forest_200` | learned | an average of deep, decorrelated trees |
| `hist_gradient_boosting` | learned | a sum of shallow trees fitted on residuals |
| `dummy_majority` / `nasa_pha_rule` | rule | a constant / a published definition |

The logistic regression is a `Pipeline`, and its composition is the lesson:

```python
linear = Pipeline(
    [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
    ]
)
```

Three steps, each answering a defect of the raw data: it contains `NaN` (chapter 7), it has
features spanning six orders of magnitude and inconsistent units (days next to Kelvin next to
solar radii), and it is class-imbalanced. `max_iter=3000` is the other half of the story — the
default 100 iterations do not converge on this scaled design, and a "weak linear baseline" that
is weak because it never converged is a straw man. `class_weight="balanced"` up-weights the rare
classes, so the linear model gets the same imbalance advantage the boosting model has from its
own weighting.

The reason to keep a linear baseline at all: if it scores 0.66, you learn that roughly 0.10 of
the boosting model's advantage comes from *non-linearity* rather than from capacity,
regularisation, or luck. Without it, all those explanations are unfalsifiable.

### `HistGradientBoostingClassifier`, concretely

```python
boosted = HistGradientBoostingClassifier(
    class_weight="balanced",
    learning_rate=0.08,
    max_iter=400,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=15,
    random_state=seed,
)
```

What the estimator is: instead of growing one very deep tree per bag (random forest), boosting
grows a sequence of shallow trees where each one is fitted to the *residuals* of the ensemble so
far, and the prediction is their weighted sum. `learning_rate=0.08` shrinks each tree's
contribution, so more trees are needed but each is less able to overfit. `max_iter=400` is the
cap on how many trees it may add.

`early_stopping=True` with `validation_fraction=0.15` and `n_iter_no_change=15` means the
estimator holds back 15 % of its own training data and stops adding trees once the loss on that
internal set fails to improve for 15 consecutive rounds. This is why `max_iter=400` is not a
risk: it is a ceiling the search will not reach if the data stops paying for more trees. It
also means the estimator's internal notion of "done" is doing work that a manual grid search
would otherwise have to do — an honest trade when the CPU budget is one laptop.

*Histogram binning* is the specific reason this variant is fast and CPU-viable. Ordinary trees
sort feature values at every split, which is `O(n log n)` per candidate split. HistGB pre-bins
each continuous feature into at most 255 discrete bins **once**, then searches over bin
boundaries instead of values — `O(n)` for the binning pass, and the split search is over a
small, fixed alphabet. On 8,054 rows and 21 features, `fit` takes 14.15 s versus 3.70 s for a
200-tree random forest; the ordering would be less obvious on 800,000 rows, which is the point.

It handles `NaN` natively (chapter 7), and `class_weight="balanced"` gives the rare class more
influence on the loss, which for the 5 % PHA task is the difference between a model that ranks
and a model that has given up.

### The full candidate tables, unedited

Exoplanets, CV macro-F1 (`reports/exoplanets/metrics.json`):

| Candidate | CV mean | CV std | Fold scores | Fit |
| --- | ---: | ---: | --- | ---: |
| hist_gradient_boosting | **0.7659** | 0.0145 | .756 .763 .771 .788 .752 | 14.15 s |
| random_forest_200 | 0.7602 | 0.0138 | .745 .754 .756 .782 .764 | 3.70 s |
| logistic_regression | 0.6632 | 0.0107 | .656 .654 .658 .679 .670 | 0.72 s |
| dummy_majority | 0.2192 | 0.0001 | .219 .219 .219 .219 .219 | 0.01 s |

Asteroids, CV average precision:

| Candidate | CV mean | CV std | Fold scores | Fit |
| --- | ---: | ---: | --- | ---: |
| hist_gradient_boosting | **0.7386** | 0.0249 | .768 .710 .762 .729 .725 | 9.92 s |
| logistic_regression | 0.5266 | 0.0307 | .505 .556 .537 .485 .550 | 0.89 s |
| nasa_pha_rule | 0.2642 | 0.0250 | .289 .294 .240 .250 .249 | 0.01 s |
| dummy_majority | 0.0534 | 0.0000 | .053 .053 .053 .053 .053 | 0.02 s |

### Reading those tables honestly

Three things a careful reviewer should notice, and which the project should say before being
asked.

**Random forest is within one standard deviation of the winner on exoplanets** — 0.7602 ± 0.0138
vs 0.7659 ± 0.0145. That is not a meaningful difference; the fold scores overlap on three of
five folds. The correct statement is "both tree ensembles do well and the pre-declared metric
picked the boosting one", not "boosting is better here". The forest is not deleted from the
report for that reason — a comparison that hides its runner-up is a comparison rewritten.

**On asteroids the gap is real**: 0.7386 ± 0.0249 versus the rule's 0.2642 ± 0.0250 and the
linear model's 0.5266. Roughly eight standard errors above the linear baseline, which is a
difference that survives the noise.

**The fold standard deviations matter more than the means** in one specific way: the boosting
model's exoplanet CV (0.7659 ± 0.0145) versus its holdout (0.7687) agree to within a fraction
of one standard error. That agreement is the evidence the result is not an artefact of one lucky
split — which is a much stronger claim than either number alone, and the reason the CV number
gets published next to the holdout number rather than hidden.

### What was not tried, and why that is fine

No hyperparameter grid search beyond the hand-set values above; no XGBoost or LightGBM; no
neural network; no GPU. The reasons are the same reason: the machine is a CPU-only Snapdragon X
laptop, and a 400-tree × 3-way grid over 5 folds would cost hours for a gain that the linear →
random-forest → boosting progression suggests would be small on 8,000 rows and 21 features. A
capstone that claims "we did not search because the marginal value looked low on this data
volume" is more defensible than one that claims a tuned result it cannot reproduce. The choice
is documented in this chapter rather than left to be inferred from an absence.

### Check yourself

*Why include a logistic regression?* To separate "non-linearity helps" from "capacity helps",
and to prove the pipeline handles imputation, scaling, and class weighting explicitly rather
than assuming the strong model's defaults.

*Why is HistGB fast?* It bins each feature into ≤ 255 buckets once and searches split points
over bins instead of sorting values at every node.

*Did gradient boosting actually win on exoplanets?* By the pre-declared metric, yes; by the
standard deviations, it is a tie with the random forest, and that is the true statement.

---

## Chapter 12 — Calibration, thresholds, and what a probability is for

### Right and confident are two different properties

A classifier ranks objects; a *probabilistic* classifier additionally claims that an object
assigned 0.8 belongs to the class 80 % of the time. Whether that claim is true is calibration,
and nothing in the training objective guarantees it. Cross-entropy does reward honest
probabilities in theory, and in practice — especially with `class_weight="balanced"` on a 5 %
positive class, which deliberately distorts the effective prior — the model is sharper than the
world.

### The reliability table, read out loud

`reports/asteroids/metrics.json` embeds the holdout reliability data in `figures.calibration_data`
(10 uniform bins over 4,516 objects), plotted as `calibration.png`:

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

Monotone — higher predicted probability really does mean higher observed frequency, which is
why average precision and ROC-AUC are good. And systematically above the diagonal, so every
bin overstates. The most-confident 10 % are 58.8 % PHA, not 94.9 %.

The correct conclusion is narrow and storable: **these probabilities are safe for ranking and
unsafe as frequencies.** "This object is in the top 3 % by score" is a defensible sentence.
"This object has a 94.9 % chance of being hazardous" is not, and the console does not print
it that way — it labels the outcome *marginal* / *clear* rather than reciting the number as a
probability of danger.

The exoplanet report has no calibration figure. Brier 0.0940 and log-loss 0.4966 on a 3-class
problem are consistent with useful probabilities but do not demonstrate calibration, and
`reports/exoplanets/` generates confusion, importance, and candidate figures instead. That
asymmetry is a real gap in the deliverable, stated here rather than left to be discovered.

### 0.5 is an assumption, not a default worth trusting

Argmax at 0.5 encodes a cost model in which a false positive and a false negative are equally
bad. On the PHA task they are not obviously equal: a missed hazardous object is the failure with
consequences, and a wasted follow-up observation is the failure with a cost. Changing the
threshold moves those two numbers against each other, and the honest way to expose that is to
publish the curve rather than pick a point.

`evaluation/metrics.py::operating_points` does exactly that: it finds thresholds that
*achieve* a target precision on the holdout.

| Target precision | Threshold | Achieved precision | Recall |
| ---: | ---: | ---: | ---: |
| 0.10 | 0.0053 | 0.3604 | 0.9959 |
| 0.25 | 0.0053 | 0.3604 | 0.9959 |
| 0.50 | 0.6359 | 0.5000 | 0.8963 |
| 0.75 | 0.9412 | 0.7536 | 0.4315 |

Two rows share a threshold, and that is not a bug — it is the model's precision floor. It cannot
be made to deliver 10 % precision because its least aggressive usable setting already yields 36
%. Reading a table where two "targets" collapse onto one row is one of the small skills of
working with operating points: the curve is not defined over the whole idealised range.

The useful sentences that come out of it: at 75 % precision you catch 104 of 241 real PHAs and
miss 137; at the default threshold you catch 225 and accept 245 false alarms. Which is correct
depends on what a wasted observation costs, and this repository deliberately does not answer
that — it hands over the trade and leaves the decision with the person who has the cost model.

### What the API says about its own threshold

The exoplanet route attaches, to every response:

```python
scored["decision_threshold_note"] = (
    "argmax over the class probabilities; no threshold was applied. Macro-F1 was "
    "selected on grouped cross-validation, so the operating point is the default one."
)
```

That string is doing an important job: a `prediction` field invites the reader to believe a
decision rule was tuned. It was not. The stated reason is the true one, and putting it in the
payload means the claim travels with the number instead of living in a document nobody opens
during the demo.

### Check yourself

*What is the difference between discrimination and calibration?* Discrimination is whether the
ranking is right (AP, ROC-AUC). Calibration is whether the numbers themselves are honest
frequencies. This model has the first and not the second.

*Why is the model overconfident?* `class_weight="balanced"` reweights the loss against a 5 %
prior, which is what makes the rare class rankable and simultaneously what makes the top bins
overstate.

*Which bin breaks the "probabilities are frequencies" claim hardest?* The 0.949 bin at 0.588
observed — the model's most confident group is wrong four times in ten.

---

## Chapter 13 — Interpretation without self-deception

### Why tree coefficients do not exist

A linear model gives you a slope per feature; you can read the sign and argue about it. An
ensemble of 400 boosted trees gives you hundreds of thousands of split conditions, and
"importance" has to be defined by what a model *does* rather than by what it *says*.

Permutation importance answers with an experiment: take the fitted model, take the holdout,
**shuffle one feature column**, re-score, and record how much worse the selection metric got.
If a column carries real information, scrambling it costs measurable performance. If it is
redundant or noise, the score barely moves.

`evaluation/metrics.py::permutation_importance_block` runs it with `n_repeats=8` on the holdout,
scoring `f1_macro` for exoplanets and `average_precision` (with `pos_label="PHA"`) for
asteroids, and stores the mean and standard deviation of the decrease per feature. Repeats
matter: one shuffle is one sample of a random permutation, and the noise in a single repeat is
comparable in size to the effects being measured.

### The top of each list, and why it is the right top

Exoplanets, mean macro-F1 decrease:

| Feature | Decrease |
| --- | ---: |
| `model_snr` | 0.1426 |
| `duration_over_period` | 0.0887 |
| `log_prad` | 0.0766 |
| `prad_inconsistency` | 0.0338 |
| `max_single_event_snr` | 0.0290 |

Signal-to-noise first and transit geometry second is the physically sensible ordering: validation
is mostly the question "is this signal real, and does it look like a planet crossing a star of
this size?". `max_single_event_snr` in fifth is the same idea from the other direction — one
strong event rather than a coherent model fit. A model that ranked `koi_kepmag` first would be
evidence of a pipeline bug rather than a discovery.

Asteroids, mean AP decrease:

| Feature | Decrease |
| --- | ---: |
| `absolute_magnitude_h` | 0.4982 |
| `moid_proxy_au` | 0.2870 |
| `diameter_from_h_km` | 0.1196 |
| `uncertainty_ratio_at_closest` | 0.0180 |
| `diameter_inconsistency_km` | 0.0123 |

`H` and the MOID proxy dominate because they are the two terms in NASA's definition — which is
the expected result and also the reason this table alone would be a weak deliverable. The
interesting part is the tail: `uncertainty_ratio_at_closest` and `t_sigma_seconds_at_closest`
(0.0089) are orbit-quality features the published rule ignores entirely, and they are precisely
where human list-makers depart from it.

### The three traps, all of them present in these files

**1. Negative importance is not a bug.** `stellar_logg` scores −0.0026 and `log_stellar_radius`
−0.0017 on exoplanets; `implied_albedo` scores −0.00015 on asteroids. Permuting a useless column
randomly reassigns values, and by chance the noise can push a few rows onto the correct side of
a boundary. The mean of eight noisy repeats can therefore be negative. The correct reading is
"this column carries nothing the ensemble was not already getting from elsewhere", not "this
column is actively harmful". Reporting them as zeros would round away the honest evidence that
the estimator's noise floor is about ±0.003 macro-F1 here — which is exactly how large a number
you should *not* interpret.

**2. Correlated features each look unimportant.** `diameter_from_h_km` is a deterministic
function of `absolute_magnitude_h` — trees cannot gain information from a monotone
re-expression — yet it scores 0.1196. What that measures is redundancy, not discovery: the
ensemble drew some trees on `H` and others on the diameter transform, so permuting one column
damages only the trees that used it. The honest combined statement is that brightness is worth
about 0.62 and geometry about 0.29. The pure form of the same phenomenon is in the zeros:
`log_moid`, `moid_lunar_distances`, `n_approaches_within_lunar`, `log_diameter_from_h`, and
`n_orbit_solutions` all score exactly 0.0, because each is a monotone recombination of something
already in the list. Those columns are in the model for the console's benefit, and this chapter
says so instead of presenting them as findings — a feature list that contains both a quantity
and its log is a feature list whose importance table must be read as a *partition of credit*,
not as a ranking of discoveries.

**3. Importance on the holdout is not causal.** It measures how much this fitted model on this
distribution leans on a column. It says nothing about whether the quantity physically causes the
label, and it changes when the training data changes. `koi_count` — how many candidates share a
star — is a good example of the danger, flagged in the module docstring itself:

> `koi_count` (how many candidates share that star) is kept but flagged in the learning guide as
> a confound: multi-candidate systems were statistically easier to validate, so it encodes a
> discovery-pipeline advantage, not physics.

It scores 0.0175, ranks eighth, and is *not* evidence that planets cluster. It is evidence that
the vetting process was easier when they did. Any interpretation of this table that leaves out
that sentence has turned a metric into a discovery, which is the single most common way
interpretable ML becomes misleading.

### Check yourself

*How does permutation importance work?* Scramble one column on held-out data, measure the drop
in the selection metric, repeat 8 times, report the mean and the spread.

*Why can importance be negative?* Because permutation is random; a useless column's noise can
occasionally help. The magnitude of those negatives is your noise floor.

*Why do `H` and its diameter transform both score highly?* Because credit for one signal is
partitioned across the columns the ensemble happened to use. Sum them; do not rank them.

*What is the confound in `koi_count`?* Multi-candidate systems were easier to validate, so the
feature encodes the vetting process, not the planets.

---

## Chapter 14 — From notebook to artifact: registry, metadata, and skew

### The failure this layer exists to prevent

A notebook trains a model. A deploy script copies the `.joblib`. Meanwhile the feature module
gained a column, reordered two, or renamed one. The API now builds a 21-column matrix and hands
it to a model fitted on 20 — or worse, hands it 21 columns in a *different order* and gets
confident predictions computed from a matrix where `log_prad` values sit in `impact`'s slots.
Nothing raises. Every number is wrong.

That is train/serve skew, and it is the most common way a correct model becomes an incorrect
system. The defence is not care, review discipline, or a checklist. It is a guard that makes the
mismatch impossible to ship.

### What gets written, and why it is two files

`models/registry.py::save_artifact` writes `<name>.joblib` (the estimator) and `<name>.json`
(the metadata), next to each other, and the metadata is deliberately *outside* the binary:

> Metadata is written next to the model rather than inside it, so a reviewer can read the
> numbers without loading a binary.

Which fields, and the question each answers (`models/exoplanet_disposition.json`):

| Field | Question it answers |
| --- | --- |
| `algorithm` | which candidate won (`learned:hist_gradient_boosting`) |
| `feature_names` | the exact 21 columns, in order |
| `feature_hash` | `41939f75dc829393` — a 16-hex digest of that ordered list |
| `n_features` | a count, for a cheap sanity check |
| `labels` | the contract the API promises callers |
| `feature_ranges` | min/max seen in training, per feature — what "extrapolation" means |
| `rows`, `groups` | what the split was built from (8,054 / 6,923) |
| `folds`, `holdout` | the sizes of train/val/test and the CV time |
| `metrics`, `train_metrics` | holdout and in-sample scores — the overfitting gap |
| `baseline` | what the reference strategy scored on the same holdout |
| `cv` | the entire candidate table, so the loser is still visible |
| `dataset_fingerprint` | which bytes (chapter 4) |
| `environment` | `python 3.12.10`, `scikit_learn 1.9.0`, `Windows-11-10.0.26300-SP0` |
| `trained_utc`, `schema_version` | when, and what this file's shape means |

A reviewer with a text editor can now audit the model's provenance without unpickling anything —
which is the difference between an artifact and a black box, and the reason `git diff` on
`models/*.json` is a meaningful code review.

### The hash, and why it is order-sensitive

```python
def feature_list_hash(names: list[str] | tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()[:16]
```

Two details, each preventing a specific collision class. The `\x1f` (unit separator) between
names stops `["ab", "c"]` and `["a", "bc"]` hashing identically — a real risk when you
concatenate strings into a digest. Truncating to 16 hex characters is fine because this is an
integrity check against accidental drift, not a defence against an adversary: at 64 bits the
collision probability across the handful of lists this project will ever hold is negligible, and
a short hash is readable in an error message.

Order-sensitivity is the point rather than a nuisance: a model trained on a matrix does not care
what the columns are *called*, it cares which numbers arrived in which position. A set-based
or sorted comparison would accept a reordered contract and produce the silent corruption this
whole chapter is about.

### The guard, verbatim

```python
recorded = list(metadata.get("feature_names") or [])
live = list(expected_feature_names)
if recorded != live:
    raise DataError(
        f"trained model {name!r} was fitted on a different feature list than the one "
        "this code builds. Refusing to predict: the matrix would be silently misaligned. "
        f"Model expects {recorded}, current pipeline produces {live}. Re-run "
        "`nasa-space-ml train`.",
        details={
            "model_hash": metadata.get("feature_hash"),
            "live_hash": feature_list_hash(live),
            "missing_in_model": sorted(set(live) - set(recorded)),
            "extra_in_model": sorted(set(recorded) - set(live)),
        },
    )
```

The message is written for the person who hits it at 11 p.m.: it states the consequence
("silently misaligned"), shows both lists, names the fix, and the `details` block distinguishes
a rename from an addition from a deletion. `missing_in_model` and `extra_in_model` are the
difference sets a caller can act on, and they are what makes `tests/test_registry.py` able to
assert the error's *shape*, not merely that it raised.

Adjacent to it, the same rigour on labels:

```python
if expected_labels is not None and list(metadata.get("labels") or []) != list(expected_labels):
    raise ConfigError(
        f"trained model {name!r} has labels {metadata.get('labels')!r} but the caller "
        f"expects {list(expected_labels)!r}; the label contract changed",
        ...,
    )
```

A changed label set is not a feature mismatch, and it gets a different exception class and a
different HTTP status: you cannot fix it by retraining blindly, because the meaning of the
endpoint changed. See [errors.md](errors.md#invalid-features).

### The version question, and failing closed

```python
installed = sklearn.__version__
trained_with = str((metadata.get("environment") or {}).get("scikit_learn", "unknown"))
if trained_with.split(".")[:1] != installed.split(".")[:1]:
    raise DataError(
        f"model {name!r} was trained with scikit-learn {trained_with} but this "
        f"environment has {installed}. Major-version unpickling is not safe; retrain.",
        ...,
    )
```

joblib round-trips pickled estimators across *minor* versions reliably; majors are where
unpickling starts producing objects that exist but behave differently. Comparing only the first
segment is therefore the right granularity, and it is the sort of thing worth knowing rather
than discovering.

The subtle part is `"unknown"`. If the sidecar has no recorded version, `trained_with` becomes
the literal string `"unknown"`, whose first segment is not `"1"`, so the comparison fails and
the artifact is refused. That is **fail-closed** behaviour, and `tests/test_registry.py` pins it
deliberately: an unverifiable artifact is not served, even though the failure is annoying on a
machine where the model would probably have loaded fine. The alternative — assume compatibility
when you do not know — is how a system ends up serving a confidently wrong prediction.

A related rough edge, recorded honestly: a *corrupt* sidecar (truncated JSON) raises
`json.JSONDecodeError` rather than a typed project error, because `load_metadata` does not wrap
the parse. It is survivable only because `api/deps.py::initialise()` catches everything the
model layer throws, records it in `model_errors`, and lets the app boot. A test asserts the
current behaviour rather than the ideal one, and the fix is a one-line `try`.

### How this reaches HTTP

`initialise()` loads each artifact once at startup. Failure does not stop the process: the model
is listed as unavailable, the routes that need it answer 503
([errors.md](errors.md#not-trained)), and everything else keeps working. Crashing at boot turns
one broken artifact into an outage for endpoints that never touched it — and because the failure
is *recorded* rather than logged-and-forgotten, `/api/v1/health` reports per-model state, which
is the only shape of health check that is useful when a service has more than one model.

### Check yourself

*What is train/serve skew?* The served code builds a different feature vector than the training
code did, and the model cannot tell.

*Why store metadata in a JSON file rather than on the estimator?* So it is reviewable without
loading a pickle, greppable in a diff, and readable by a person deciding whether to trust the
artifact.

*Why hash the feature list in order?* Because a model consumes a matrix. A reorder is a
corruption, and a set-based comparison would accept it.

*What happens if `scikit_learn` is missing from the sidecar?* The artifact is refused. Fail
closed: an unverifiable model is not served.

---

## Chapter 15 — Serving, testing, shipping, and what is not proven

### The app factory, and why it is not a global

`api/app.py` exposes `create_app(config_path=None)` returning a fresh `FastAPI` instance, and a
module-level `app = create_app()` that exists *only* so `uvicorn nasa_space_ml.api.app:app`
works. The factory is what makes the test suite possible: `tests/test_api.py` builds a second
app against a temporary repository with empty model and data directories — the `degraded_client`
fixture — and asserts that this configuration still serves `/api/v1/health` at 200 and `/` at
404. None of that is expressible against a module-level singleton, and a test suite that cannot
construct a broken app cannot prove it degrades gracefully.

Everything runtime lives on `app.state.nsm`, one `AppState` dataclass: config, cache, loaded
models, model errors, and the memo. Two apps in one process cannot contaminate each other, which
matters more than it sounds — the alternative is a module-level cache that makes test ordering
into a hidden dependency.

Route handlers are plain `def`, not `async def`. FastAPI runs them in a threadpool, and that is
correct here because the request path is CPU-bound (feature derivation plus a tree traversal)
rather than I/O-bound: the only network I/O in the whole system happens at *fetch* time, in a
command you run deliberately. Marking these handlers `async` would put CPU work on the event
loop and make every concurrent request slower. The same reasoning says `httpx.Client` rather
than `AsyncClient`: async here would be a fashion choice with a measurable cost.

### The contract is physics

`api/schemas.py` gives every field bounds derived from what the quantity is, not from what the
dataset happened to contain: `koi_depth ≤ 1e6` ppm (a transit cannot block more than the whole
star), `koi_incl` in [0, 180], `closest_approach_month` in [1, 12], `implied_albedo` in [0, 2],
`moid_proxy_au ≤ 5.0`. And `extra="forbid"` everywhere.

The bounds serve two purposes at once. They catch a units mistake at the boundary — a depth
entered in fractional units (0.01) rather than ppm (10,000) would otherwise be a perfectly valid
feature value describing an impossibly tiny planet and a prediction nobody would question. And
they make the schema self-documenting: `/api/v1/exoplanets/schema` returns the fields with units
and notes, so a reviewer can check whether the bound is *right* without reading Pydantic.

Bounds from physics, ranges from data — that distinction is what makes the schema auditable.
`feature_ranges` in the sidecar is the data-derived twin, and the API uses it differently:
`koi_prad = 50,000` is *accepted* (nothing in physics forbids a large radius) and flagged as
extrapolation in the response, because refusing to answer is not the same as pretending the
input was in distribution.

### Errors as documents, requests as traceable

Every non-2xx is `application/problem+json` produced by exactly one function
(`problem.py:from_internal_error`), carrying `type` (a documentation anchor), `title`, `status`,
`detail`, `instance`, `code`, `request_id`, and a structured `errors` block. No handler
constructs a body by hand; no traceback crosses the boundary.

`RequestIdMiddleware` gives every request a 16-hex-digit `X-Request-Id`, honours a caller-supplied
one up to 64 characters, replaces an over-long one, and puts the id in the error body as well as
the header. The purpose is concrete and testable: a screenshot taken during a review session
identifies one line in the server log. Two assertions in `tests/test_api.py` pin the property
that makes this useful — a generated id is always 16 hex characters, and the same id appears in
both the response header and the JSON body.

### The 13-second memo

`GET /api/v1/asteroids/sample` took 13 seconds to answer. Profiling put `load_events` at 0.17 s
and `aggregate_events` at 13.12 s: collapsing 32,390 events into 22,581 objects in Python, on
every request. The fix is `AppState.asteroid_tables()`, memoised against
`dataset_fingerprint()`:

```python
if self._asteroid_tables is None or self._asteroid_tables[0] != fingerprint:
    events = asteroid_data.load_events(self.config, self.cache)
    objects = asteroid_features.aggregate_events(events, self.config)
    self._asteroid_tables = (fingerprint, events, objects)
```

Keying on the fingerprint rather than on "have we computed this yet" is the part worth defending:
a re-fetch changes the hash, which invalidates the memo automatically. A boolean flag would have
made the API serve features computed from a snapshot the model was not trained on, which is
chapter 14's skew through the back door. Both the events table and the aggregated objects are
cached together because they are always needed as a pair — `/sample` needs objects, `/predict`'s
rule comparison needs events.

Four tests cover the memo: identity reuse on a second call, recomputation when the fingerprint
changes, `pd.testing.assert_frame_equal` against a direct un-memoised call (proving the cache
cannot change the *answer*, only the speed), and repeated HTTP calls sharing one aggregation.

### The console, and the test that keeps it offline

`web/` is three files: no framework, no build step, and — deliberately — no CDN. A test enforces
it structurally:

```python
assert re.findall(r'(?:src|href)\s*=\s*["\']https?://[^"\']+', text) == []
```

Zero external origins in the served HTML. The reason is not ideology, it is a demo: a page that
loads React from a CDN and a font from Google fails silently in a room with bad wifi, and half
the layout arrives late or not at all. The console is a screening instrument — FETCH ROW, RUN
SCREENING, then probabilities, the features that mattered, and a provenance block naming the
model, algorithm, sklearn version, training timestamp, dataset fingerprint, and request id
underneath every answer. It calls `/sample` then `/predict`, and never computes a score in
JavaScript: exactly one implementation of the model's behaviour exists, and it is Python's.

### The test suite, and the two conventions that make it honest

371 tests, ~26 s, `warnings.simplefilter("error")` active. Three structural choices do the work:

**1. `httpx.MockTransport`, everywhere.** `tests/conftest.py` and the data-layer tests construct
clients with a fake transport returning canned responses, which makes the interesting failures
reachable: a 503 then a 200; a 400 that must *not* be retried; a TAP error page where CSV was
expected; `signature.version` at "2.0"; a page that returns zero rows; a malformed body that must
not become a cached bad body. With real endpoints these cases are untestable, and untested retry
logic is almost always wrong.

**2. Skip when the data is absent, never fail.** The tests that need real NASA bytes are guarded
by `@pytest.mark.skipif(not _data_cached(), reason="...run nasa-space-ml fetch")`. A fresh clone
gets a green suite with a documented reason; a machine with data gets real coverage. The
alternative — a red test that cannot pass without 6 MB of gitignored bytes — teaches people to
ignore the suite. The same pattern guards `_models_trained()`, so `TestRealArtifacts` in
`tests/test_registry.py` loads the *committed* joblibs against the live `FEATURE_COLUMNS` on any
machine that has them.

**3. Tests assert behaviour, including rough behaviour.** When a corrupt sidecar raises a raw
`json.JSONDecodeError` instead of a typed error, the test asserts that fact with a comment
explaining it is a recorded rough edge. A test suite that only asserts what the author liked is
documentation; one that asserts what the code does, including the parts that are unpleasant, is
a specification.

Three tests are marked `@network` and deselected by default (`addopts = "-m 'not network'"`):
`tests/test_live_probe.py` fetches both sources into temporary caches and asserts the endpoints
still answer, still contain the requested columns, and — in the third — that a freshly computed
`dataset_fingerprint` still matches the sidecars'. It `pytest.skip`s with an explicit message
when they diverge, because a re-published NASA table is not a repository bug, and it *fails* only
if the two committed artifacts disagree with each other, which would be. Those probes are the
only things in the suite that speak to whether the README's numbers still describe the world.

### CI, and the three claims each job is making

`.github/workflows/ci.yml` runs three jobs on every push:

- **`lint`** — `ruff check` and `ruff format --check` over `src`, `tests`, `scripts`. Two
  decisions embedded in `pyproject.toml`: `line-length = 100` with `E501` ignored (long lines in
  *docstrings and comments* are fine; the formatter handles code), and a rule selection that
  includes `B` (bugbear) and `SIM` — which is how `zip(..., strict=True)` and the
  mutable-default traps got caught before runtime.
- **`test`** — Python 3.11 and 3.12, with a step that is the interesting one: it imports the
  live `FEATURE_COLUMNS` from both feature modules and asserts equality against
  `models/*.json`'s `feature_names` before running pytest. This is the skew guard of chapter 14
  promoted to a CI gate, so a contributor who edits a feature list discovers it on their pull
  request rather than in a deployed 422. Then `pytest -m "not network"` and `--offline status`.
- **`smoke`** — boots `python -m nasa_space_ml serve --port 8000` in the background, polls
  `/api/v1/health` up to 40 times, POSTs a real KOI body and asserts HTTP 200, curls `/`,
  `/static/app.js`, `/docs`, then `docker build` and `docker run` with a `/api/v1/meta` poll.
  `trap` cleans up the server and dumps its log on failure. Its existence is the claim that this
  repository is *runnable* and not merely importable — a capstone that does not boot is worth
  nothing to a reviewer with ten minutes.

The container is what makes the path-resolution code in `api/app.py` necessary: a wheel installed
into site-packages has no repository above it, so `resolve_config_path` honours `NSM_CONFIG` and
`web_candidates()` honours `NSM_WEB_DIR` before falling back to the package-relative and
working-directory guesses. `web_dir()` returns `None` when no page exists, and the app then
serves the API with `/` returning 404 — because the alternative is a crash at import time on an
install that is otherwise perfectly functional.

### What is not proven

A list of the honest gaps, since a document that claims completeness cannot be trusted about
anything else.

- **The Dockerfile has never been built on the author's machine.** Docker Desktop is not running
  on this laptop, so the image is verified only by CI's `smoke` job. Everything else here is
  verified locally *or* in CI; this is verified in CI only.
- **`Retry-After` is ignored.** The retry policy uses its own schedule even when a server states
  how long to wait (chapter 3).
- **No per-column missingness bound at fetch time.** A column that becomes 100 % missing is
  noticed downstream by a library, not upstream by an intention (chapter 7).
- **The exoplanet report has no calibration figure** (chapter 12).
- **No authentication, no rate limiting, no multi-tenancy, no load testing.** There is no
  deployment that needs them, and theatre-grade security in a capstone misrepresents what has
  actually been exercised. The single-process service is expected to handle a reviewer and a
  browser tab.
- **No accessibility audit** on the console beyond semantic markup and keyboard reachability.
- **A corrupt sidecar raises an untyped error** (chapter 14) — survivable only because
  `initialise()` catches everything.
- **A concurrent-request benchmark has never been run.** `AppState`'s memo is not
  explicitly locked; two simultaneous first-call requests to `/api/v1/asteroids/sample` would
  both aggregate and the second overwrite the first. That wastes 13 seconds and returns correct
  data — it is a performance bug, not a correctness bug — but it is a real property of the code
  that no test covers.
- **Random forest on exoplanets is statistically indistinguishable from the winner** (chapter
  11). The published number is the pre-declared metric's choice, not proof of superiority.
- **The models are snapshots of a moving catalogue.** NASA re-publishes; the fingerprint is how
  you notice, and re-running the probes is how you confirm the README is still true.

### Check yourself

*Why an app factory?* So tests can construct a deliberately broken app, and so containers can
inject a config path. A singleton can neither be degraded on purpose nor configured.

*Why sync route handlers?* The work is CPU-bound; `async def` would block the event loop. All
network I/O happens in `fetch`, not in a request.

*Why does no external script get loaded by the console?* A structural test asserts it, and the
reason is that a demo must survive a room with no wifi.

*What does the CI feature-list step catch that the runtime guard does not?* Timing: a pull
request instead of a production 422.

*Name three things this project has not verified.* Docker on the author's machine; behaviour
under concurrent load; calibration for the exoplanet model.

---

## Glossary

| Term | Meaning as used here |
| --- | --- |
| ADQL | Astronomical Data Query Language; the SQL dialect NASA's TAP service accepts. |
| TAP | Table Access Protocol; IVOA standard for querying an astronomical archive over HTTP. |
| KOI | Kepler Object of Interest — a detected transit candidate awaiting validation. |
| Disposition | The published verdict on a KOI: `CONFIRMED`, `CANDIDATE`, `FALSE POSITIVE`. |
| PHA | Potentially Hazardous Object: MOID ≤ 0.05 au and absolute magnitude H ≤ 22.0. |
| MOID | Minimum orbit intersection distance — the closest two orbits ever get. |
| `moid_proxy_au` | This project's stand-in: the smallest *catalogued* approach distance. |
| `H` | Absolute magnitude: brightness the object would have at standard geometry; a proxy for size. |
| Geometric albedo `p_v` | Reflectivity at zero phase angle; assumed 0.14 when unmeasured. |
| Transit depth | Fractional dimming of a star, in ppm; equal to `(Rp/Rs)²`. |
| Impact parameter `b` | How centrally a transit crosses the stellar disk. |
| SNR | Signal-to-noise ratio of the detected transit signal. |
| Label leakage | A feature that encodes the answer; invisible in metrics, fatal in production. |
| Group | The unit of independence that must not be split: a `kepid` star, a `des` object. |
| Stratified | Each fold keeps the class proportions of the whole. |
| Holdout | Data untouched by selection and used once, for the published number. |
| macro-F1 | Mean of per-class F1 scores; each class gets one equal vote. |
| Average precision | Area under the precision-recall curve; the honest metric under imbalance. |
| ROC-AUC | Probability a positive outranks a negative; flattered by a large negative class. |
| Brier score | Mean squared error of the probability vector against the one-hot label. |
| Calibration | Whether predicted probabilities match observed frequencies. |
| Operating point | A threshold chosen to achieve a stated precision or recall. |
| Baseline | A trivial or published rule scored by the same pipeline as the model. |
| Permutation importance | Drop in the metric when one feature column is shuffled. |
| Train/serve skew | The served feature vector differing from the trained one; prevented here by a guard. |
| Content-addressed cache | Artifacts identified by the hash of their own bytes. |
| Dataset fingerprint | One hash over the whole manifest, stamped into every artifact. |
| `problem+json` | RFC 9457's media type for machine-readable error documents. |

## Where to go next

For the data sources themselves: the NASA Exoplanet Archive's Kepler services documentation
(`https://exoplanetarchive.ipac.caltech.edu/docs/kepler_services.html`) and its TAP/ADQL pages;
the JPL SBDB Close Approaches API reference (`https://ssd-api.jpl.nasa.gov/doc/cad.html`); and
CNEOS (`https://cneos.jpl.nasa.gov/`) for how hazard determination is actually done, which is
the standard this project's limitations section holds itself against.

For the methods: the scikit-learn user guide on cross-validation
(`https://scikit-learn.org/stable/modules/cross_validation.html`), including the section on
group-aware splitting; the model evaluation guide
(`https://scikit-learn.org/stable/modules/model_evaluation.html`) for the exact definitions of
macro/weighted averaging and `average_precision`; the imbalanced-learn documentation for
resampling strategies this project deliberately did not use; and the sklearn permutation
importance page, which discusses both the negative-value and correlated-feature traps of chapter
13 before you meet them in someone else's paper.

For the physics: Harris' chapter on the absolute magnitude to diameter relation is what
`D = 1329/√p · 10^(-H/5)` comes from, and the Exoplanet Archive's own column definitions
(`https://exoplanetarchive.ipac.caltech.edu/docs/kepcoldef.html`) explain the KOI columns
chapter 6 builds features from.

For the engineering: RFC 9457 for problem details, FastAPI's documentation on dependency
injection and app factories, and — closest to the spirit of chapter 4 — any writing on content
addressability, because the habit of making a claim checkable by a hash is the one idea in this
repository that transfers to everything else you will build.

The most useful next step, though, is the one a reviewer can take: clone the repository, run
`nasa-space-ml fetch && nasa-space-ml train`, and compare the two fingerprints. If they match,
every number in these documents is yours to check. If they do not, you have discovered — in about
ninety seconds, and with no trust required — that the archive moved. That comparison is the
entire design goal of this project, and it is the sentence the rest of the guide exists to make
meaningful.


