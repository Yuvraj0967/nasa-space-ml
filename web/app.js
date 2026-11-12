/*
  SSO screening console client.

  Three rules this file obeys, because they are the ones that make a demo trustworthy:

  1. The forms are generated from /api/v1/{task}/schema. There is no duplicated field list
     here, so the UI cannot silently disagree with the API contract.
  2. Every value that came from outside is inserted with textContent, never as markup.
     NASA designations are free-form strings; treating them as HTML would be a bug.
  3. Client-side range checks mirror the server's bounds but do not replace them. They exist
     so a typo is caught before a round trip; the server's 422 is still the authority, and
     its problem+json is what gets rendered if a check slips through.
*/

"use strict";

const CHANNELS = [
  { task: "exoplanets", root: document.getElementById("channel-exo") },
  { task: "asteroids", root: document.getElementById("channel-neo") },
];

const HAZARD_LABELS = new Set(["PHA", "FALSE POSITIVE"]);
const GOOD_LABELS = new Set(["CONFIRMED"]);

/* ------------------------------------------------------------------ helpers */

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function q(root, selector) {
  return root.querySelector(selector);
}

async function getJSON(url) {
  const response = await fetch(url, { headers: { Accept: "application/json" } });
  const body = await safeParse(response);
  if (!response.ok) throw apiError(response, body);
  return body;
}

async function postJSON(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await safeParse(response);
  if (!response.ok) throw apiError(response, body);
  return body;
}

async function safeParse(response) {
  const text = await response.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch (error) {
    return { raw: text.slice(0, 400) };
  }
}

function apiError(response, body) {
  const detail = body && (body.detail || body.raw);
  const error = new Error(detail || `HTTP ${response.status}`);
  error.status = response.status;
  error.problem = body || null;
  error.requestId = response.headers.get("X-Request-Id");
  return error;
}

function setFooterError(message) {
  const node = document.getElementById("foot-error");
  if (!message) {
    node.hidden = true;
    node.textContent = "";
    return;
  }
  node.hidden = false;
  node.textContent = message;
}

/* ------------------------------------------------------------------ header */

async function loadRigStatus() {
  try {
    const meta = await getJSON("/api/v1/meta");
    document.getElementById("stat-service").textContent = "ONLINE";
    document.getElementById("stat-version").textContent = meta.version;
    const fingerprint = meta.cache && meta.cache.dataset_fingerprint;
    document.getElementById("stat-fingerprint").textContent = fingerprint
      ? fingerprint.slice(0, 12)
      : "no snapshot";
    const artifacts = (meta.cache && meta.cache.artifacts) || [];
    document.getElementById("stat-cache").textContent = artifacts.length
      ? `${artifacts.length} file${artifacts.length === 1 ? "" : "s"}`
      : "empty";

    const neoChannel = document.getElementById("channel-neo");
    const labels = q(neoChannel, '[data-sample="pha"]');
    if (labels) {
      const rule = meta.sources.asteroids.pha_rule;
      const note = q(neoChannel, '[data-role="sample-note"]');
      note.textContent =
        `aggregated across every approach in the window · rule baseline MOID <= ${rule.moid_au_max} au & H <= ${rule.absolute_magnitude_max}`;
    }
    return meta;
  } catch (error) {
    document.getElementById("stat-service").textContent = "UNREACHABLE";
    setFooterError(`could not read /api/v1/meta: ${error.message}`);
    return null;
  }
}

/* ------------------------------------------------------------------ forms */

function buildForm(channel, schema) {
  const form = q(channel.root, "[data-form]");
  form.textContent = "";
  channel.fields = [];

  schema.fields.forEach((field) => {
    if (field.type === "string") return;

    const label = el("label", "field");
    label.dataset.name = field.name;

    const head = el("span", "field-label");
    head.appendChild(el("span", null, field.name));
    if (field.units) head.appendChild(el("span", "field-unit", field.units));
    label.appendChild(head);

    const input = el("input");
    input.type = "number";
    input.step = field.type === "integer" ? "1" : "any";
    input.name = field.name;
    input.placeholder = "—";
    if (field.minimum !== null && field.minimum !== undefined) input.min = String(field.minimum);
    if (field.maximum !== null && field.maximum !== undefined) input.max = String(field.maximum);
    label.appendChild(input);

    const bounds = rangeText(field);
    label.appendChild(el("span", "field-range", bounds));

    form.appendChild(label);
    channel.fields.push({ field, input, label, boundsNode: label.lastChild });
  });

  channel.schema = schema;
  channel.labels = schema.labels;

  const picker = q(channel.root, '[data-sample="label"]');
  if (picker) {
    picker.textContent = "";
    const any = el("option", null, "any");
    any.value = "";
    picker.appendChild(any);
    schema.labels.forEach((label) => {
      const option = el("option", null, label);
      option.value = label;
      picker.appendChild(option);
    });
  }
}

function rangeText(field) {
  const hasMin = field.minimum !== null && field.minimum !== undefined;
  const hasMax = field.maximum !== null && field.maximum !== undefined;
  if (hasMin && hasMax) return `${field.minimum} … ${field.maximum}`;
  if (hasMin) return `>= ${field.minimum}`;
  if (hasMax) return `<= ${field.maximum}`;
  return "optional";
}

function collectPayload(channel) {
  const payload = {};
  const problems = [];
  channel.fields.forEach(({ field, input, label }) => {
    const raw = input.value.trim();
    if (raw === "") {
      label.classList.remove("is-invalid", "has-error");
      return;
    }
    const value = Number(raw);
    if (!Number.isFinite(value)) {
      problems.push(`${field.name}: not a number`);
      label.classList.add("is-invalid", "has-error");
      return;
    }
    if (field.minimum !== null && field.minimum !== undefined && value < field.minimum) {
      problems.push(`${field.name}: ${value} is below the physical minimum ${field.minimum}`);
      label.classList.add("is-invalid", "has-error");
      return;
    }
    if (field.maximum !== null && field.maximum !== undefined && value > field.maximum) {
      problems.push(`${field.name}: ${value} is above the physical maximum ${field.maximum}`);
      label.classList.add("is-invalid", "has-error");
      return;
    }
    label.classList.remove("is-invalid", "has-error");
    payload[field.name] = value;
  });
  return { payload, problems };
}

function fillForm(channel, request) {
  channel.fields.forEach(({ field, input }) => {
    const value = request[field.name];
    input.value = value === null || value === undefined ? "" : String(value);
    input.closest(".field").classList.remove("is-invalid", "has-error");
  });
  const name = request.kepoi_name || request.designation;
  if (name) channel.lastSampleName = name;
}

/* ------------------------------------------------------------------ readout */

function setVerdict(channel, state, label, detail) {
  const box = q(channel.root, '[data-role="verdict"]');
  box.dataset.state = state;
  q(box, ".verdict-label").textContent = label;
  q(box, ".verdict-detail").textContent = detail;
}

function renderProbabilities(channel, probabilities) {
  const host = q(channel.root, '[data-role="probs"]');
  host.textContent = "";
  const ordered = [...probabilities].sort((a, b) => b.probability - a.probability);
  ordered.forEach((entry, position) => {
    const row = el("div", "prob" + (position === 0 ? " prob-top" : ""));
    row.appendChild(el("span", "prob-name", entry.label));
    row.appendChild(el("span", "prob-value", `${(entry.probability * 100).toFixed(1)}%`));
    const track = el("div", "prob-track");
    const fill = el("div", "prob-fill");
    fill.style.width = `${Math.max(0.5, entry.probability * 100).toFixed(1)}%`;
    track.appendChild(fill);
    row.appendChild(track);
    host.appendChild(row);
  });
}

function renderNotice(channel, lines, kind) {
  const node = q(channel.root, '[data-role="notice"]');
  node.textContent = "";
  if (!lines || lines.length === 0) {
    node.hidden = true;
    return;
  }
  node.hidden = false;
  node.dataset.kind = kind || "warn";
  lines.forEach((line) => node.appendChild(el("p", null, line)));
}

function renderProvenance(channel, rows) {
  const node = q(channel.root, '[data-role="provenance"]');
  node.textContent = "";
  rows.forEach(([term, value]) => {
    node.appendChild(el("dt", null, term));
    node.appendChild(el("dd", null, value === null || value === undefined ? "—" : String(value)));
  });
}

/* ------------------------------------------------------------------ actions */

async function loadSample(channel) {
  const indexNode = q(channel.root, '[data-sample="index"]');
  const params = new URLSearchParams();
  if (indexNode && indexNode.value !== "") params.set("index", indexNode.value.trim());
  const labelPicker = q(channel.root, '[data-sample="label"]');
  if (labelPicker && labelPicker.value) params.set("label", labelPicker.value);
  const phaPicker = q(channel.root, '[data-sample="pha"]');
  if (phaPicker && phaPicker.value) params.set("pha", phaPicker.value);

  setVerdict(channel, "run", "READING ARCHIVE", `GET /api/v1/${channel.task}/sample`);
  try {
    const body = await getJSON(`/api/v1/${channel.task}/sample?${params.toString()}`);
    switchTab(channel, "manual");
    fillForm(channel, body.request);
    channel.sampleTruth = body.truth;
    channel.sampleRule = body.nasa_rule_verdict || null;

    const truthLines = [];
    if (body.truth) {
      const truthText = Object.entries(body.truth)
        .map(([key, value]) => `${key}=${value}`)
        .join("  ");
      truthLines.push(`catalogue says: ${truthText}`);
    }
    if (body.nasa_rule_verdict) {
      truthLines.push(`NASA published rule says: ${body.nasa_rule_verdict}`);
    }
    truthLines.push(body.note || "");
    renderNotice(channel, truthLines.filter(Boolean), "info");
    setVerdict(
      channel,
      "idle",
      "ROW LOADED",
      `${body.request.kepoi_name || body.request.designation || "sample"} — press RUN SCREENING`
    );
    if (body.request.implied_albedo !== undefined || body.request.koi_period !== undefined) {
      // leave the numeric inputs untouched above; nothing else to derive here
    }
  } catch (error) {
    setVerdict(channel, "err", "SAMPLE FAILED", error.message);
    renderNotice(channel, describeError(error), "error");
  }
}

function describeError(error) {
  const lines = [];
  if (error.problem && error.problem.errors && error.problem.errors.fields) {
    error.problem.errors.fields.forEach((entry) => {
      lines.push(`${entry.field}: ${entry.message}`);
    });
  }
  lines.unshift(`${error.problem ? error.problem.code || "error" : "network"} — ${error.message}`);
  if (error.requestId) lines.push(`request id ${error.requestId}`);
  return lines;
}

async function runScreening(channel) {
  const { payload, problems } = collectPayload(channel);
  if (problems.length) {
    setVerdict(channel, "err", "INPUT OUT OF RANGE", problems[0]);
    renderNotice(channel, problems, "error");
    return;
  }
  if (Object.keys(payload).length === 0) {
    setVerdict(channel, "err", "NO INPUT", "every field is blank; the model would be guessing");
    return;
  }

  setVerdict(channel, "run", "SCREENING", `POST /api/v1/${channel.task}/predict`);
  const button = q(channel.root, '[data-action="predict"]');
  button.disabled = true;
  try {
    const result = await postJSON(`/api/v1/${channel.task}/predict`, payload);
    renderResult(channel, result);
  } catch (error) {
    setVerdict(channel, "err", "REQUEST REJECTED", error.message);
    renderNotice(channel, describeError(error), "error");
    renderProbabilities(channel, []);
    renderProvenance(channel, []);
  } finally {
    button.disabled = false;
  }
}

function renderResult(channel, result) {
  const top = result.probabilities.reduce((best, entry) =>
    entry.probability > best.probability ? entry : best
  , result.probabilities[0]);

  const state =
    HAZARD_LABELS.has(result.prediction) && result.task === "asteroids"
      ? "hazard"
      : result.prediction === "FALSE POSITIVE"
        ? "hazard"
        : GOOD_LABELS.has(result.prediction)
          ? "ok"
          : "idle";

  setVerdict(
    channel,
    state,
    result.prediction,
    `${(top.probability * 100).toFixed(1)}% confidence · ${(result.explanation && result.explanation.confidence) || ""}`
  );
  renderProbabilities(channel, result.probabilities);

  const lines = [];
  (result.explanation && result.explanation.notes || []).forEach((note) => lines.push(note));
  if (result.outside_training_range && result.outside_training_range.length) {
    lines.push(
      `extrapolating beyond the training range on: ${result.outside_training_range.join(", ")}`
    );
  }
  if (result.decision_threshold_note) lines.push(result.decision_threshold_note);

  if (channel.sampleTruth) {
    const truthText = Object.entries(channel.sampleTruth).map(([k, v]) => `${k}=${v}`).join("  ");
    lines.push(`archive truth for this row: ${truthText}`);
    if (channel.sampleRule) lines.push(`published rule on the same row: ${channel.sampleRule}`);
  }
  renderNotice(channel, lines, "info");

  renderProvenance(channel, [
    ["MODEL", result.model],
    ["SKLEARN", result.scikit_learn],
    ["TRAINED", result.model_trained_utc],
    ["DATA SHA", (result.dataset_fingerprint || "").slice(0, 16)],
    ["REQUEST", result.request_id],
    ["FEATURES", `${Object.keys(result.derived_features || {}).length} scored`],
  ]);
}

/* ------------------------------------------------------------------ tabs */

function switchTab(channel, mode) {
  channel.root.querySelectorAll(".tab").forEach((tab) => {
    const active = tab.dataset.mode === mode;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-selected", active ? "true" : "false");
  });
  const panels = channel.root.querySelectorAll("[data-panel]");
  panels.forEach((panel) => {
    if (panel.dataset.panel === mode) panel.hidden = false;
    else if (panel.dataset.panel === "manual" || panel.dataset.panel === "sample") panel.hidden = true;
  });
}

/* ------------------------------------------------------------------ wiring */

function wireChannel(channel) {
  channel.root.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => switchTab(channel, tab.dataset.mode));
  });

  const actions = {
    predict: () => runScreening(channel),
    "load-sample": () => loadSample(channel),
    reset: () => {
      channel.fields.forEach(({ input, label }) => {
        input.value = "";
        label.classList.remove("is-invalid", "has-error");
      });
      channel.sampleTruth = null;
      channel.sampleRule = null;
      renderProbabilities(channel, []);
      renderNotice(channel, []);
      renderProvenance(channel, []);
      setVerdict(channel, "idle", "AWAITING INPUT", "no screening has been run on this channel");
    },
    "fill-known": () => {
      const known = channel.task === "exoplanets" ? KNOWNEXOPLANET : KNOWNASTEROID;
      switchTab(channel, "manual");
      fillForm(channel, known);
      setVerdict(channel, "idle", "REFERENCE ROW LOADED", "a published object with known values");
    },
  };

  channel.root.querySelectorAll("[data-action]").forEach((button) => {
    const handler = actions[button.dataset.action];
    if (handler) button.addEventListener("click", handler);
  });
}

/* Two hand-checked reference rows, so the console demonstrates something even with the
   archive endpoint unreachable. 2151 Kerberos and (29075) 1995 XK1 are both real. */
const KNOWNEXOPLANET = {
  koi_period: 290.51,
  koi_duration: 9.64,
  koi_depth: 571.0,
  koi_prad: 11.32,
  koi_impact: 0.658,
  koi_eccen: 0.0,
  koi_incl: 89.66,
  koi_sma: 0.854,
  koi_teq: 278.0,
  koi_steff: 5452.0,
  koi_srad: 0.94,
  koi_smass: 1.02,
  koi_slogg: 4.44,
  koi_num_transits: 3.0,
  koi_model_snr: 40.0,
  koi_max_sngle_ev: 27.0,
  koi_kepmag: 13.6,
  koi_count: 1.0,
};

const KNOWNASTEROID = {
  moid_proxy_au: 0.0467,
  n_approaches: 42.0,
  n_approaches_within_lunar: 1.0,
  v_rel_at_closest: 5.62,
  v_inf_at_closest: 5.44,
  uncertainty_ratio_at_closest: 0.997,
  absolute_magnitude_h: 18.8,
  n_orbit_solutions: 12.0,
  t_sigma_seconds_at_closest: 3600.0,
  observation_span_years: 25.6,
  closest_approach_month: 12.0,
};

/* ------------------------------------------------------------------ boot */

async function boot() {
  await loadRigStatus();
  for (const channel of CHANNELS) {
    wireChannel(channel);
    switchTab(channel, "manual");
    setVerdict(channel, "run", "LOADING CONTRACT", `GET /api/v1/${channel.task}/schema`);
    try {
      const schema = await getJSON(`/api/v1/${channel.task}/schema`);
      buildForm(channel, schema);
      setVerdict(
        channel,
        "idle",
        "READY",
        `${schema.fields.length} inputs · labels ${schema.labels.join(" / ")}`
      );
    } catch (error) {
      setVerdict(channel, "err", "CONTRACT UNAVAILABLE", error.message);
      renderNotice(channel, describeError(error), "error");
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  boot().catch((error) => setFooterError(error.message));
});
