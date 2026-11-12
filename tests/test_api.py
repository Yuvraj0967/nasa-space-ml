"""The HTTP layer, exercised end to end against the models committed in this repository.

Nothing here mocks a model. `create_app()` boots exactly the way `nasa-space-ml serve` does -
same config, same `data/raw` cache, same `models/*.joblib` - so these tests fail if the
artifact and the code that serves it disagree, which is the failure mode the registry guard
exists to prevent.

Two deliberate shapes:

  * a request body comes from `GET /sample` whenever one is needed, so the payload is a real
    NASA object rather than a number invented for the test;
  * the "degraded" app is a second, real app whose cache and model directory are empty,
    because "the service comes up and explains itself" is a promise this project makes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from nasa_space_ml import __version__
from nasa_space_ml.api.app import create_app, resolve_config_path, web_dir
from nasa_space_ml.api.deps import AppState
from nasa_space_ml.api.problem import TYPE_URIS
from nasa_space_ml.config import Config, resolve_default_config_path
from nasa_space_ml.data.cache import Cache
from nasa_space_ml.errors import ConfigError
from nasa_space_ml.features import asteroids as asteroid_features
from nasa_space_ml.features import exoplanets as exoplanet_features

API = "/api/v1"
REAL_CONFIG = Config.load(resolve_default_config_path())


def _models_trained() -> bool:
    return all(
        (REAL_CONFIG.paths.model_dir / f"{name}.joblib").exists()
        for name in ("exoplanet_disposition", "asteroid_pha")
    )


def _data_cached() -> bool:
    try:
        return bool(Cache(REAL_CONFIG.paths.raw_dir).manifest())
    except OSError:  # pragma: no cover - the directory is created by `ensure()`
        return False


requires_models = pytest.mark.skipif(
    not _models_trained(), reason="no trained models in this checkout; run `nasa-space-ml train`"
)
requires_cache = pytest.mark.skipif(
    not _data_cached(), reason="data/raw is empty in this checkout; run `nasa-space-ml fetch`"
)


@pytest.fixture(scope="module")
def client() -> Any:
    """A real service, lifespan included, shared across the module."""
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def exo_sample(client: TestClient) -> dict[str, Any]:
    if not _data_cached():
        pytest.skip("data/raw is empty; run `nasa-space-ml fetch`")
    response = client.get(f"{API}/exoplanets/sample")
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture(scope="module")
def ast_sample(client: TestClient) -> dict[str, Any]:
    if not _data_cached():
        pytest.skip("data/raw is empty; run `nasa-space-ml fetch`")
    response = client.get(f"{API}/asteroids/sample")
    assert response.status_code == 200, response.text
    return response.json()


class TestHealthAndMeta:
    def test_health_reports_both_models_ready(self, client: TestClient) -> None:
        body = client.get(f"{API}/health").json()
        assert body["status"] in {"ready", "degraded"}
        assert set(body["models"]) == {"exoplanet_disposition", "asteroid_pha"}
        assert body["version"] == __version__

    def test_health_carries_the_provenance_of_each_model(self, client: TestClient) -> None:
        for name, described in client.get(f"{API}/health").json()["models"].items():
            if not described["loaded"]:
                assert described["reason"]
                continue
            assert described["name"] == name
            assert described["trained_utc"]
            assert described["dataset_fingerprint"]
            assert described["scikit_learn"]
            assert described["task"] in {"exoplanets", "asteroids"}

    def test_meta_names_both_nasa_sources_and_their_contracts(self, client: TestClient) -> None:
        sources = client.get(f"{API}/meta").json()["sources"]
        assert "ipac.caltech.edu" in sources["exoplanets"]["endpoint"]
        assert sources["exoplanets"]["group_column"] == "kepid"
        assert sources["exoplanets"]["label_column"] == "koi_disposition"
        assert "jpl.nasa.gov" in sources["asteroids"]["endpoint"]
        assert sources["asteroids"]["group_column"] == "des"

    def test_meta_publishes_the_pha_rule_thresholds(self, client: TestClient) -> None:
        rule = client.get(f"{API}/meta").json()["sources"]["asteroids"]["pha_rule"]
        assert rule == {
            "moid_au_max": 0.05,
            "absolute_magnitude_max": 22.0,
            "assumed_geometric_albedo": 0.14,
        }

    def test_meta_feature_contract_matches_the_code(self, client: TestClient) -> None:
        contract = client.get(f"{API}/meta").json()["feature_contract"]
        assert contract["exoplanets"] == list(exoplanet_features.FEATURE_COLUMNS)
        assert contract["asteroids"] == list(asteroid_features.FEATURE_COLUMNS)

    def test_meta_declares_what_the_leakage_guard_blocks(self, client: TestClient) -> None:
        guard = client.get(f"{API}/meta").json()["leakage_guard"]
        assert "koi_pdisposition" in guard["exoplanets_forbidden_columns"]
        assert "koi_disposition" not in guard["exoplanets_forbidden_columns"]
        assert "preliminary verdict" in guard["why"]

    def test_meta_reports_the_cache_state_honestly(self, client: TestClient) -> None:
        cache = client.get(f"{API}/meta").json()["cache"]
        assert Path(cache["directory"]).is_dir()
        assert isinstance(cache["artifacts"], list)
        if cache["artifacts"]:
            assert len(cache["dataset_fingerprint"]) == 64

    def test_meta_split_block_is_the_config(self, client: TestClient) -> None:
        block = client.get(f"{API}/meta").json()["split"]
        assert block["seed"] == REAL_CONFIG.seed
        assert block["n_folds"] == REAL_CONFIG.split.n_folds
        assert block["test_size"] == pytest.approx(REAL_CONFIG.split.test_size)

    def test_api_root_indexes_every_endpoint(self, client: TestClient) -> None:
        body = client.get(API).json()
        assert body["service"] == "nasa-space-ml"
        assert len(body["endpoints"]) == 8
        assert all(line.split()[1].startswith(f"{API}/") for line in body["endpoints"])

    def test_openapi_and_swagger_are_served(self, client: TestClient) -> None:
        assert client.get("/openapi.json").status_code == 200
        assert client.get("/docs").status_code == 200


class TestSchemas:
    @pytest.mark.parametrize(
        ("task", "columns"),
        [
            ("exoplanets", exoplanet_features.FEATURE_COLUMNS),
            ("asteroids", asteroid_features.FEATURE_COLUMNS),
        ],
    )
    def test_schema_lists_features_and_labels(
        self, client: TestClient, task: str, columns: tuple[str, ...]
    ) -> None:
        body = client.get(f"{API}/{task}/schema").json()
        assert body["task"] == task
        assert body["endpoint"] == f"{API}/{task}/predict"
        assert body["feature_columns"] == list(columns)
        settings = REAL_CONFIG.exoplanets if task == "exoplanets" else REAL_CONFIG.asteroids
        assert body["labels"] == list(settings.labels)
        assert body["notes"]

    def test_exoplanet_schema_bounds_come_from_the_physics(self, client: TestClient) -> None:
        fields = {
            field["name"]: field
            for field in client.get(f"{API}/exoplanets/schema").json()["fields"]
        }
        assert fields["koi_period"]["minimum"] == pytest.approx(1e-4)
        assert fields["koi_period"]["units"] == "days"
        assert fields["koi_impact"]["maximum"] == pytest.approx(2.0)
        assert fields["koi_incl"]["maximum"] == pytest.approx(180.0)
        assert fields["koi_depth"]["units"] == "ppm"

    def test_no_field_is_required_because_blanks_mean_not_measured(
        self, client: TestClient
    ) -> None:
        for task in ("exoplanets", "asteroids"):
            for field in client.get(f"{API}/{task}/schema").json()["fields"]:
                assert field["required"] is False, field["name"]

    def test_the_label_column_is_absent_from_the_exoplanet_contract(
        self, client: TestClient
    ) -> None:
        names = {field["name"] for field in client.get(f"{API}/exoplanets/schema").json()["fields"]}
        assert "koi_disposition" not in names
        assert "koi_pdisposition" not in names

    def test_the_asteroid_contract_does_not_admit_its_own_label(self, client: TestClient) -> None:
        names = {field["name"] for field in client.get(f"{API}/asteroids/schema").json()["fields"]}
        assert "pha_flag" not in names
        assert "moid_proxy_au" in names
        assert "absolute_magnitude_h" in names


@requires_cache
@requires_models
class TestSamples:
    def test_exoplanet_sample_is_a_valid_request_body(
        self, client: TestClient, exo_sample: dict
    ) -> None:
        request_body = exo_sample["request"]
        assert request_body["kepoi_name"].startswith("K0")
        assert exo_sample["truth"]["koi_disposition"] in {
            "CANDIDATE",
            "CONFIRMED",
            "FALSE POSITIVE",
        }
        assert exo_sample["truth"]["kepid"]
        response = client.post(f"{API}/exoplanets/predict", json=request_body)
        assert response.status_code == 200, response.text

    def test_asteroid_sample_is_a_valid_request_body(
        self, client: TestClient, ast_sample: dict
    ) -> None:
        request_body = ast_sample["request"]
        assert request_body["designation"]
        assert ast_sample["nasa_rule_verdict"] in {"PHA", "NOT_PHA"}
        assert isinstance(ast_sample["truth"]["pha_flag"], bool)
        response = client.post(f"{API}/asteroids/predict", json=request_body)
        assert response.status_code == 200, response.text

    def test_truth_is_returned_but_never_read_by_the_predict_path(
        self, client: TestClient, exo_sample: dict
    ) -> None:
        """`truth` is demo scaffolding; sending it back must be rejected, not ignored."""
        poisoned = {**exo_sample["request"], "koi_disposition": "CONFIRMED"}
        response = client.post(f"{API}/exoplanets/predict", json=poisoned)
        assert response.status_code == 422
        assert response.json()["code"] == "RequestValidationError"

    def test_every_sample_key_is_a_field_the_predict_route_accepts(
        self, client: TestClient
    ) -> None:
        """`/sample` promises "a valid request body", and for the asteroid route it lied: the
        aggregation emits `diameter_from_h_km` and `diameter_inconsistency_km`, which are
        derived at predict time and therefore not request fields, so the demo's own payload
        came back as a 422. Checked structurally now, not by replaying one row."""
        from nasa_space_ml.api.schemas import AsteroidRequest, ExoplanetRequest

        for task, model in (("exoplanets", ExoplanetRequest), ("asteroids", AsteroidRequest)):
            body = client.get(f"{API}/{task}/sample").json()["request"]
            unknown = sorted(set(body) - set(model.model_fields))
            assert unknown == [], f"{task} sample sends {unknown}"

    def test_index_selects_deterministically(self, client: TestClient) -> None:
        first = client.get(f"{API}/exoplanets/sample", params={"index": 3}).json()
        again = client.get(f"{API}/exoplanets/sample", params={"index": 3}).json()
        assert first["request"] == again["request"]

    def test_index_is_wrapped_modulo_the_rows(self, client: TestClient) -> None:
        huge = client.get(f"{API}/exoplanets/sample", params={"index": 10**9}).json()
        assert huge["request"]["kepoi_name"]

    def test_label_filter_returns_only_that_disposition(self, client: TestClient) -> None:
        body = client.get(
            f"{API}/exoplanets/sample", params={"label": "FALSE POSITIVE", "index": 5}
        ).json()
        assert body["truth"]["koi_disposition"] == "FALSE POSITIVE"

    def test_pha_filter_returns_only_that_class(self, client: TestClient) -> None:
        body = client.get(f"{API}/asteroids/sample", params={"pha": True}).json()
        assert body["truth"]["pha_flag"] is True

    def test_an_unknown_disposition_is_a_502_problem(self, client: TestClient) -> None:
        response = client.get(f"{API}/exoplanets/sample", params={"label": "ALIEN"})
        assert response.status_code == 502
        assert response.json()["code"] == "DataError"
        # The order is the config's, not alphabetical: labels[0] is the negative class by
        # convention, and the error has to show the caller the contract as it is written.
        assert response.json()["errors"]["known"] == list(REAL_CONFIG.exoplanets.labels)

    def test_sample_works_with_no_parameters_at_all(self, client: TestClient) -> None:
        """The front end sends no `index` when the row box is blank, and the first version of
        these routes did `int(index)` anyway - a TypeError, so a 500, on the most-clicked
        button in the demo. Pinned here because the failure was silent in every other test."""
        for task in ("exoplanets", "asteroids"):
            response = client.get(f"{API}/{task}/sample")
            assert response.status_code == 200, response.text
            assert response.json()["request"]

    def test_an_unknown_candidate_names_the_rows_it_searched(self, client: TestClient) -> None:
        response = client.get(f"{API}/exoplanets/sample", params={"kepoi_name": "K99999.99"})
        assert response.status_code == 502
        assert response.json()["errors"]["rows"] >= 0

    def test_an_unknown_designation_is_a_502(self, client: TestClient) -> None:
        response = client.get(f"{API}/asteroids/sample", params={"designation": "1900 XK1"})
        assert response.status_code == 502
        assert "no close approaches" in response.json()["detail"]

    def test_a_negative_index_is_rejected_by_query_validation(self, client: TestClient) -> None:
        response = client.get(f"{API}/exoplanets/sample", params={"index": -1})
        assert response.status_code == 422
        assert response.json()["code"] == "RequestValidationError"


@requires_models
class TestExoplanetPredict:
    def test_probabilities_are_a_distribution_over_the_three_labels(
        self, client: TestClient, exo_sample: dict
    ) -> None:
        body = client.post(f"{API}/exoplanets/predict", json=exo_sample["request"]).json()
        assert {item["label"] for item in body["probabilities"]} == {
            "CANDIDATE",
            "CONFIRMED",
            "FALSE POSITIVE",
        }
        assert sum(item["probability"] for item in body["probabilities"]) == pytest.approx(1.0)
        assert (
            body["prediction"]
            == max(body["probabilities"], key=lambda item: item["probability"])["label"]
        )

    def test_classes_are_reported_in_the_models_own_column_order(
        self, client: TestClient, exo_sample: dict
    ) -> None:
        """`predict_proba` columns follow sorted `classes_`, not config order. The response
        has to say which order it used or the numbers are unreadable."""
        body = client.post(f"{API}/exoplanets/predict", json=exo_sample["request"]).json()
        assert body["classes_order"] == sorted(body["classes_order"])
        assert [item["label"] for item in body["probabilities"]] == body["classes_order"]

    def test_the_response_echoes_the_vector_that_was_actually_scored(
        self, client: TestClient, exo_sample: dict
    ) -> None:
        body = client.post(f"{API}/exoplanets/predict", json=exo_sample["request"]).json()
        assert set(body["derived_features"]) == set(exoplanet_features.FEATURE_COLUMNS)
        assert set(body["features_used"]) == set(exoplanet_features.RAW_INPUT_COLUMNS)

    def test_provenance_is_attached_to_every_answer(
        self, client: TestClient, exo_sample: dict
    ) -> None:
        body = client.post(f"{API}/exoplanets/predict", json=exo_sample["request"]).json()
        assert body["model"] == "exoplanet_disposition"
        assert body["dataset_fingerprint"]
        assert body["model_trained_utc"]
        assert body["request_id"]

    def test_an_empty_request_is_answered_not_rejected(self, client: TestClient) -> None:
        """Every field is optional: an all-blank row is a genuinely missing measurement, and
        the boosted model was trained with NaNs, so it must degrade honestly rather than 4xx."""
        response = client.post(f"{API}/exoplanets/predict", json={})
        assert response.status_code == 200
        body = response.json()
        assert len(body["probabilities"]) == 3
        assert all(value is None for value in body["features_used"].values())

    def test_the_explanation_block_says_how_marginal_the_call_is(
        self, client: TestClient, exo_sample: dict
    ) -> None:
        explanation = client.post(f"{API}/exoplanets/predict", json=exo_sample["request"]).json()[
            "explanation"
        ]
        assert explanation["confidence"] in {"clear", "weak: no class reaches 50%"}
        assert (
            explanation["confidence"]
            != "marginal: the two leading classes are within 5 percentage points"
        )
        assert 0.0 <= explanation["margin"] <= 1.0
        assert any("macro_f1" in note for note in explanation["notes"])

    def test_extrapolation_is_reported_instead_of_blocked(self, client: TestClient) -> None:
        """The training range for `log_prad` reaches ~26,000 Earth radii, because the KOI
        table contains eclipse-shaped things that are certainly not planets. So the value has
        to be genuinely outside the catalogue, not merely outside a plausible one - which is
        itself the argument for reporting the range instead of enforcing a prettier one.
        """
        response = client.post(
            f"{API}/exoplanets/predict", json={"koi_prad": 50_000.0, "koi_depth": 1_000_000.0}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["feature_ranges_available"] is True
        assert "log_prad" in body["outside_training_range"]
        assert "log_depth" in body["outside_training_range"]
        assert body["probabilities"]

    def test_a_value_inside_the_training_range_is_not_flagged(self, client: TestClient) -> None:
        body = client.post(
            f"{API}/exoplanets/predict",
            json={"koi_period": 3.5, "koi_depth": 1200.0, "koi_prad": 12.0, "koi_srad": 1.0},
        ).json()
        assert body["outside_training_range"] == []

    @pytest.mark.parametrize(
        ("field", "value", "reason"),
        [
            ("koi_period", -3.0, "greater than or equal to 0.0001"),
            ("koi_depth", 2_000_000.0, "less than or equal to 1000000"),
            ("koi_incl", 400.0, "less than or equal to 180"),
            ("koi_eccen", 1.0, "less than 1"),
            ("koi_teq", -500.0, "greater than or equal to -273.15"),
        ],
    )
    def test_physically_impossible_values_are_rejected_at_the_boundary(
        self, client: TestClient, exo_sample: dict, field: str, value: float, reason: str
    ) -> None:
        body = dict(exo_sample["request"])
        body[field] = value
        response = client.post(f"{API}/exoplanets/predict", json=body)
        assert response.status_code == 422, response.text
        problem = response.json()
        assert problem["code"] == "RequestValidationError"
        assert any(field in entry["field"] for entry in problem["errors"]["fields"])
        assert any(reason in entry["message"] for entry in problem["errors"]["fields"])

    def test_the_leakage_guard_is_enforced_by_the_request_model(self, client: TestClient) -> None:
        response = client.post(f"{API}/exoplanets/predict", json={"koi_pdisposition": "CONFIRMED"})
        assert response.status_code == 422
        assert "Extra inputs are not permitted" in json.dumps(response.json())


@requires_models
class TestAsteroidPredict:
    def test_probabilities_are_binary_over_the_pha_labels(
        self, client: TestClient, ast_sample: dict
    ) -> None:
        body = client.post(f"{API}/asteroids/predict", json=ast_sample["request"]).json()
        assert {item["label"] for item in body["probabilities"]} == {"NOT_PHA", "PHA"}
        assert sum(item["probability"] for item in body["probabilities"]) == pytest.approx(1.0)

    def test_the_response_shows_where_the_published_rule_stands(
        self, client: TestClient, ast_sample: dict
    ) -> None:
        """The rule is the reference; the model is an approximation of it. Saying so in the
        response is what stops a reader treating a disagreement as a bug."""
        body = client.post(f"{API}/asteroids/predict", json=ast_sample["request"]).json()
        assert "NASA's published rule" in body["decision_threshold_note"]
        assert "0.05" in body["decision_threshold_note"]
        assert "22.0" in body["decision_threshold_note"]

    def test_a_rule_obvious_pha_is_scored_as_one(self, client: TestClient) -> None:
        body = client.post(
            f"{API}/asteroids/predict",
            json={
                "designation": "1999 RQ36-like",
                "moid_proxy_au": 0.01,
                "absolute_magnitude_h": 18.0,
            },
        ).json()
        assert body["prediction"] == "PHA"
        assert "PHA" in body["decision_threshold_note"]

    def test_a_distant_bright_object_is_not_hazardous(self, client: TestClient) -> None:
        body = client.post(
            f"{API}/asteroids/predict",
            json={"designation": "far", "moid_proxy_au": 1.4, "absolute_magnitude_h": 26.5},
        ).json()
        assert body["prediction"] == "NOT_PHA"

    def test_the_rule_note_admits_when_it_cannot_be_applied(self, client: TestClient) -> None:
        body = client.post(f"{API}/asteroids/predict", json={"moid_proxy_au": 0.03}).json()
        assert "needs both MOID and H" in body["decision_threshold_note"]

    def test_diameter_is_derived_from_h_with_the_settings_albedo_not_the_callers(
        self, client: TestClient
    ) -> None:
        body = client.post(
            f"{API}/asteroids/predict", json={"absolute_magnitude_h": 20.0, "moid_proxy_au": 0.02}
        ).json()
        expected = 1329.0 / (0.14**0.5) * 10.0 ** (-20.0 / 5.0)
        assert body["derived_features"]["diameter_from_h_km"] == pytest.approx(expected, rel=1e-6)

    def test_a_caller_cannot_inject_the_settings_albedo(self, client: TestClient) -> None:
        """`implied_albedo` is a feature a caller may supply; the albedo *constant* is not.

        If the conversion factor were request-controlled, the same object would score
        differently depending on who asked - which would make the model un-auditable.
        """
        response = client.post(f"{API}/asteroids/predict", json={"default_geometric_albedo": 0.9})
        assert response.status_code == 422
        assert "Extra inputs" in json.dumps(response.json())

    def test_the_month_is_rejected_when_outside_a_calendar(self, client: TestClient) -> None:
        response = client.post(f"{API}/asteroids/predict", json={"closest_approach_month": 13})
        assert response.status_code == 422

    def test_seasonality_encoding_is_on_the_unit_circle(self, client: TestClient) -> None:
        body = client.post(
            f"{API}/asteroids/predict", json={"closest_approach_month": 7, "moid_proxy_au": 0.02}
        ).json()
        sin = body["derived_features"]["closest_month_sin"]
        cos = body["derived_features"]["closest_month_cos"]
        assert sin * sin + cos * cos == pytest.approx(1.0, abs=1e-9)

    def test_an_empty_asteroid_request_still_answers(self, client: TestClient) -> None:
        response = client.post(f"{API}/asteroids/predict", json={})
        assert response.status_code == 200
        assert len(response.json()["probabilities"]) == 2


class TestRequestIds:
    def test_a_generated_id_is_returned_on_every_response(self, client: TestClient) -> None:
        response = client.get(f"{API}/health")
        identifier = response.headers["X-Request-Id"]
        assert len(identifier) == 16
        assert int(identifier, 16) >= 0
        assert response.headers["Cache-Control"] == "no-store"
        assert float(response.headers["X-Response-Time-Ms"]) >= 0.0

    def test_a_supplied_id_is_echoed_not_replaced(self, client: TestClient) -> None:
        response = client.get(f"{API}/meta", headers={"X-Request-Id": "incident-0341-abc"})
        assert response.headers["X-Request-Id"] == "incident-0341-abc"

    def test_an_absurd_supplied_id_is_replaced(self, client: TestClient) -> None:
        long = "x" * 500
        assert (
            client.get(f"{API}/health", headers={"X-Request-Id": long}).headers["X-Request-Id"]
            != long
        )

    def test_the_id_appears_in_the_error_body(self, client: TestClient) -> None:
        response = client.post(
            f"{API}/exoplanets/predict",
            json={"koi_incl": 999.0},
            headers={"X-Request-Id": "trace-me"},
        )
        assert response.status_code == 422
        assert response.json()["request_id"] == "trace-me"
        assert (
            response.json()["type"].endswith("errors.md")
            or "docs/errors.md" in response.json()["type"]
        )


class TestProblemJson:
    def test_validation_errors_use_rfc_9457_media_type(self, client: TestClient) -> None:
        response = client.post(f"{API}/exoplanets/predict", json={"koi_period": "not-a-number"})
        assert response.status_code == 422
        assert response.headers["content-type"].startswith("application/problem+json")
        body = response.json()
        assert body["status"] == 422
        assert body["title"] == "Invalid request body"
        assert body["instance"] == f"{API}/exoplanets/predict"

    def test_the_error_body_names_the_offending_field_only(self, client: TestClient) -> None:
        body = client.post(f"{API}/asteroids/predict", json={"moid_proxy_au": 99.0}).json()
        entry = body["errors"]["fields"][0]
        assert entry["field"] == "moid_proxy_au"
        assert entry["in"] == "body"
        assert entry["input"] == 99.0

    def test_a_mismatched_type_is_reported_without_leaking_a_traceback(
        self, client: TestClient
    ) -> None:
        body = client.post(f"{API}/exoplanets/predict", json={"kepoi_name": {"nested": 1}}).json()
        assert "Traceback" not in json.dumps(body)
        assert body["errors"]["fields"]

    def test_unknown_routes_get_fastapis_own_404(self, client: TestClient) -> None:
        assert client.get(f"{API}/nope").status_code == 404


class TestFrontendIsServed:
    def test_index_is_served_from_the_root(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert "space object screening console" in response.text.lower()
        assert "/static/app.js" in response.text

    @pytest.mark.parametrize("asset", ["app.js", "styles.css"])
    def test_static_assets_are_mounted(self, client: TestClient, asset: str) -> None:
        assert client.get(f"/static/{asset}").status_code == 200


@pytest.fixture(scope="module")
def degraded_client(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """The same factory, pointed at an empty checkout: no cache, no models."""
    directory = tmp_path_factory.mktemp("empty-checkout")
    config_file = directory / "configs" / "settings.yaml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(yaml.safe_dump(REAL_CONFIG.raw), encoding="utf-8")
    with TestClient(create_app(config_file)) as test_client:
        yield test_client


class TestDegradedService:
    """A checkout with no models and no cached data must still start, and must say why."""

    def test_the_service_boots_without_any_models(self, degraded_client: TestClient) -> None:
        assert degraded_client.get(f"{API}/health").status_code == 200

    def test_health_says_degraded_and_lists_the_reason(self, degraded_client: TestClient) -> None:
        body = degraded_client.get(f"{API}/health").json()
        assert body["status"] == "degraded"
        for described in body["models"].values():
            assert described["loaded"] is False
            assert "ModelNotTrainedError" in described["reason"]

    def test_predicting_without_a_model_is_a_503_not_a_500(
        self, degraded_client: TestClient
    ) -> None:
        response = degraded_client.post(f"{API}/exoplanets/predict", json={"koi_prad": 9.0})
        assert response.status_code == 503
        assert response.headers["content-type"].startswith("application/problem+json")
        body = response.json()
        assert body["code"] == "ModelNotTrainedError"
        assert body["type"].endswith("docs/errors.md#not-trained")

    def test_the_503_tells_the_operator_what_to_run(self, degraded_client: TestClient) -> None:
        body = degraded_client.post(f"{API}/asteroids/predict", json={}).json()
        assert "nasa-space-ml train" in body["detail"]

    def test_every_problem_type_uri_is_clickable_and_lands_somewhere(self) -> None:
        """`type` now promises a real URL, so both halves of the promise are checked.

        This is not a link checker - it would be testing GitHub's HTML rather than this
        repository's contract. What it does test is the half that rots quietly: renaming an
        error class, rewriting a heading, or dropping the `/blob/main/` prefix breaks a
        reviewer's click and nothing else would notice.
        """
        doc = Path(__file__).resolve().parents[1] / "docs" / "errors.md"
        text = doc.read_text(encoding="utf-8")
        for code, uri in TYPE_URIS.items():
            base, _, anchor = uri.partition("#")
            assert base.endswith("/blob/main/docs/errors.md"), f"{code}: {uri}"
            assert f'<a name="{anchor}"></a>' in text, f"{code}: errors.md has no #{anchor}"

    def test_meta_still_describes_the_contract(self, degraded_client: TestClient) -> None:
        body = degraded_client.get(f"{API}/meta").json()
        assert body["feature_contract"]["exoplanets"] == list(exoplanet_features.FEATURE_COLUMNS)
        assert body["models"]["asteroid_pha"]["loaded"] is False

    def test_the_schema_endpoints_work_without_models(self, degraded_client: TestClient) -> None:
        """Forms can still render: they describe the contract, not the artifact."""
        for task in ("exoplanets", "asteroids"):
            assert degraded_client.get(f"{API}/{task}/schema").status_code == 200

    def test_samples_fail_as_502_because_the_cache_is_empty(
        self, degraded_client: TestClient
    ) -> None:
        response = degraded_client.get(f"{API}/exoplanets/sample")
        assert response.status_code == 502
        assert response.json()["code"] in {"DataError", "ModelNotTrainedError"}


class TestAsteroidTableMemo:
    """`/sample` used to re-aggregate all 32,390 close approaches on every click - 13 s per
    request, on the button a reviewer presses first. The tables are now held in app state and
    dropped when the dataset fingerprint changes, so these tests pin both halves."""

    def _state(self, config: Config, event_cache: Cache) -> AppState:
        return AppState(config=config, cache=event_cache, started_utc="2026-11-12T00:00:00+00:00")

    def test_the_tables_are_built_once_and_reused(
        self, config: Config, event_cache: Cache, cad_raw_events: pd.DataFrame
    ) -> None:
        state = self._state(config, event_cache)
        events, objects = state.asteroid_tables()
        again_events, again_objects = state.asteroid_tables()
        assert events is again_events and objects is again_objects
        assert len(events) == len(cad_raw_events)
        assert len(objects) == cad_raw_events["des"].nunique()

    def test_a_new_fingerprint_forces_a_recompute(
        self, config: Config, event_cache: Cache, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = self._state(config, event_cache)
        _events, objects = state.asteroid_tables()
        monkeypatch.setattr(event_cache, "dataset_fingerprint", lambda: "f" * 64)
        _events2, rebuilt = state.asteroid_tables()
        assert rebuilt is not objects
        assert len(rebuilt) == len(objects)

    def test_the_cached_table_is_the_same_one_a_direct_call_produces(
        self, config: Config, event_cache: Cache
    ) -> None:
        """Caching must not become a second implementation of the aggregation."""
        state = self._state(config, event_cache)
        events, objects = state.asteroid_tables()
        expected = asteroid_features.aggregate_events(events, config)
        pd.testing.assert_frame_equal(objects, expected)

    @requires_cache
    @requires_models
    def test_repeated_sample_requests_share_one_aggregation(self, client: TestClient) -> None:
        state = client.app.state.nsm
        client.get(f"{API}/asteroids/sample")
        _events, first = state.asteroid_tables()
        client.get(f"{API}/asteroids/sample", params={"index": 7, "pha": False})
        _events2, second = state.asteroid_tables()
        assert second is first


class TestEnvironmentOverrides:
    """What the Dockerfile relies on: a container cannot keep the repository's layout."""

    def test_nsm_config_wins_over_the_package_relative_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checkout = tmp_path / "image"
        (checkout / "configs").mkdir(parents=True)
        config_file = checkout / "configs" / "settings.yaml"
        config_file.write_text(yaml.safe_dump(REAL_CONFIG.raw), encoding="utf-8")
        monkeypatch.setenv("NSM_CONFIG", str(config_file))

        assert resolve_config_path() == config_file
        with TestClient(create_app()) as test_client:
            state = test_client.app.state.nsm
            assert state.config.root == checkout.resolve()
            assert test_client.get(f"{API}/health").status_code == 200

    def test_an_explicit_argument_beats_the_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NSM_CONFIG", str(tmp_path / "from_env.yaml"))
        assert resolve_config_path(tmp_path / "from_arg.yaml") == tmp_path / "from_arg.yaml"

    def test_no_environment_and_no_argument_keeps_the_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("NSM_CONFIG", raising=False)
        assert resolve_config_path() is None
        assert resolve_config_path(None) is None

    def test_nsm_web_dir_wins_over_the_repository_layout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        alt = tmp_path / "assets"
        alt.mkdir()
        (alt / "index.html").write_text("<!doctype html><title>alt</title>", encoding="utf-8")
        monkeypatch.setenv("NSM_WEB_DIR", str(alt))
        assert web_dir() == alt

    def test_the_page_is_only_mounted_when_it_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A site-packages install with no `web/` copied in must still serve the API."""
        from nasa_space_ml.api import app as app_module

        monkeypatch.setattr(app_module, "web_candidates", lambda: [tmp_path / "not-copied"])
        assert web_dir() is None
        with TestClient(create_app()) as test_client:
            assert test_client.get(f"{API}/health").status_code == 200
            assert test_client.get("/").status_code == 404
            assert test_client.get(f"{API}/exoplanets/schema").status_code == 200

    def test_a_broken_env_override_falls_through_to_the_next_candidate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A typo in `NSM_WEB_DIR` should not cost the reviewer the console."""
        monkeypatch.setenv("NSM_WEB_DIR", str(tmp_path / "typo"))
        assert web_dir() is not None
        with TestClient(create_app()) as test_client:
            assert test_client.get("/").status_code == 200

    def test_the_package_relative_candidate_beats_the_working_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The property is the *order* of the candidates, not this machine's layout.

        Comparing `web_dir()` against the `web/` next to this checkout only holds for an
        editable install. A wheel in site-packages resolves the same function somewhere else
        entirely, which is what the repository's second CI run found when a test that passed
        at home failed on a runner. Two fabricated candidate directories make the precedence
        assertable in any install layout.
        """
        from nasa_space_ml.api import app as app_module

        package_web = tmp_path / "install" / "web"
        checkout_web = tmp_path / "checkout" / "web"
        for directory in (package_web, checkout_web):
            directory.mkdir(parents=True)
            (directory / "index.html").write_text("<!doctype html>", encoding="utf-8")
        monkeypatch.delenv("NSM_WEB_DIR", raising=False)
        monkeypatch.setattr(app_module, "web_candidates", lambda: [package_web, checkout_web])

        assert web_dir() == package_web


class TestAppFactory:
    def test_two_apps_do_not_share_state(self) -> None:
        first, second = create_app(), create_app()
        with TestClient(first), TestClient(second):
            assert first.state.nsm is not second.state.nsm

    def test_the_module_level_app_exists_for_uvicorn_import(self) -> None:
        import nasa_space_ml.api.app as app_module

        assert app_module.app.title == "nasa-space-ml"

    def test_a_missing_config_file_is_a_config_error_at_construction(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="config file not found"):
            create_app(tmp_path / "nowhere.yaml")

    def test_the_web_directory_is_resolved_relative_to_the_package(self) -> None:
        web_root = Path(__file__).resolve().parents[1] / "web"
        assert web_dir().samefile(web_root)
        assert (web_root / "index.html").is_file()

    def test_the_page_loads_nothing_from_the_internet(self) -> None:
        """`app.py` promises the demo survives a laptop with the wifi off, so the promise is
        checked: no external script or stylesheet origin anywhere in the shipped page."""
        text = (web_dir() / "index.html").read_text(encoding="utf-8")
        external = re.findall(r'(?:src|href)\s*=\s*["\']https?://[^"\']+', text)
        assert external == []
        assert "cdn." not in text.lower()
