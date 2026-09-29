import pytest
import requests

from lthc_dashboard.config import COUNTRY, ENGLISH, LANGUAGE, NOT_SPECIFIED, REGION, YEARS
from lthc_dashboard.inference import InferenceClient
from lthc_dashboard.prediction import (
    PredictionRequest,
    build_model_input,
    risk_level,
    run_prediction,
    validate_request,
)

CHINA = PredictionRequest(
    mode="country",
    country="China (excludes SARs and Taiwan)",
    years="More than 10 years",
    age="65 and over",
    sex="Female",
    condition="Diabetes",
)


def test_country_mode_fills_fields_like_the_training_rows(data):
    record = build_model_input(CHINA, data)
    assert record[ENGLISH] == NOT_SPECIFIED
    assert record[LANGUAGE] == data.lookups.country_language[CHINA.country]
    assert record[REGION] == "North-East Asia"


def test_language_mode_fills_fields_like_the_training_rows(data):
    request = PredictionRequest(
        mode="language",
        language="Vietnamese",
        english="Very well or well",
        age="45–64",
        sex="Persons",
        condition="Arthritis",
    )
    record = build_model_input(request, data)
    assert record[YEARS] == NOT_SPECIFIED
    assert record[COUNTRY] == "Vietnam"


def test_validation_catches_bad_input(data):
    bad = PredictionRequest(mode="country", country="Atlantis", years="Forever", age="90+", sex="?", condition="x")
    assert len(validate_request(bad, data)) == 5
    assert validate_request(CHINA, data) == []


@pytest.mark.parametrize(
    ("predicted", "benchmark", "key"),
    [
        (5, 10, "below"),
        (10, 10, "average"),
        (12.4, 10, "average"),
        (13, 10, "above"),
        (25, 10, "well-above"),
        (5, None, "unknown"),
    ],
)
def test_risk_level_bands(predicted, benchmark, key):
    assert risk_level(predicted, benchmark).key == key


def test_default_example_is_a_held_out_group_with_a_published_value(data, local_client):
    outcome = run_prediction(CHINA, data, local_client)
    assert outcome.split == "test"
    assert outcome.observed == pytest.approx(16.106, abs=0.001)
    assert outcome.group_population == 26_859
    assert 0 <= outcome.predicted <= 100
    assert len(outcome.profile) == 13 and len(outcome.trajectory) == 9
    assert outcome.served_by == "local"
    assert outcome.insights


def test_unpublished_and_unseen_group(data, local_client):
    request = PredictionRequest(
        mode="language",
        language="Catalan",
        english="Very well or well",
        age="45–64",
        sex="Persons",
        condition="Arthritis",
    )
    outcome = run_prediction(request, data, local_client)
    assert outcome.unseen_inputs
    assert any(text.startswith("Caution") for text in outcome.insights)


def test_unreachable_api_falls_back_to_bundled_model(data):
    client = InferenceClient(api_url="http://127.0.0.1:9", timeout=2)
    outcome = run_prediction(CHINA, data, client)
    assert outcome.served_by == "local"
    assert "could not be reached" in (outcome.note or "")


def test_api_without_batch_endpoint_uses_single_predictions(data, monkeypatch, model):

    class FakeResponse:
        def __init__(self, status, body=None):
            self.status_code, self._body = status, body

        def json(self):
            return self._body

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(str(self.status_code))

    def fake_post(url, json=None, timeout=None):
        if url.endswith("/predict/batch"):
            return FakeResponse(404)
        return FakeResponse(200, {"predicted_prevalence": model.predict_records([json])[0]})

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(
        requests.Session, "post", lambda self, url, json=None, timeout=None: fake_post(url, json, timeout)
    )
    outcome = run_prediction(CHINA, data, InferenceClient(api_url="http://old-api.example"))
    assert outcome.served_by == "api"
    local = run_prediction(CHINA, data, InferenceClient(api_url=""))
    assert outcome.predicted == local.predicted
