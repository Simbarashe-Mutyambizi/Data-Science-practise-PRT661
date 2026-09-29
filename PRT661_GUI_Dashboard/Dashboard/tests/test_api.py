import pytest
from fastapi.testclient import TestClient

from api.main import app

EXAMPLE = {
    "country_of_birth": "India",
    "years_in_australia": "More than 10 years",
    "age_group": "45–64",
    "sex": "Female",
    "lthc": "Diabetes",
    "language": "Kannada",
    "english_proficiency": "Not specified",
    "region": "Southern and Central Asia",
    "subregion": "Southern Asia",
}


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health(client):
    assert client.get("/").json() == {"message": "LTHC Prediction API is running"}
    assert client.get("/health").json()["status"] == "running"


def test_predict_keeps_original_response_keys(client, model):
    body = client.post("/predict", json=EXAMPLE).json()
    for key in ("predicted_prevalence", "unit", "interpretation", "warning"):
        assert key in body
    assert body["unit"] == "percent"
    assert body["predicted_prevalence"] == model.predict_records([EXAMPLE])[0]
    assert body["unseen_inputs"] == []


def test_batch_matches_single_predictions(client):
    records = [EXAMPLE, {**EXAMPLE, "age_group": "65 and over"}, {**EXAMPLE, "lthc": "Arthritis"}]
    batch = client.post("/predict/batch", json={"records": records}).json()["predictions"]
    singles = [client.post("/predict", json=r).json()["predicted_prevalence"] for r in records]
    assert [item["predicted_prevalence"] for item in batch] == singles


def test_missing_field_is_rejected(client):
    incomplete = {k: v for k, v in EXAMPLE.items() if k != "sex"}
    assert client.post("/predict", json=incomplete).status_code == 422
