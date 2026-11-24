"""Tests for the prediction API (app/main.py and app/schemas.py)."""
import io
import json
import logging

import pytest
from fastapi.testclient import TestClient

from app.logging_config import JsonFormatter, get_logger
from app.main import app
from app.schemas import FEATURES


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_valid_request(client, valid_bike, dummy_model):
    response = client.post("/predict", json=valid_bike)
    assert response.status_code == 200
    body = response.json()
    assert body["resale_thousand_rs"] == pytest.approx(42.5)
    # The model gets exactly the eight request features, in order.
    assert list(dummy_model.last_input.columns) == FEATURES


def test_age_above_25_is_rejected(client, valid_bike):
    valid_bike["age_years"] = 26
    assert client.post("/predict", json=valid_bike).status_code == 422


def test_age_exactly_25_is_allowed(client, valid_bike):
    valid_bike["age_years"] = 25
    assert client.post("/predict", json=valid_bike).status_code == 200


def test_engine_cc_below_50_is_rejected(client, valid_bike):
    valid_bike["engine_cc"] = 49
    assert client.post("/predict", json=valid_bike).status_code == 422


def test_engine_cc_exactly_50_is_allowed(client, valid_bike):
    valid_bike["engine_cc"] = 50
    assert client.post("/predict", json=valid_bike).status_code == 200


@pytest.mark.parametrize("field", FEATURES)
def test_missing_field_is_rejected(client, valid_bike, field):
    valid_bike.pop(field)
    assert client.post("/predict", json=valid_bike).status_code == 422


def test_extra_field_is_rejected(client, valid_bike):
    # The leaking column must never be accepted by the service.
    valid_bike["final_price_from_rc_transfer"] = 55.0
    assert client.post("/predict", json=valid_bike).status_code == 422


@pytest.mark.parametrize(
    "field, value",
    [("brand_tier", "luxury"), ("city_zone", "East"), ("km_driven", -1), ("accidents", -999)],
)
def test_bad_values_are_rejected(client, valid_bike, field, value):
    valid_bike[field] = value
    assert client.post("/predict", json=valid_bike).status_code == 422


def test_predict_without_model_returns_503(valid_bike, monkeypatch):
    monkeypatch.setattr("app.main.load_model", lambda *a, **k: None)
    app.state.model = None
    with TestClient(app) as c:
        assert c.get("/health").json() == {"status": "ok", "model_loaded": False}
        assert c.post("/predict", json=valid_bike).status_code == 503


def test_prediction_is_logged_as_json(client, valid_bike):
    # Capture the service logger's output with the same JSON formatter.
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = get_logger()
    logger.addHandler(handler)
    try:
        client.post("/predict", json=valid_bike)
    finally:
        logger.removeHandler(handler)

    records = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
    pred = [r for r in records if r.get("event") == "prediction"]
    assert len(pred) == 1
    assert pred[0]["prediction"] == pytest.approx(42.5)
    assert pred[0]["level"] == "INFO"
