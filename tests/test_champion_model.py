"""Smoke test: the champion model committed in models/ works through the API.

Skipped when models/model.joblib is not present, so the rest of the
suite still runs on a fresh clone without the model file.
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app, load_model

CHAMPION = Path(__file__).resolve().parents[1] / "models" / "model.joblib"

pytestmark = pytest.mark.skipif(not CHAMPION.exists(), reason="models/model.joblib not present")


@pytest.fixture
def champion_client():
    app.state.model = load_model(CHAMPION)
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.state.model = None


def test_champion_model_loads():
    model = load_model(CHAMPION)
    assert model is not None
    assert hasattr(model, "predict")


def test_champion_prediction_is_a_sensible_price(champion_client, valid_bike):
    response = champion_client.post("/predict", json=valid_bike)
    assert response.status_code == 200
    price = response.json()["resale_thousand_rs"]
    # A price in thousand rupees: positive and well below 10 lakh.
    assert 0 < price < 1000


def test_older_bike_is_not_worth_more(champion_client, valid_bike):
    newer = dict(valid_bike, age_years=2)
    older = dict(valid_bike, age_years=15)
    p_new = champion_client.post("/predict", json=newer).json()["resale_thousand_rs"]
    p_old = champion_client.post("/predict", json=older).json()["resale_thousand_rs"]
    assert p_old <= p_new
