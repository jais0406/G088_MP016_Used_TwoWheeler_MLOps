"""Check the service can load and use a real joblib model file.

A tiny scikit-learn pipeline is trained on made-up rows inside the test,
so no CSV or real model file is needed.
"""
import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.main import app, load_model
from app.schemas import FEATURES

NUMERIC = FEATURES[:6]
CATEGORICAL = ["brand_tier", "city_zone"]


def _tiny_model():
    rows = pd.DataFrame(
        {
            "age_years": [1, 3, 5, 8, 10, 12],
            "km_driven": [5000, 20000, 35000, 60000, 80000, 95000],
            "engine_cc": [110, 125, 150, 200, 350, 100],
            "owners": [1, 1, 2, 2, 3, 4],
            "service_records": [2, 5, 6, 8, 10, 3],
            "accidents": [0, 0, 1, 0, 2, 1],
            "brand_tier": ["budget", "mid", "premium", "mid", "premium", "budget"],
            "city_zone": ["North", "South", "Central", "North", "South", "Central"],
        }
    )
    target = [60, 55, 70, 40, 65, 15]
    pre = ColumnTransformer(
        [
            ("num", "passthrough", NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return Pipeline([("pre", pre), ("reg", LinearRegression())]).fit(rows, target)


def test_load_model_missing_file_returns_none(tmp_path):
    assert load_model(tmp_path / "nope.joblib") is None


def test_service_uses_saved_joblib_model(tmp_path, valid_bike):
    path = tmp_path / "model.joblib"
    model = _tiny_model()
    joblib.dump(model, path)

    loaded = load_model(path)
    assert loaded is not None

    expected = float(model.predict(pd.DataFrame([valid_bike], columns=FEATURES))[0])
    app.state.model = loaded
    try:
        with TestClient(app) as c:
            response = c.post("/predict", json=valid_bike)
    finally:
        app.state.model = None

    assert response.status_code == 200
    assert response.json()["resale_thousand_rs"] == pytest.approx(expected)
