"""Tests for twowheeler.predict."""
import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from twowheeler.predict import FEATURES, load_model, predict_df


@pytest.fixture
def valid_row():
    return {
        "age_years": 5.0,
        "km_driven": 30000,
        "engine_cc": 150,
        "owners": 1,
        "service_records": 6,
        "accidents": 0,
        "brand_tier": "mid",
        "city_zone": "North",
    }


def test_predict_on_valid_row(valid_row):
    model_path = Path("models/model.joblib")
    if not model_path.exists():
        pytest.skip("models/model.joblib does not exist")

    model = load_model(model_path)
    df = pd.DataFrame([valid_row])
    preds = predict_df(model, df)

    assert isinstance(preds, np.ndarray)
    assert len(preds) == 1
    val = float(preds[0])
    assert math.isfinite(val)


def test_predict_df_raises_value_error_on_missing_column(dummy_model, valid_row):
    df = pd.DataFrame([valid_row])
    df_missing = df.drop(columns=["age_years"])

    with pytest.raises(ValueError) as excinfo:
        predict_df(dummy_model, df_missing)
    assert "age_years" in str(excinfo.value)


def test_predict_df_raises_value_error_multiple_missing(dummy_model, valid_row):
    df = pd.DataFrame([valid_row])
    df_missing = df.drop(columns=["km_driven", "city_zone"])

    with pytest.raises(ValueError) as excinfo:
        predict_df(dummy_model, df_missing)
    assert "km_driven" in str(excinfo.value)
    assert "city_zone" in str(excinfo.value)


def test_predict_df_ignores_extra_columns_and_preserves_order(dummy_model, valid_row):
    # Add an irrelevant column and scramble column order
    row_with_extra = dict(valid_row)
    row_with_extra["irrelevant_col"] = 999
    cols = ["city_zone", "irrelevant_col"] + [c for c in FEATURES if c != "city_zone"]
    df = pd.DataFrame([row_with_extra])[cols]

    preds = predict_df(dummy_model, df)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == 1
    assert list(dummy_model.last_input.columns) == FEATURES
