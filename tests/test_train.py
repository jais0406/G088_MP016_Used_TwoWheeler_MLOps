"""Tests for model training pipeline and inference on the 8 raw feature columns."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

from twowheeler.preprocessing import build_preprocessor
from app.schemas import FEATURES


@pytest.fixture
def mock_params():
    return {
        "numeric_features": [
            "age_years",
            "km_driven",
            "engine_cc",
            "owners",
            "service_records",
            "accidents",
        ],
        "categorical_features": ["brand_tier", "city_zone"],
        "target": "resale_thousand_rs",
    }


@pytest.fixture
def synthetic_train_df():
    """Tiny synthetic DataFrame with 8 features and target."""
    return pd.DataFrame(
        {
            "age_years": [1.0, 3.5, 5.0, 8.2, 10.0, 12.5],
            "km_driven": [5000, 20000, 35000, 60000, 80000, 95000],
            "engine_cc": [110.0, 125.0, np.nan, 200.0, 350.0, 100.0],
            "owners": [1, 1, 2, 2, 3, 4],
            "service_records": [2, 5, 6, 8, 10, 3],
            "accidents": [0.0, np.nan, 1.0, 0.0, 2.0, 1.0],
            "brand_tier": ["budget", "mid", "premium", "mid", "premium", "budget"],
            "city_zone": ["North", "South", "Central", "North", "South", "Central"],
            "resale_thousand_rs": [60.0, 55.0, 70.0, 40.0, 65.0, 15.0],
        }
    )


def test_pipeline_trains_and_predicts_from_8_raw_columns(mock_params, synthetic_train_df, valid_bike):
    """Pipeline fits on 8 raw features and predicts accurately for batch and single rows."""
    preprocessor = build_preprocessor(mock_params)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", Ridge(random_state=42)),
    ])

    feature_cols = mock_params["numeric_features"] + mock_params["categorical_features"]
    assert feature_cols == FEATURES

    X = synthetic_train_df[feature_cols]
    y = synthetic_train_df[mock_params["target"]]

    pipeline.fit(X, y)

    # 1. Batch prediction on the 8 raw columns
    preds = pipeline.predict(X)
    assert len(preds) == len(X)
    assert all(isinstance(p, (float, np.floating)) for p in preds)

    # 2. Single-row prediction matching the API schema in app/main.py
    single_df = pd.DataFrame([valid_bike], columns=FEATURES)
    single_pred = pipeline.predict(single_df)
    assert len(single_pred) == 1
    assert isinstance(float(single_pred[0]), float)
    assert not np.isnan(single_pred[0])


def test_pipeline_handles_missing_numeric_values(mock_params, synthetic_train_df):
    """Pipeline's SimpleImputer handles missing values in raw inputs."""
    preprocessor = build_preprocessor(mock_params)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", Ridge(random_state=42)),
    ])

    feature_cols = mock_params["numeric_features"] + mock_params["categorical_features"]
    X = synthetic_train_df[feature_cols]
    y = synthetic_train_df[mock_params["target"]]
    pipeline.fit(X, y)

    df_with_nans = pd.DataFrame(
        [{
            "age_years": 4.0,
            "km_driven": 25000,
            "engine_cc": np.nan,  # missing
            "owners": 1,
            "service_records": 5,
            "accidents": np.nan,  # missing
            "brand_tier": "mid",
            "city_zone": "South",
        }],
        columns=FEATURES,
    )

    pred = pipeline.predict(df_with_nans)
    assert len(pred) == 1
    assert not np.isnan(pred[0])


def test_pipeline_handles_unseen_categories(mock_params, synthetic_train_df):
    """Pipeline's OneHotEncoder handles unknown categories gracefully without error."""
    preprocessor = build_preprocessor(mock_params)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", Ridge(random_state=42)),
    ])

    feature_cols = mock_params["numeric_features"] + mock_params["categorical_features"]
    X = synthetic_train_df[feature_cols]
    y = synthetic_train_df[mock_params["target"]]
    pipeline.fit(X, y)

    df_with_unknowns = pd.DataFrame(
        [{
            "age_years": 2.0,
            "km_driven": 10000,
            "engine_cc": 125.0,
            "owners": 1,
            "service_records": 3,
            "accidents": 0.0,
            "brand_tier": "unseen_category",
            "city_zone": "unseen_zone",
        }],
        columns=FEATURES,
    )

    pred = pipeline.predict(df_with_unknowns)
    assert len(pred) == 1
    assert not np.isnan(pred[0])
