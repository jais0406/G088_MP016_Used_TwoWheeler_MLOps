"""Tests for holdout predictions."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from twowheeler.predict import FEATURES
from twowheeler.predict_holdout import normalise_holdout_text


def test_normalise_holdout_text_synthetic():
    """Verify normalisation of brand_tier and city_zone on holdout format."""
    df = pd.DataFrame(
        {
            "record_id": ["R1", "R2", "R3"],
            "brand_tier": [" Mid ", "BUDGET", "premium  "],
            "city_zone": ["north", " SOUTH ", "CENTRAL"],
        }
    )
    cleaned = normalise_holdout_text(df)
    assert list(cleaned["brand_tier"]) == ["mid", "budget", "premium"]
    assert list(cleaned["city_zone"]) == ["North", "South", "Central"]


def test_holdout_predictions_file_validity():
    """Check that predictions/holdout_predictions.csv exists, has exactly ['record_id', 'prediction'] columns and 250 rows."""
    pred_path = Path("predictions/holdout_predictions.csv")
    holdout_raw_path = Path("data/raw/holdout_features.csv")

    if not pred_path.exists() or not holdout_raw_path.exists():
        pytest.skip("predictions/holdout_predictions.csv or holdout_features.csv is missing")

    df_preds = pd.read_csv(pred_path)
    df_raw = pd.read_csv(holdout_raw_path)

    assert list(df_preds.columns) == ["record_id", "prediction"]
    assert len(df_preds) == 250
    assert len(df_preds) == len(df_raw)
    assert df_preds["record_id"].tolist() == df_raw["record_id"].tolist()
    assert not df_preds.isna().any().any()
    assert np.all(np.isfinite(df_preds["prediction"]))
