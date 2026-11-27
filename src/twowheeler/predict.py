"""Prediction module for MP016 Used Two-Wheeler Price Prediction."""
from pathlib import Path
from typing import Union
import joblib
import numpy as np
import pandas as pd

FEATURES = [
    "age_years",
    "km_driven",
    "engine_cc",
    "owners",
    "service_records",
    "accidents",
    "brand_tier",
    "city_zone",
]


def load_model(path: Union[str, Path] = "models/model.joblib"):
    """Load the trained scikit-learn Pipeline model from a joblib file."""
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return joblib.load(model_path)


def predict_df(model, df: pd.DataFrame) -> np.ndarray:
    """Predict resale price using the loaded model.

    Checks that all 8 required feature columns exist (raises ValueError naming missing ones),
    extracts only those columns in the required order, and returns a numpy array of predictions.
    """
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    X = df[FEATURES]
    preds = model.predict(X)
    return np.asarray(preds)
