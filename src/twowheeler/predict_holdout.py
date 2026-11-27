"""Holdout prediction module for MP016 Used Two-Wheeler Price Prediction.

Loads the champion model (v2) from models/model.joblib, normalises text columns,
generates predictions on data/raw/holdout_features.csv using ONLY the 8 features,
validates assertions (250 rows, no NaNs, identical record_ids, finite numbers),
and saves to predictions/holdout_predictions.csv.

Usage:
    PYTHONPATH=src python3 -m twowheeler.predict_holdout
"""
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from .predict import FEATURES, load_model, predict_df


def normalise_holdout_text(df: pd.DataFrame, cat_cols=("brand_tier", "city_zone")) -> pd.DataFrame:
    """Normalise text columns the same way clean_train does (strip, lowercase, city_zone capitalised)."""
    df_clean = df.copy()
    for col in cat_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
    if "city_zone" in df_clean.columns:
        df_clean["city_zone"] = df_clean["city_zone"].str.capitalize()
    return df_clean


def predict_holdout(params_path: str = "params.yaml"):
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    holdout_cfg = params.get("holdout", {})
    train_cfg = params.get("train", {})
    data_cfg = params.get("data", {})

    model_path = Path(train_cfg.get("model_path", "models/model.joblib"))
    holdout_path = Path(
        holdout_cfg.get("holdout_path", data_cfg.get("raw_holdout", "data/raw/holdout_features.csv"))
    )
    predictions_path = Path(
        holdout_cfg.get("predictions_path", "predictions/holdout_predictions.csv")
    )

    # 1. Load the champion model and holdout data
    model = load_model(model_path)
    df = pd.read_csv(holdout_path)

    # 2. Normalise text columns (brand_tier, city_zone)
    cat_features = params.get("categorical_features", ["brand_tier", "city_zone"])
    df_clean = normalise_holdout_text(df, cat_cols=cat_features)

    # 3. Predict using only the 8 feature columns
    preds = predict_df(model, df_clean)

    # 4. Create predictions DataFrame with EXACTLY two columns: record_id, prediction
    preds_df = pd.DataFrame(
        {
            "record_id": df["record_id"],
            "prediction": preds,
        }
    )

    # 5. Assertions
    assert len(preds_df) == 250, f"Expected 250 rows, got {len(preds_df)}"
    assert not preds_df.isna().any().any(), "Predictions DataFrame contains NaN values"
    assert (
        preds_df["record_id"].tolist() == df["record_id"].tolist()
    ), "record_ids in output do not match input file or order"
    assert np.all(np.isfinite(preds_df["prediction"])), "Predictions contain non-finite values"

    # Save without index column
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    preds_df.to_csv(predictions_path, index=False)
    print(f"Generated {len(preds_df)} holdout predictions saved to: {predictions_path}")
    print("\nFirst 5 rows:")
    print(preds_df.head(5).to_string(index=False))

    return preds_df


def main(params_path: str = "params.yaml"):
    predict_holdout(params_path=params_path)


if __name__ == "__main__":
    main()
