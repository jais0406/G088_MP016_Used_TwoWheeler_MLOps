"""Retraining module for MP016 Used Two-Wheeler Price Prediction.

Loads champion v1 model, splits later.csv by date (70% train / 30% check),
retrains v2 on train_clean + first70, compares MAE on last30,
applies client decision rule (promote if mae_v2 <= 0.88 * mae_v1),
logs to MLflow, registers model in MLflow Model Registry with alias 'champion',
and writes models/comparison.json.

Usage:
    python3 -m twowheeler.retrain
"""
import json
from pathlib import Path
import re
import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
import yaml
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from .predict import FEATURES, load_model


def normalise_text_columns(df: pd.DataFrame, categorical_features: list) -> pd.DataFrame:
    """Normalise text columns the same way clean_train does (strip, lower, city_zone capitalized)."""
    df_clean = df.copy()
    for col in categorical_features:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
    if "city_zone" in df_clean.columns:
        df_clean["city_zone"] = df_clean["city_zone"].str.capitalize()
    return df_clean


def split_later_by_date(df: pd.DataFrame, split_ratio: float = 0.7):
    """Sort DataFrame by date and split into (first70, last30)."""
    df_sorted = df.sort_values(by="date").reset_index(drop=True)
    split_idx = int(split_ratio * len(df_sorted))
    first = df_sorted.iloc[:split_idx].reset_index(drop=True)
    last = df_sorted.iloc[split_idx:].reset_index(drop=True)
    return first, last


def decide_promotion(mae_v1: float, mae_v2: float, threshold_ratio: float = 0.88) -> str:
    """Decision rule from client: promote only if mae_v2 <= threshold_ratio * mae_v1."""
    if mae_v2 <= threshold_ratio * mae_v1:
        return "promote"
    return "keep"


def retrain(params_path: str = "params.yaml"):
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    data_cfg = params.get("data", {})
    train_cfg = params.get("train", {})
    retrain_cfg = params.get("retrain", {})

    target_col = params["target"]
    feature_cols = params["numeric_features"] + params["categorical_features"]
    cat_cols = params["categorical_features"]

    model_path = Path(train_cfg.get("model_path", "models/model.joblib"))
    clean_train_path = Path(data_cfg.get("clean_train", "data/processed/train_clean.csv"))
    raw_later_path = Path(data_cfg.get("raw_later", "data/raw/later.csv"))

    # 1. Load v1 from models/model.joblib, train_clean, and later.csv
    v1 = load_model(model_path)

    train_clean = pd.read_csv(clean_train_path)
    train_clean = train_clean.dropna(subset=[target_col]).reset_index(drop=True)

    later = pd.read_csv(raw_later_path)

    # 2. Normalise later.csv text columns and split 70/30 by date
    later = normalise_text_columns(later, cat_cols)
    if target_col in later.columns:
        later = later.dropna(subset=[target_col]).reset_index(drop=True)

    split_ratio = retrain_cfg.get("split_ratio", 0.7)
    first70, last30 = split_later_by_date(later, split_ratio=split_ratio)

    # 3. Train v2 = clone(v1) on train_clean + first70
    combined_train = pd.concat([train_clean, first70], ignore_index=True)
    X_train = combined_train[feature_cols]
    y_train = combined_train[target_col]

    v2 = clone(v1)
    v2.fit(X_train, y_train)

    # 4. Compute MAE of v1 and v2 on last30, and percentage improvement
    X_test = last30[feature_cols]
    y_test = last30[target_col]

    preds_v1 = v1.predict(X_test)
    preds_v2 = v2.predict(X_test)

    mae_v1 = float(mean_absolute_error(y_test, preds_v1))
    mae_v2 = float(mean_absolute_error(y_test, preds_v2))
    improvement_pct = float((mae_v1 - mae_v2) / mae_v1 * 100.0)

    # 5. Decision rule
    threshold_ratio = retrain_cfg.get("threshold_ratio", 0.88)
    decision = decide_promotion(mae_v1, mae_v2, threshold_ratio=threshold_ratio)

    print("=" * 50)
    print("RETRAINING EVALUATION SUMMARY")
    print(f"MAE v1 on check: {mae_v1:.4f}")
    print(f"MAE v2 on check: {mae_v2:.4f}")
    print(f"Improvement:     {improvement_pct:.2f}%")
    print(f"Threshold:       mae_v2 <= {threshold_ratio * mae_v1:.4f}")
    print(f"Decision:        {decision}")
    print("=" * 50)

    # 6. Log v2 to MLflow
    tracking_uri = train_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    experiment_name = train_cfg.get("experiment_name", "mp016-resale")
    registered_model_name = retrain_cfg.get("registered_model_name", "mp016-resale-model")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    client = MlflowClient(tracking_uri)

    with mlflow.start_run(run_name="retrain_v2") as v2_run:
        v2_run_id = v2_run.info.run_id
        regressor = v2.named_steps["regressor"]
        mlflow.log_param("model_name", regressor.__class__.__name__)
        for k, v in regressor.get_params().items():
            mlflow.log_param(k, v)
        mlflow.log_metric("mae_v1_on_check", mae_v1)
        mlflow.log_metric("mae_v2_on_check", mae_v2)
        mlflow.log_metric("improvement_pct", improvement_pct)
        mlflow.set_tag("decision", decision)
        mlflow.sklearn.log_model(
            v2,
            name="model",
            serialization_format="cloudpickle",
        )

    if decision == "promote":
        # Register v2 as a new model version
        mv = mlflow.register_model(f"runs:/{v2_run_id}/model", registered_model_name)
        champion_version = str(mv.version)
        client.set_registered_model_alias(registered_model_name, "champion", champion_version)

        # Save v2 as models/model.joblib and keep v1 as models/model_v1.joblib
        v1_archive_path = Path(retrain_cfg.get("v1_archive_path", "models/model_v1.joblib"))
        v1_archive_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(v1, v1_archive_path)
        joblib.dump(v2, model_path)
        print(f"Promoted v2! Saved to {model_path}, archived v1 to {v1_archive_path}")
    else:
        # Keep v1: register v1's run with alias "champion", leave model.joblib unchanged
        v1_info_path = Path(train_cfg.get("info_path", "models/v1_info.json"))
        with open(v1_info_path) as f:
            v1_info = json.load(f)
        v1_run_id = v1_info["run_id"]
        mv = mlflow.register_model(f"runs:/{v1_run_id}/model", registered_model_name)
        champion_version = str(mv.version)
        client.set_registered_model_alias(registered_model_name, "champion", champion_version)
        print(f"Kept v1! Registered v1 run {v1_run_id} as champion version {champion_version}")

    # 7. Write models/comparison.json
    comparison_data = {
        "mae_v1_on_check": mae_v1,
        "mae_v2_on_check": mae_v2,
        "improvement_pct": improvement_pct,
        "decision": decision,
        "registered_model_name": registered_model_name,
        "champion_version": champion_version,
        "n_train": len(train_clean),
        "n_first70": len(first70),
        "n_last30": len(last30),
    }

    comparison_path = Path(retrain_cfg.get("comparison_path", "models/comparison.json"))
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    with open(comparison_path, "w") as f:
        json.dump(comparison_data, f, indent=2)
    print(f"Saved comparison to {comparison_path}")

    # Sync tracking and monitoring in findings.yaml if present
    findings_path = Path(train_cfg.get("findings_path", "findings.yaml"))
    if findings_path.exists():
        try:
            content = findings_path.read_text()
            content = re.sub(
                r'(registered_model_name:\s*)(null|"[^"]*"|\'[^\']*\')',
                f'\\1"{registered_model_name}"',
                content,
            )
            content = re.sub(
                r'(champion_version:\s*)(null|"[^"]*"|\'[^\']*\')',
                f'\\1"{champion_version}"',
                content,
            )
            content = re.sub(
                r'(mae_v1_on_check:\s*)(null|[\d\.]+)',
                rf'\g<1>{round(mae_v1, 4)}',
                content,
            )
            content = re.sub(
                r'(mae_v2_on_check:\s*)(null|[\d\.]+)',
                rf'\g<1>{round(mae_v2, 4)}',
                content,
            )
            content = re.sub(
                r'(decision:\s*)(null|"[^"]*"|\'[^\']*\')',
                f'\\1"{decision}"',
                content,
            )
            findings_path.write_text(content)
            print(f"Updated tracking & monitoring in {findings_path}")
        except Exception as e:
            print(f"Warning: could not update {findings_path}: {e}")

    return comparison_data


def main(params_path: str = "params.yaml"):
    retrain(params_path=params_path)


if __name__ == "__main__":
    main()
