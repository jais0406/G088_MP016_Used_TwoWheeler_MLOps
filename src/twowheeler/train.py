"""Model training script for MP016 Used Two-Wheeler Price Prediction.

Compares candidate regression models (Ridge, RandomForestRegressor, GradientBoostingRegressor)
using 5-fold CV with negative MAE scoring and GridSearchCV, logs every candidate to MLflow,
refits the champion model on all of train_clean, saves the pipeline to models/model.joblib,
and writes run metadata to models/v1_info.json.

Usage:
    PYTHONPATH=src python -m twowheeler.train
"""
import inspect
import json
from pathlib import Path
import yaml
import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn

from .preprocessing import build_preprocessor

MODEL_REGISTRY = {
    "Ridge": Ridge,
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
}


def parse_param_grid(raw_grid):
    """Ensure parameters are properly prefixed with regressor__ for scikit-learn Pipeline."""
    grid = {}
    if isinstance(raw_grid, dict):
        params_dict = (
            raw_grid.get("params", raw_grid)
            if "params" in raw_grid and isinstance(raw_grid["params"], dict)
            else raw_grid
        )
        for k, v in params_dict.items():
            if k in ("params", "type", "name"):
                continue
            prefixed = k if k.startswith("regressor__") else f"regressor__{k}"
            grid[prefixed] = v if isinstance(v, list) else [v]
    return grid


def get_regressor_instance(model_name: str, random_state: int = 42):
    """Instantiate a regressor by name, passing random_state if supported."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model name: {model_name}. Allowed: {list(MODEL_REGISTRY.keys())}"
        )
    cls = MODEL_REGISTRY[model_name]
    sig = inspect.signature(cls)
    kwargs = {}
    if "random_state" in sig.parameters:
        kwargs["random_state"] = random_state
    return cls(**kwargs)


def train(params_path: str = "params.yaml"):
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    train_cfg = params.get("train", {})
    clean_data_path = Path(params["data"]["clean_train"])

    # 1. Read clean training data and drop rows where the target is NaN
    df = pd.read_csv(clean_data_path)
    target_col = params["target"]
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    feature_cols = params["numeric_features"] + params["categorical_features"]
    X = df[feature_cols]
    y = df[target_col]

    # Configure MLflow
    tracking_uri = train_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    experiment_name = train_cfg.get("experiment_name", "mp016-resale")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    cv_folds = train_cfg.get("cv_folds", 5)
    random_state = train_cfg.get("random_state", 42)
    scoring = train_cfg.get("scoring", "neg_mean_absolute_error")

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    model_configs = train_cfg.get("models", {})

    candidate_records = []

    print(f"Loaded {len(df)} rows. Starting GridSearchCV for models: {list(model_configs.keys())}")

    # 2 & 3. Compare models using GridSearchCV
    for model_name, model_raw_cfg in model_configs.items():
        base_regressor = get_regressor_instance(model_name, random_state=random_state)
        preprocessor = build_preprocessor(params)
        base_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", base_regressor),
        ])

        param_grid = parse_param_grid(model_raw_cfg)

        grid_search = GridSearchCV(
            estimator=base_pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            refit=True,
        )
        grid_search.fit(X, y)

        # 4. Log every candidate to MLflow
        cv_results = grid_search.cv_results_
        n_candidates = len(cv_results["params"])
        for i in range(n_candidates):
            candidate_params = cv_results["params"][i]
            mean_score = cv_results["mean_test_score"][i]
            cv_mae = float(-mean_score)

            if i == grid_search.best_index_:
                fitted_candidate_pipeline = grid_search.best_estimator_
            else:
                fitted_candidate_pipeline = clone(base_pipeline).set_params(**candidate_params)
                fitted_candidate_pipeline.fit(X, y)

            with mlflow.start_run(run_name=f"{model_name}_candidate_{i}") as candidate_run:
                mlflow.log_param("model_name", model_name)
                for p_k, p_v in candidate_params.items():
                    mlflow.log_param(p_k, p_v)
                    clean_k = p_k.replace("regressor__", "")
                    if clean_k != p_k:
                        mlflow.log_param(clean_k, p_v)
                mlflow.log_metric("cv_mae", cv_mae)
                mlflow.sklearn.log_model(
                    fitted_candidate_pipeline,
                    name="model",
                    serialization_format="cloudpickle",
                )

            candidate_records.append({
                "model_name": model_name,
                "params": candidate_params,
                "cv_mae": cv_mae,
                "base_pipeline": base_pipeline,
                "run_id": candidate_run.info.run_id,
            })
            print(f"Candidate {model_name} #{i}: params={candidate_params}, CV MAE={cv_mae:.4f}")

    if not candidate_records:
        raise RuntimeError("No candidate models were evaluated.")

    # 5. Refit the best model on all of train_clean, log it as the final run
    best_candidate = min(candidate_records, key=lambda c: c["cv_mae"])
    best_model_name = best_candidate["model_name"]
    best_params = best_candidate["params"]
    best_cv_mae = best_candidate["cv_mae"]

    final_pipeline = clone(best_candidate["base_pipeline"]).set_params(**best_params)
    final_pipeline.fit(X, y)

    with mlflow.start_run(run_name="final_model") as final_run:
        final_run_id = final_run.info.run_id
        mlflow.log_param("model_name", best_model_name)
        mlflow.log_param("final_model", True)
        for p_k, p_v in best_params.items():
            mlflow.log_param(p_k, p_v)
            clean_k = p_k.replace("regressor__", "")
            if clean_k != p_k:
                mlflow.log_param(clean_k, p_v)
        mlflow.log_metric("cv_mae", best_cv_mae)
        mlflow.sklearn.log_model(
            final_pipeline,
            name="model",
            serialization_format="cloudpickle",
        )

    # Print run_id and CV MAE
    print("\n" + "=" * 50)
    print("FINAL CHAMPION MODEL SELECTION")
    print(f"Run ID: {final_run_id}")
    print(f"Model Name: {best_model_name}")
    print(f"CV MAE: {best_cv_mae:.4f}")
    print("=" * 50 + "\n")

    # Save to models/model.joblib
    model_path = Path(train_cfg.get("model_path", "models/model.joblib"))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, model_path)
    print(f"Saved model pipeline to: {model_path}")

    # 6. Write run_id, model name and CV MAE to models/v1_info.json
    info_path = Path(train_cfg.get("info_path", "models/v1_info.json"))
    info_path.parent.mkdir(parents=True, exist_ok=True)
    v1_info = {
        "run_id": final_run_id,
        "model_name": best_model_name,
        "cv_mae": float(best_cv_mae),
    }
    with open(info_path, "w") as f:
        json.dump(v1_info, f, indent=2)
    print(f"Saved run metadata to: {info_path}")

    # Optionally sync tracking info to findings.yaml if present
    findings_path = Path(train_cfg.get("findings_path", "findings.yaml"))
    if findings_path.exists():
        try:
            content = findings_path.read_text()
            import re
            content = re.sub(
                r'(experiment_name:\s*)(null|"[^"]*"|\'[^\']*\')',
                f'\\1"{experiment_name}"',
                content,
            )
            content = re.sub(
                r'(best_run_id:\s*)(null|"[^"]*"|\'[^\']*\')',
                f'\\1"{final_run_id}"',
                content,
            )
            findings_path.write_text(content)
            print(f"Updated tracking in {findings_path}")
        except Exception as e:
            print(f"Warning: could not update {findings_path}: {e}")

    return final_run_id, best_model_name, best_cv_mae, final_pipeline


def main(params_path: str = "params.yaml"):
    train(params_path=params_path)


if __name__ == "__main__":
    main()
