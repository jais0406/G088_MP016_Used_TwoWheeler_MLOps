"""FastAPI service for MP016: predicts the resale price of a used two-wheeler.

Run locally:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.logging_config import get_logger
from app.schemas import FEATURES, BikeFeatures, PredictionResponse

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "champion")

logger = get_logger()


def load_model(path: Path = MODEL_PATH):
    """Load the champion model saved by the training code, or None if missing."""
    if not path.exists():
        logger.warning("model file not found", extra={"fields": {"model_path": str(path)}})
        return None
    model = joblib.load(path)
    logger.info("model loaded", extra={"fields": {"model_path": str(path)}})
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Tests may put a model on app.state before startup; keep it if so.
    if getattr(app.state, "model", None) is None:
        app.state.model = load_model()
    yield


app = FastAPI(title="MP016 Used Two-Wheeler Price API", version="1.0", lifespan=lifespan)
app.state.model = None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": app.state.model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(bike: BikeFeatures):
    model = app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    start = time.perf_counter()
    row = pd.DataFrame([bike.model_dump()], columns=FEATURES)
    prediction = float(model.predict(row)[0])
    latency_ms = (time.perf_counter() - start) * 1000

    # Only bike features are logged - no personal details of the customer.
    logger.info(
        "prediction",
        extra={
            "fields": {
                "event": "prediction",
                "features": bike.model_dump(),
                "prediction": round(prediction, 3),
                "model_version": MODEL_VERSION,
                "latency_ms": round(latency_ms, 2),
            }
        },
    )
    return PredictionResponse(resale_thousand_rs=prediction, model_version=MODEL_VERSION)
