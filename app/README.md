# app/ — Prediction API

FastAPI service that predicts `resale_thousand_rs` for a used two-wheeler.

| File | Purpose |
|---|---|
| `main.py` | `GET /health`, `POST /predict`; loads `models/model.joblib` at startup |
| `schemas.py` | Request schema with the client's rules (`age_years <= 25`, `engine_cc >= 50`) |
| `logging_config.py` | JSON logs to stdout, one line per prediction |

**Run**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Example**
```bash
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"age_years":5,"km_driven":30000,"engine_cc":150,"owners":1,
       "service_records":6,"accidents":0,"brand_tier":"mid","city_zone":"North"}'
```

Invalid input returns **422**; no model loaded returns **503**.
Set `MODEL_PATH` to load a model from a different path.
