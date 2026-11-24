# tests/ — API tests

Pytest tests for the prediction API. They need **no CSVs and no real model**:
a dummy model or a tiny model trained inside the test is used.

| File | Covers |
|---|---|
| `conftest.py` | Shared fixtures: dummy model, `TestClient`, a valid request |
| `test_api.py` | `/health`, `/predict`, client rules (422), missing/extra fields, 503, JSON logging |
| `test_model_loading.py` | Saving and loading a real `joblib` model in the service |

**Run**
```bash
pytest -q
```
