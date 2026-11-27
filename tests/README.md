# tests/ — Test suite

Pytest tests for the whole project. None of them need the CSV files (those
are not in git): small data is built inside each test.

| File | Covers |
|---|---|
| `conftest.py` | Shared fixtures: dummy model, `TestClient`, a valid request |
| `test_api.py` | `/health`, `/predict`, the client's rules (422), missing/extra fields, 500 and 503, JSON logging of predictions and rejected requests |
| `test_model_loading.py` | Saving a small `joblib` pipeline and serving it through the API |
| `test_champion_model.py` | The committed `models/model.joblib` works through the API (skipped if the file is absent) |
| `test_predict.py` | `src/twowheeler/predict.py` inference helper |
| `test_train.py` | v1 training code |
| `test_retrain.py` | v2 retraining and the promotion rule |
| `test_predict_holdout.py` | Writing `predictions/holdout_predictions.csv` |

## Run

From the project root, with the dependencies from `requirements.txt` installed:

```bash
pytest -q                 # everything
pytest -q tests/test_api.py   # only the API tests
```

`pytest.ini` puts `src/` and the project root on the import path, so
`import twowheeler` and `import app` both work.
