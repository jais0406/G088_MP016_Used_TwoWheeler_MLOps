import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Make the project root importable so `import app` works when running pytest.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402


class DummyModel:
    """Stands in for the real model so tests need no CSV or model file."""

    def __init__(self):
        self.last_input = None

    def predict(self, X):
        self.last_input = X
        return [42.5] * len(X)


@pytest.fixture
def dummy_model():
    return DummyModel()


@pytest.fixture
def client(dummy_model):
    app.state.model = dummy_model
    with TestClient(app) as c:
        yield c
    app.state.model = None


@pytest.fixture
def valid_bike():
    return {
        "age_years": 5.0,
        "km_driven": 30000,
        "engine_cc": 150,
        "owners": 1,
        "service_records": 6,
        "accidents": 0,
        "brand_tier": "mid",
        "city_zone": "North",
    }
