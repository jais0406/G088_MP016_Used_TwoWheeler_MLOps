# MP016 — Used Two-Wheeler Price (Lucknow)

End-to-end MLOps pipeline that estimates a fair resale price for used
two-wheelers brought into the Lucknow second-hand bike market.

| | |
|---|---|
| **Group** | G088 |
| **Problem** | MP016 — Used Two-Wheeler Price (Lucknow) |
| **Course** | SCSE3040 MLOps, Batch 7 |
| **Faculty** | Gaurav Tripathi |
| **Task** | Regression — predict `resale_thousand_rs` (thousand rupees), measured with MAE |

## Team

| Enrolment no. | Name |
|---|---|
| S24CSEU0730 | Harshit Agrawal |
| S24CSEU0707 | Ashwinder Singh |
| S24CSEU0717 | Hrishabh Singh |
| S24CSEU0682 | Jaismeen Kaur |

---

## Contents

1. [Tech stack](#tech-stack)
2. [Project structure](#project-structure)
3. [Setup](#setup) — macOS, Linux, Windows
4. [Running the pipeline](#running-the-pipeline)
5. [Prediction API](#prediction-api)
6. [Docker](#docker)
7. [Tests](#tests)
8. [Results](#results)
9. [Troubleshooting](#troubleshooting)

---

## Tech stack

| Area | Tool |
|---|---|
| Language | Python 3.12 |
| Data handling | pandas |
| Modelling | scikit-learn (Pipeline, ColumnTransformer, Ridge / RandomForest / GradientBoosting, GridSearchCV) |
| Data versioning | DVC |
| Experiment tracking and model registry | MLflow (SQLite backend, `mlflow.db`) |
| Model serialisation | joblib |
| Prediction service | FastAPI + pydantic, served by uvicorn |
| Logging | Python `logging` with a JSON formatter, to stdout |
| Testing | pytest, FastAPI `TestClient` (httpx) |
| Container | Docker (`python:3.12-slim`) |
| Configuration | `params.yaml`, read with `yaml.safe_load` |

All library versions are pinned in `requirements.txt`.

---

## Project structure

```
.
├── README.md                  this file
├── requirements.txt           pinned dependencies
├── params.yaml                paths, feature lists, model grids, retrain settings
├── findings.yaml              the seven data problems, tracking and monitoring results
├── pytest.ini                 test settings (adds src/ and the root to the import path)
├── Dockerfile                 image for the prediction API
├── .dockerignore
├── .dvc/                      DVC configuration
├── data/
│   └── raw/
│       ├── train.csv.dvc      DVC pointer to the original train.csv
│       ├── later.csv.dvc      DVC pointer to later.csv
│       └── holdout_features.csv
├── src/twowheeler/            the ML package
│   ├── data_checks.py         detects the seven kinds of data problem
│   ├── cleaning.py            cleans train.csv using those checks
│   ├── run_cleaning.py        CLI: run the checks and write the cleaned CSV
│   ├── preprocessing.py       impute + scale numbers, one-hot encode categories
│   ├── train.py               v1 training with GridSearchCV, logged to MLflow
│   ├── retrain.py             v2 retraining, v1 vs v2 comparison, promotion rule
│   ├── predict.py             load a model and predict a DataFrame
│   └── predict_holdout.py     predictions for holdout_features.csv
├── app/                       the FastAPI service (see app/README.md)
│   ├── main.py                GET /health, POST /predict
│   ├── schemas.py             request schema with the client's rules
│   └── logging_config.py      JSON log formatter
├── models/
│   ├── model.joblib           champion model served by the API
│   ├── model_v1.joblib        archived v1 model
│   ├── v1_info.json           v1 run id and CV MAE
│   └── comparison.json        v1 vs v2 on the last 30% of later.csv
├── predictions/
│   └── holdout_predictions.csv   record_id,prediction
└── tests/                     pytest suite (see tests/README.md)
```

---

## Setup

You need **Python 3.12** (3.13 also works locally), **Git**, and optionally
**Docker** to build the container. The CSV files are not stored in Git: copy
`train.csv` and `later.csv` from the problem zip into `data/raw/`. Do not open
`train.csv` in Excel and save it, or its hash will stop matching.

### macOS

```bash
# 1. Tools (with Homebrew)
brew install python@3.12 git
# Docker Desktop: https://www.docker.com/products/docker-desktop/

# 2. Get the code
git clone https://github.com/harshitagr42/G088_MP016_Used_TwoWheeler_MLOps.git
cd G088_MP016_Used_TwoWheeler_MLOps

# 3. Virtual environment and dependencies
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install dvc            # only needed to work with the .dvc files

# 4. Data: copy train.csv and later.csv from the zip into data/raw/
cp /path/to/zip/train.csv /path/to/zip/later.csv data/raw/
dvc status                 # should report that data and pipelines are up to date
```

### Linux (Ubuntu / Debian)

```bash
# 1. Tools
sudo apt update
sudo apt install -y python3.12 python3.12-venv git
# Docker Engine: https://docs.docker.com/engine/install/

# 2. Get the code
git clone https://github.com/harshitagr42/G088_MP016_Used_TwoWheeler_MLOps.git
cd G088_MP016_Used_TwoWheeler_MLOps

# 3. Virtual environment and dependencies
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install dvc

# 4. Data
cp /path/to/zip/train.csv /path/to/zip/later.csv data/raw/
dvc status
```

On other distributions, install Python 3.12 and Git with your package manager
(`dnf`, `pacman`, ...); the remaining steps are the same.

### Windows (PowerShell)

```powershell
# 1. Tools
winget install -e --id Python.Python.3.12
winget install -e --id Git.Git
# Docker Desktop: https://www.docker.com/products/docker-desktop/

# 2. Get the code
git clone https://github.com/harshitagr42/G088_MP016_Used_TwoWheeler_MLOps.git
cd G088_MP016_Used_TwoWheeler_MLOps

# 3. Virtual environment and dependencies
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install dvc

# 4. Data
Copy-Item C:\path\to\zip\train.csv, C:\path\to\zip\later.csv data\raw\
dvc status
```

If PowerShell refuses to run `Activate.ps1`, allow local scripts once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
In **Command Prompt** (cmd) activate with `.venv\Scripts\activate.bat` instead.

---

## Running the pipeline

Run every command from the project root with the virtual environment active.
The `src/` folder must be on `PYTHONPATH` so that `twowheeler` can be imported.

| OS / shell | Set `PYTHONPATH` for the session |
|---|---|
| macOS / Linux | `export PYTHONPATH=src` |
| Windows PowerShell | `$env:PYTHONPATH = "src"` |
| Windows cmd | `set PYTHONPATH=src` |

Then run the stages in order:

```bash
# 1. Data checks and cleaning -> data/processed/train_clean.csv
python -m twowheeler.run_cleaning

# 2. Train v1 (3 model families, grid search, every run logged to mlflow.db)
python -m twowheeler.train

# 3. Retrain v2 on train + first 70% of later.csv, compare on the last 30%,
#    apply the promotion rule and set the MLflow @champion alias
python -m twowheeler.retrain

# 4. Predict the holdout file with the champion -> predictions/holdout_predictions.csv
python -m twowheeler.predict_holdout
```

All paths and settings come from `params.yaml`.

To browse the experiments:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

then open <http://127.0.0.1:5000>.

---

## Prediction API

Start the service (any OS, from the project root):

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Interactive docs: <http://localhost:8000/docs>

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | `{"status": "ok", "model_loaded": true, "model_version": "champion"}` |
| `POST` | `/predict` | Returns `{"resale_thousand_rs": <price>, "model_version": "champion"}` |

The request body has exactly these fields: `age_years`, `km_driven`,
`engine_cc`, `owners`, `service_records`, `accidents`, `brand_tier`,
`city_zone`. The client's rules are enforced in the pydantic schema, so a
bad request gets **422** and never reaches the model:

1. `age_years` above 25 is rejected.
2. `engine_cc` below 50 is rejected.

Extra fields (such as the leaking `final_price_from_rc_transfer`) and
unknown categories are also rejected. Every prediction and every rejected
request is logged as one JSON line on stdout.

**Example request — macOS / Linux**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age_years":5,"km_driven":30000,"engine_cc":150,"owners":1,"service_records":6,"accidents":0,"brand_tier":"mid","city_zone":"North"}'
```

**Example request — Windows PowerShell**

```powershell
$body = @{
  age_years = 5; km_driven = 30000; engine_cc = 150; owners = 1
  service_records = 6; accidents = 0; brand_tier = "mid"; city_zone = "North"
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict `
  -ContentType "application/json" -Body $body
```

More detail is in [`app/README.md`](app/README.md).

---

## Docker

The image contains only the service code and the champion model.
The commands are the same on macOS, Linux and Windows (with Docker Desktop running):

```bash
docker build -t mp016-api:1.0 .
docker run --rm -p 8000:8000 mp016-api:1.0
```

Check it:

```bash
curl http://localhost:8000/health
```

To serve a different model file, mount it and point `MODEL_PATH` at it:

```bash
docker run --rm -p 8000:8000 -v "$(pwd)/models:/app/models" \
  -e MODEL_PATH=models/model_v1.joblib -e MODEL_VERSION=v1 mp016-api:1.0
```

(In PowerShell use `${PWD}` instead of `$(pwd)`.)

---

## Tests

```bash
pytest -q
```

The tests need no CSV files: small data is built inside each test. The API
tests use FastAPI's `TestClient` with a dummy model, a tiny model trained in
the test, and the committed champion model (that last group is skipped if
`models/model.joblib` is missing). See [`tests/README.md`](tests/README.md).

---

## Results

**Data problems found in `train.csv`** (details in `findings.yaml`):

| Problem | Present | Details |
|---|---|---|
| Leaking column | yes | `final_price_from_rc_transfer` |
| Unit mix-up | no | |
| Sentinel value | yes | `accidents = -999`, 68 rows |
| Duplicate rows | yes | 36 extra copies |
| Stuck sensor | yes | `engine_cc`, 2026-03-23 to 2026-04-13 |
| Target typos | no | |
| Spelling variants | no | |

Cleaning takes `train.csv` from 1036 to 1000 rows.

**Models**

| Model | Data | MAE |
|---|---|---|
| v1 — Ridge (alpha 0.1) | cleaned `train.csv`, 5-fold CV | 5.17 |
| v1 | last 30% of `later.csv` | 14.44 |
| v2 — retrained | cleaned `train.csv` + first 70% of `later.csv`, checked on last 30% | 12.27 |

v2 is 15% better than v1 on the check rows. The client's rule promotes v2
only if it is at least 12% better, so **v2 was promoted** to `@champion`
and is the model in `models/model.joblib` served by the API.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'twowheeler'` | Set `PYTHONPATH=src` (see the table above) |
| `ModuleNotFoundError: No module named 'yaml'` or `'mlflow'` | The virtual environment is not active, or `pip install -r requirements.txt` was not run |
| `/predict` returns 503 | `models/model.joblib` is missing; run the pipeline or set `MODEL_PATH` |
| Port 8000 already in use | Run on another port: `uvicorn app.main:app --port 8001` or `docker run -p 8001:8000 ...` |
| `dvc status` says `train.csv` changed | The file was edited (for example saved from Excel); copy it again from the zip |
| `Activate.ps1 cannot be loaded` on Windows | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
