"""python -m twowheeler.run_cleaning  (run from repo root with PYTHONPATH=src)"""
import json
import pathlib

import pandas as pd
import yaml

from . import data_checks, cleaning


def main(params_path="params.yaml"):
    p = yaml.safe_load(open(params_path))
    train = pd.read_csv(p["data"]["raw_train"])
    holdout = pd.read_csv(p["data"]["raw_holdout"])
    print("findings:", json.dumps(data_checks.run_all(train, holdout, p), indent=2, default=str))
    clean = cleaning.clean_train(train, holdout, p)
    out = pathlib.Path(p["data"]["clean_train"])
    out.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out, index=False)
    print(f"cleaned {train.shape} -> {clean.shape}, NaNs left:\n{clean.isna().sum()[clean.isna().sum() > 0]}")


if __name__ == "__main__":
    main()
