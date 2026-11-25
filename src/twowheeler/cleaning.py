"""Cleaning steps for train.csv, driven by the data checks."""
import numpy as np
import pandas as pd

from . import data_checks as dc


def clean_train(train, holdout, params):
    df = train.copy()
    num, cat = params["numeric_features"], params["categorical_features"]

    df = df.drop_duplicates().reset_index(drop=True)                      # duplicate_rows
    df = df.drop(columns=[c for c in params["leak_columns"] if c in df])  # leaking_column

    for c in cat:                                                         # spelling_variants
        df[c] = df[c].astype(str).str.strip().str.lower()
    df["city_zone"] = df["city_zone"].str.capitalize()

    for c, v, _ in dc.sentinel_value(df, num, params["checks"]["sentinel_candidates"]):
        df.loc[df[c] == v, c] = np.nan                                    # sentinel_value

    dates = pd.to_datetime(df["date"])
    for c, _, start, end in dc.stuck_sensor(df, num, params["checks"]["stuck_min_run_days"]):
        df.loc[(dates >= start) & (dates <= end), c] = np.nan             # stuck_sensor

    for c, g, ratio in dc.unit_mixup(df, params["group_column"], num):    # unit_mixup
        mask = df[params["group_column"]] == g
        df.loc[mask, c] = df.loc[mask, c] / round(ratio)

    # target_typos: none found in this train.csv, so nothing to fix.
    return df
