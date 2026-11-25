"""Detect the seven kinds of data problem in train.csv. Every function only reads data."""
import numpy as np
import pandas as pd


def leaking_column(train, holdout, target, numeric_cols):
    """Columns missing from holdout that almost equal the target."""
    out = []
    for c in train.columns:
        if c in holdout.columns or c == target or c not in numeric_cols:
            continue
        if abs(train[c] - train[target]).mean() / train[target].mean() < 0.05:
            out.append(c)
    return out


def unit_mixup(train, group_col, cols, min_ratio=5):
    """Return (column, group_value, ratio) where one group's mean is far off the others'."""
    hits = []
    for c in cols:
        means = train[train[c] > 0].groupby(group_col)[c].mean()
        for g, m in means.items():
            rest = means.drop(g).mean()
            r = m / rest
            if r >= min_ratio or r <= 1 / min_ratio:
                hits.append((c, g, r))
    return hits


def sentinel_value(train, cols, candidates):
    hits = []
    for c in cols:
        vc = train[c].value_counts()
        for v in candidates:
            if v in vc.index and vc[v] > 3:
                hits.append((c, v, int(vc[v])))
    return hits


def duplicate_rows(train):
    return int(train.duplicated().sum())


def stuck_sensor(train, cols, min_run_days=7):
    """A column whose single value covers *every* row on a run of consecutive days."""
    hits = []
    dates = pd.to_datetime(train["date"])
    for c in cols:
        for v in train[c].value_counts().head(3).index:
            per_day = (train[c] == v).groupby(dates).mean()
            full = (per_day == 1.0).astype(int)
            run_id = (full.diff().fillna(full) != 0).cumsum()
            for _, run in full.groupby(run_id):
                if run.iloc[0] == 1 and len(run) >= min_run_days:
                    hits.append((c, v, run.index.min().date().isoformat(),
                                 run.index.max().date().isoformat()))
    return hits


def target_typos(train, target, factor=10):
    """Rows whose target is ~factor x the matching leaked/related column or an outlier gap."""
    t = train[target]
    s = np.sort(t.values)
    gaps = np.diff(s)
    return int((t > s[gaps.argmax()]).sum()) if gaps.max() > 0.5 * s.max() else 0


def spelling_variants(train, text_cols):
    hits = {}
    for c in text_cols:
        raw = train[c].nunique()
        norm = train[c].astype(str).str.strip().str.lower().str.replace(r"[^a-z]", "", regex=True).nunique()
        if raw != norm or raw > 6:
            hits[c] = norm
    return hits


def run_all(train, holdout, params):
    num, cat = params["numeric_features"], params["categorical_features"]
    return {
        "leaking_column": leaking_column(train, holdout, params["target"],
                                         num + list(params["leak_columns"])),
        "unit_mixup": unit_mixup(train, params["group_column"], num),
        "sentinel_value": sentinel_value(train, num, params["checks"]["sentinel_candidates"]),
        "duplicate_rows": duplicate_rows(train),
        "stuck_sensor": stuck_sensor(train, num, params["checks"]["stuck_min_run_days"]),
        "target_typos": target_typos(train, params["target"], params["checks"]["typo_factor"]),
        "spelling_variants": spelling_variants(train, cat),
    }
