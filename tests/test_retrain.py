"""Tests for twowheeler.retrain."""
import pandas as pd
import pytest

from twowheeler.retrain import decide_promotion, normalise_text_columns, split_later_by_date


def test_split_later_by_date_synthetic():
    """Verify 70/30 chronological date split on synthetic data."""
    dates = [
        "2026-07-10",
        "2026-07-01",
        "2026-07-05",
        "2026-07-02",
        "2026-07-08",
        "2026-07-03",
        "2026-07-09",
        "2026-07-04",
        "2026-07-06",
        "2026-07-07",
    ]
    df = pd.DataFrame({"record_id": [f"R{i}" for i in range(10)], "date": dates})

    first70, last30 = split_later_by_date(df, split_ratio=0.7)

    assert len(first70) == 7
    assert len(last30) == 3

    # Ensure chronologically sorted
    assert list(first70["date"]) == sorted(dates)[:7]
    assert list(last30["date"]) == sorted(dates)[7:]
    assert first70["date"].max() <= last30["date"].min()


def test_decide_promotion_at_12_pct_better():
    """Promote if mae_v2 is at least 12% better (mae_v2 <= 0.88 * mae_v1)."""
    mae_v1 = 100.0
    mae_v2 = 88.0  # exactly 12% improvement
    assert decide_promotion(mae_v1, mae_v2, threshold_ratio=0.88) == "promote"

    mae_v2_better = 85.0  # 15% improvement
    assert decide_promotion(mae_v1, mae_v2_better, threshold_ratio=0.88) == "promote"


def test_decide_promotion_at_11_pct_better():
    """Keep if mae_v2 is only 11% better (mae_v2 > 0.88 * mae_v1)."""
    mae_v1 = 100.0
    mae_v2 = 89.0  # 11% improvement (< 12%)
    assert decide_promotion(mae_v1, mae_v2, threshold_ratio=0.88) == "keep"

    mae_v2_same = 100.0  # 0% improvement
    assert decide_promotion(mae_v1, mae_v2_same, threshold_ratio=0.88) == "keep"

    mae_v2_worse = 110.0  # worse MAE
    assert decide_promotion(mae_v1, mae_v2_worse, threshold_ratio=0.88) == "keep"


def test_normalise_text_columns_synthetic():
    """Verify text column stripping, lowercasing, and city_zone capitalization."""
    raw = pd.DataFrame(
        {
            "brand_tier": [" Mid ", "PREMIUM", "budget  "],
            "city_zone": ["north", " SOUTH ", "cEntraL"],
            "numeric_col": [1, 2, 3],
        }
    )
    cleaned = normalise_text_columns(raw, ["brand_tier", "city_zone"])

    assert list(cleaned["brand_tier"]) == ["mid", "premium", "budget"]
    assert list(cleaned["city_zone"]) == ["North", "South", "Central"]
    assert list(cleaned["numeric_col"]) == [1, 2, 3]
