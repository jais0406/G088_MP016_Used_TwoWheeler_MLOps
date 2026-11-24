"""Request and response schemas for the prediction API.

The client's two rules live here, inside the pydantic Field definitions,
so a bad request is rejected with 422 before it ever reaches the model:
  1. age_years above 25 is impossible  -> le=25
  2. engine_cc below 50 is a typo       -> ge=50
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FEATURES = [
    "age_years",
    "km_driven",
    "engine_cc",
    "owners",
    "service_records",
    "accidents",
    "brand_tier",
    "city_zone",
]


class BikeFeatures(BaseModel):
    # Exactly these fields: unknown fields (e.g. the leaking
    # final_price_from_rc_transfer) are rejected with 422.
    model_config = ConfigDict(extra="forbid")

    age_years: float = Field(..., ge=0, le=25, description="Age of the bike in years")
    km_driven: float = Field(..., ge=0, description="Odometer reading in km")
    engine_cc: float = Field(..., ge=50, description="Engine size in cc")
    owners: int = Field(..., ge=0, description="Number of previous owners")
    service_records: int = Field(..., ge=0, description="Service visits on record")
    accidents: int = Field(..., ge=0, description="Reported accidents")
    brand_tier: Literal["budget", "mid", "premium"]
    city_zone: Literal["North", "South", "Central"]


class PredictionResponse(BaseModel):
    resale_thousand_rs: float
    model_version: str
