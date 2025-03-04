from typing import TYPE_CHECKING

import pandas as pd

from twitter_analysis.config.settings import RateLimitEndpoint, RateLimits

if TYPE_CHECKING:
    from pydantic import ValidationInfo

def check_rate_limits(value: RateLimits, info: "ValidationInfo") -> RateLimits:
    """Ensure required endpoints are present in rate_limits."""
    required_endpoints: set[RateLimitEndpoint] = {"users_tweets", "tweets"}
    if not required_endpoints.issubset(value.keys()):
        error_message = f"rate_limits must include {required_endpoints}"
        raise ValueError(error_message)
    return value

def check_series_dtype(v: pd.Series) -> pd.Series:
    """Ensure the Series has int64 dtype."""
    if not pd.api.types.is_integer_dtype(v) or v.dtype != "int64":
        error_message = "Series must be of dtype int64"
        raise ValueError(error_message)
    return v
