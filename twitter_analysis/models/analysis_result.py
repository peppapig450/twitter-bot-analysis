import pandas as pd
from pydantic import BaseModel, Field, field_validator


class TemporalAnalysisResult(BaseModel):
    """Model for temporal analysis results."""

    hour_counts: pd.Series[int] = Field(default_factory=lambda: pd.Series([], dtype="int64"))
    day_counts: pd.Series[int] = Field(default_factory=lambda: pd.Series([], dtype="int64"))
    avg_time_between: float = 0.0
    std_time_between: float = 0.0
    cv_time_between: float = 0.0
    num_bursts: int = 0
    avg_burst_size: float = 0.0

    @field_validator("hour_counts", "day_counts")
    @classmethod
    def check_series_dtype(cls, v: pd.Series) -> pd.Series:
        """Ensure the Series has int64 dtype."""
        if not pd.api.types.is_integer_dtype(v) or v.dtype != "int64":
            error_message = "Series must be of dtype int64"
            raise ValueError(error_message)
        return v

    model_config = {
        "arbitrary_types_allowed": True,
    }


class ContentAnalysisResult(BaseModel):
    """Model for content analysis results."""

    top_terms: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    type_token_ratio: float = 0.0
    avg_sentence_length: float = 0.0
    text_similarity: float = 0.0
    sentiment_stats: dict[str, float] = Field(
        default_factory=lambda: {"compound": 0.0, "positive": 0.0, "negative": 0.0, "neutral": 0.0}
    )

    model_config = {
        "arbitrary_types_allowed": True,
    }
