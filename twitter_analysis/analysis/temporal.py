import logging

import pandas as pd

from twitter_analysis.models.analysis_result import TemporalAnalysisResult


def analyze_temporal_patterns(df: pd.DataFrame, burst_threshold: float = 5.0) -> TemporalAnalysisResult:
    """
    Analyze temporal patterns in tweet data with optimized Pandas operations.

    Args:
        df: DataFrame with a 'created_at' column of datetime64[ns] type.
        burst_threshold: Time gap in minutes to separate bursts (default: 5.0).

    Returns:
        TemporalAnalysisResult: Object with temporal statistics.

    Raises:
        ValueError: If 'created_at' column is missing or not datetime type.

    """
    # Handle empty DataFrame or missing column
    if df.empty or "created_at" not in df.columns:
        logging.debug("DataFrame is empty or missing 'created_at' column.")
        return TemporalAnalysisResult()

    if not pd.api.types.is_datetime64_any_dtype(df["created_at"]):
        error_message = "Column 'created_at' must be a datetime64 type."
        raise ValueError(error_message)

    try:
        # Pre-compute datetime series for reuse
        times = df["created_at"].dropna()

        # Hourly and date counts
        hour_counts = times.dt.hour.value_counts().reindex(range(24), fill_value=0).astype("Int64")
        day_counts = times.dt.dayofweek.value_counts().reindex(range(7), fill_value=0).astype("Int64")

        if times.empty:
            logging.debug("No valid datetime values in 'created_at' after dropping NaNs.")
            return TemporalAnalysisResult(hour_counts=hour_counts, day_counts=day_counts)

        # Time difference (in minutes) without redundant sorting if not needed
        time_diff = times.sort_values().diff().dt.total_seconds().div(60).dropna()

        # Compute statistics efficiently in one pass
        stats = time_diff.agg(["mean", "std"]).astype(float)
        avg_time = stats["mean"] if pd.notna(stats["mean"]) else 0.0
        std_time = stats["std"] if pd.notna(stats["std"]) else 0.0
        cv_time = std_time / avg_time if avg_time > 0 and std_time > 0 else 0.0

        # Burst analysis: bursts are sequences separated by large gaps
        burst_boundaries = time_diff > burst_threshold
        num_bursts = burst_boundaries.sum() + 1 if not time_diff.empty else 0
        avg_burst_size = len(times) / num_bursts if num_bursts > 0 else 0.0

        return TemporalAnalysisResult(
            hour_counts=hour_counts,
            day_counts=day_counts,
            avg_time_between=avg_time,
            std_time_between=std_time,
            cv_time_between=cv_time,
            num_bursts=num_bursts,
            avg_burst_size=avg_burst_size,
        )

    except Exception:
        logging.exception("Error during temporal analysis.")
        return TemporalAnalysisResult()
