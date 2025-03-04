import logging

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sentiment_analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(df: pd.DataFrame) -> dict[str, float]:
    """
    Analyze sentiment of text in a DataFrame using VADER sentiment analysis.

    Args:
        df: DataFrame with a "text" column containing string data.

    Returns:
        dict[str, float]: Average sentiment scores (compound, positive, negative, neutral).

    Notes:
        Optimized for social media text. Returns zeros if no valid text is provided.

    """
    if df.empty or "text" not in df.columns:
        logging.warning("DataFrame is empty or missing 'text' column for sentiment analysis")
        return {"compound": 0.0, "positive": 0.0, "negative": 0.0, "neutral": 0.0}

    if not pd.api.types.is_string_dtype(df["text"]):
        logging.warning("Converting 'text' column to string type for sentiment analysis")
        df["text"] = df["text"].astype(str)

    texts = df["text"].dropna().str.strip().tolist()
    if not texts:
        return {"compound": 0.0, "positive": 0.0, "negative": 0.0, "neutral": 0.0}

    try:
        sentiment_scores = [sentiment_analyzer.polarity_scores(text) for text in texts if text.strip()]
        if not sentiment_scores:
            return {"compound": 0.0, "positive": 0.0, "negative": 0.0, "neutral": 0.0}

        return {
            "compound": float(np.mean([s["compound"] for s in sentiment_scores])),
            "positive": float(np.mean([s["pos"] for s in sentiment_scores])),
            "negative": float(np.mean([s["neg"] for s in sentiment_scores])),
            "neutral": float(np.mean([s["neu"] for s in sentiment_scores])),
        }
    except Exception:
        logging.exception("Sentiment analysis failed")
        return {"compound": 0.0, "positive": 0.0, "negative": 0.0, "neutral": 0.0}
