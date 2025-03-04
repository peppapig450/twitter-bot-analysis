"""
Twitter Analysis Package

A modular package for fetching, processing, analyzing, and visualizing Twitter data.
Includes functionality for temporal patterns, content analysis (TF-IDF, topics),
sentiment analysis, and visualizations.
"""


from .analysis import analyze_content_dynamic, analyze_sentiment, analyze_temporal_patterns, preprocess_text
from .config import ConfigModel, load_config, setup_logging
from .data import (
    fetch_original_posts,
    fetch_tweets,
    get_bearer_token,
    load_from_json,
    process_tweets,
    save_to_json,
)
from .utils import Paths, RateLimiter, load_nlp_model
from .visualization import plot_content_visualizations, plot_temporal_patterns

__all__ = [
    "ConfigModel",
    "Paths",
    "RateLimiter",
    "analyze_content_dynamic",
    "analyze_sentiment",
    "analyze_temporal_patterns",
    "fetch_original_posts",
    "fetch_tweets",
    "get_bearer_token",
    "load_config",
    "load_from_json",
    "load_nlp_model",
    "plot_content_visualizations",
    "plot_temporal_patterns",
    "preprocess_text",
    "process_tweets",
    "save_to_json",
    "setup_logging",
]

__version__ = "1.0.0"
