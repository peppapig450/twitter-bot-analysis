from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Required,
    TypedDict,
    cast,
)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import spacy
import spacy.cli
import tweepy
import tweepy.errors
import zstandard as zstd
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator
from tabulate import tabulate
from tweepy.asynchronous import AsyncClient, AsyncPaginator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

from twitter_analysis.analysis.content import analyze_content_dynamic

if TYPE_CHECKING:
    from spacy.language import Language

try:
    from nltk.tokenize import word_tokenize

    NLTK_AVAILABLE: bool = True
except ImportError:
    NLTK_AVAILABLE: bool = False

# TypeAliases for clarity
type TweetList = list[tweepy.Tweet]
type TweetDict = dict[int, tweepy.Tweet | None]
type UserMap = dict[int, str]
type AnalysisResult = dict[str, Any]
type RateLimitEndpoint = Literal["users_tweets", "tweets"]

sentiment_analyzer = SentimentIntensityAnalyzer()


def get_bearer_token() -> str:
    """Retrieve bearer token securely from environment or .env file."""

    def _get_token_from_env() -> str | None:
        """Helper function to get token from environment."""
        return os.getenv("X_BEARER_TOKEN")

    # Try environment variables first
    if token := _get_token_from_env():
        return token

    # Fall back to .env file in project root
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path)
        if token := _get_token_from_env():
            return token
    error_message = "Bearer token not found. Set X_BEARER_TOKEN in environment or create a .env file with X_BEARER_TOKEN=your_token_here. See README for setup."
    raise ValueError(error_message)


# Define the structure of rate_limits
class RateLimits(TypedDict):
    users_tweets: Required[int]
    tweets: Required[int]


# Define the structure of tweet data using TypedDict for type safety
class TweetData(TypedDict):
    id: int
    text: str
    created_at: pd.Timestamp | None
    mentions: list[str]
    hashtags: list[str]
    referenced_tweets: list[dict[str, str | int]]


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


# Config model definition
class ConfigModel(BaseModel):
    """Configuration model for the Twitter analysis script."""

    max_tweets: int = Field(default=500, gt=0, description="Maximum tweets to fetch")
    rate_limits: RateLimits = Field(
        default=cast(RateLimits, {"users_tweets": 1500, "tweets": 900}),
        description="Rate limits per API Endpoint.",
    )
    reset_interval: int = Field(default=900, gt=0, description="Rate limit reset interval in seconds")
    nlp_model: str = Field(default="en_core_web_sm", description="NLP model name")

    @field_validator("rate_limits")
    @classmethod
    def check_rate_limits(cls, value: RateLimits) -> RateLimits:
        """Ensure required endpoints are present in rate_limits."""
        required_endpoints: set[RateLimitEndpoint] = {"users_tweets", "tweets"}
        if not required_endpoints.issubset(value.keys()):
            error_message = f"rate_limits must included {required_endpoints}"
            raise ValueError(error_message)
        return value

    # Enable model configuration for strictness
    model_config = {
        "str_strip_whitespace": True,  # Automatically strip whitespace from strings
        "frozen": True,  # Make the model immutable after creation
    }


def load_config(config_path: Path | None = None) -> ConfigModel:
    """Load configuration with validation"""
    if config_path and config_path.exists():
        try:
            return ConfigModel.model_validate_json(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValidationError) as e:
            error_message = f"Invalid config: {e}. Using defaults."
            logging.warning(error_message)
    # Return default config if no file exists or validation files
    return ConfigModel()


# Setup logging
def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging with a specified level."""
    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    logging.basicConfig(
        level=levels.get(log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@dataclass
class Paths:
    account: str
    _PATH_TEMPLATES: ClassVar[dict[str, str]] = {
        "DATA_FILE": "{account}_tweets.json",
        "OUTPUT_CSV": "{account}_analysis.csv",
        "TEMPORAL_PLOT": "{account}_temporal_patterns.png",
        "NETWORK_PLOT": "{account}_network_graph.png",
        "WORDCLOUD_PLOT": "{account}_wordcloud.png",
    }
    DATA_FILE: Path = field(default_factory=lambda: Path(), init=False)
    OUTPUT_CSV: Path = field(default_factory=lambda: Path(), init=False)
    TEMPORAL_PLOT: Path = field(default_factory=lambda: Path(), init=False)
    NETWORK_PLOT: Path = field(default_factory=lambda: Path(), init=False)
    WORDCLOUD_PLOT: Path = field(default_factory=lambda: Path(), init=False)

    def __post_init__(self) -> None:
        """Initialize paths dynamically based on the account name."""
        self.__dict__.update(
            {
                attr: Path(template.format(account=self.account))
                for attr, template in self._PATH_TEMPLATES.items()
            }
        )


@dataclass
class RateLimiter:
    """A rate limiter for tracking and enforcing API request limits per endpoint."""

    max_requests: RateLimits  # Maximum requests per endpoint
    reset_interval: int = 900  # Reset interval in seconds (default: 15 minutes)
    requests: dict[str, int] = field(default_factory=dict)  # Current request counts
    reset_times: dict[str, float] = field(default_factory=dict)  # Reset timestamps

    def increment(self, endpoint: str) -> None:
        """
        Increment the request count for the specified endpoint, resetting if needed.

        Args:
            endpoint: The API endpoint identifier (e.g., 'users_tweets').

        """
        current_time = asyncio.get_event_loop().time()
        if endpoint not in self.requests:
            self.requests[endpoint] = 0
            self.reset_times[endpoint] = current_time + self.reset_interval
        elif current_time >= self.reset_times[endpoint]:
            logging.debug("Resetting rate limit for %s at %.2f", endpoint, current_time)
            self.requests[endpoint] = 0
            self.reset_times[endpoint] = current_time + self.reset_interval
        self.requests[endpoint] += 1

    def is_limited(self, endpoint: str) -> bool:
        """
        Check if the rate limit has been reached for the specified endpoint.

        Args:
            endpoint: The API endpoint identifier.

        Returns:
            bool: True if the limit is reached, False otherwise.

        """
        current_time = asyncio.get_event_loop().time()
        if endpoint in self.reset_times and current_time >= self.reset_times[endpoint]:
            logging.debug("Rate limit expired for %s resetting.", endpoint)
            self.requests[endpoint] = 0
            del self.reset_times[endpoint]

        limit = self.max_requests.get(endpoint, 0)
        if not isinstance(limit, int):
            limit = 0
        return self.requests.get(endpoint, 0) >= limit

    def get_remaining_time(self, endpoint: str) -> float:
        """
        Calculate the remaining time until the rate limit resets.

        Args:
            endpoint: The API endpoint identifier.

        Returns:
            float: Seconds remaining until reset, or 0 if not limited or reset time passed.

        """
        current_time = asyncio.get_event_loop().time()
        reset_time = self.reset_times.get(endpoint, current_time)
        return max(0, reset_time - current_time)


def load_nlp_model(model_name: str) -> Language | None:
    """
    Load a spaCy language model with robust error handling.

    Args:
        model_name: Name of the spaCy model to load (e.g., 'en_core_web_sm').

    Returns:
        Loaded spaCy Language object if successful, None otherwise.

    Notes:
        Assumes model_name is a valid string (e.g., from ConfigModel).
        Logs warnings or errors if the model is not installed or fails to load.

    """
    if not spacy.util.is_package(model_name):
        logging.warning("spaCy model %s is not installed", model_name)
        return None

    try:
        return spacy.load(model_name)
    except OSError:
        logging.exception("Failed to load spaCy model '%s'", model_name)
        return None
    except ValueError:
        logging.warning("Invalid spaCy model name '%s'", model_name)
        return None


async def fetch_tweets(
    client: AsyncClient, user_id: int, rate_limiter: RateLimiter, max_tweets: int = 500
) -> TweetList:
    """Fetch recent tweets asynchronously with proper error handling and backoff."""
    tweets: TweetList = []
    tweet_fields = ["text", "created_at", "entities", "in_reply_to_user_id", "referenced_tweets"]
    max_tweets = min(max_tweets, 500)
    endpoint = "users_tweets"

    # Define an async generator for paginated tweets
    async def paginated_tweets() -> AsyncGenerator[TweetList, None]:
        paginator = AsyncPaginator(
            client.get_users_tweets,
            id=user_id,
            tweet_fields=tweet_fields,
            max_results=100,
            limit=5,  # Maximum of 5 pages, up to 500 tweets
        )
        async for response in paginator:
            # Check rate limit before processing each page
            if rate_limiter.is_limited(endpoint):
                wait_time = rate_limiter.get_remaining_time(endpoint)
                if wait_time > 0:
                    logging.info("Rate limit reached for %s. Waiting %f.2f seconds...", endpoint, wait_time)
                    await asyncio.sleep(wait_time)
                continue
            rate_limiter.increment(endpoint)

            # Handle different response types
            if isinstance(response, tweepy.Response):
                yield response.data or []
            elif isinstance(response, dict):
                yield response.get("data", [])
            else:
                error_message = f"Unknown response type: {type(response)}"
                raise TypeError(error_message)
            await asyncio.sleep(1)  # Small delay to respect API rate limits

    try:
        # Iterate over paginated tweets and collect them
        async for tweet_batch in paginated_tweets():
            tweets.extend(tweet_batch)
            if len(tweets) >= max_tweets:
                break
    except tweepy.errors.TweepyException as e:
        match e:
            case tweepy.errors.TooManyRequests():
                logging.warning("Rate limit hit during pagination. Consider retrying later.")
            case _:
                logging.exception("Error fetching tweets")
    except Exception:
        logging.exception("Unexpected error fetching tweets")

    return tweets[:max_tweets]


async def fetch_original_posts(
    client: AsyncClient, tweets: TweetList, rate_limiter: RateLimiter
) -> tuple[TweetDict, UserMap]:
    """Fetch original posts asynchronously."""
    # Collect tweet IDs from referenced tweets efficiently
    tweet_ids = [
        ref_tweet.id
        for tweet in tweets
        if hasattr(tweet, "referenced_tweets") and tweet.referenced_tweets
        for ref_tweet in tweet.referenced_tweets
        if ref_tweet.type in {"replied_to", "quoted"}
    ][:100]  # Limit to 100 per Twitter API batch size

    # Early return if no tweet IDs
    if not tweet_ids:
        return {}, {}

    endpoint = "tweets"
    # Check and wait for rate limit if necessary
    if rate_limiter.is_limited(endpoint):
        wait_time = rate_limiter.get_remaining_time(endpoint)
        if wait_time > 0:
            logging.info("Rate limit reached for %s. Waiting %f.2f seconds...", endpoint, wait_time)
            await asyncio.sleep(wait_time)

    try:
        response = await client.get_tweets(
            ids=tweet_ids,
            tweet_fields=["text", "created_at", "entities", "author_id"],
            expansions=["author_id"],
            user_fields=["username"],
        )
    except tweepy.errors.TooManyRequests:
        logging.warning("Too many requests for %s. Consider retrying later.", endpoint)
        return {}, {}
    except tweepy.errors.TweepyException:
        logging.exception("Tweepy error fetching original posts")
        return {}, {}
    except Exception:
        logging.exception("Unexpected error in fetch_original_posts")
        return {}, {}
    else:
        # Increment rate limiter only on successful API call
        rate_limiter.increment(endpoint)

        # Process response into tweet and user dictionaries
        original_posts: TweetDict = {
            tweet.id: tweet for tweet in (response.data or []) if tweet.id in tweet_ids
        }
        user_map: UserMap = {user.id: user.username for user in response.includes.get("users", [])}

        return original_posts, user_map


def save_to_json(tweets: TweetList, original_posts: TweetDict, user_map: UserMap, path: Path) -> bool:
    """
    Save tweet data using context manager with modern JSON features and zstd compression.

    Args:
        tweets: List of Tweet objects.
        original_posts: Dictionary mapping tweet IDs to Tweet objects or None.
        user_map: Dictionary mapping user IDs to usernames.
        path: Path object where the data will be saved.

    Returns:
        bool: True if the data was saved successfully, False otherwise.

    """
    try:
        # Ensure the directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data dictionary
        data = {
            "version": "1.0",
            "account_tweets": [tweet.data for tweet in tweets],
            "original_posts": {str(k): v.data if v else None for k, v in original_posts.items()},
            "user_map": {str(k): v for k, v in user_map.items()},
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Use zstd.open in text write mode with a specified compression level
        compressor = zstd.ZstdCompressor(level=4)
        with zstd.open(
            path.with_suffix(".json.zst"),
            "wt",
            cctx=compressor,
            encoding="utf-8",
        ) as f:
            json.dump(data, f, indent=None)

    except Exception:
        logging.exception("Error saving data to %s", str(path))
        return False
    else:
        logging.info("Data successfully saved to %s", path.with_suffix(".json_zst"))
        return True


def load_from_json(path: Path) -> tuple[TweetList, TweetDict, UserMap]:
    """
    Load tweet data with pattern matching and error handling.

    Args:
        path: Path object from where the data will be loaded.

    Returns:
        Tuple of:
            - List[Tweet]: List of Tweet objects.
            - Dict[int, Tweet | None]: Dictionary mapping tweet IDs to Tweet objects or None.
            - Dict[int, str]: Dictionary mapping user IDs to usernames.

    """
    if not (zst_path := path.with_suffix(".json.zst")):
        logging.warning("No data file found at %s", zst_path)
        return [], {}, {}

    try:
        # Decompress and load data using zstd
        decompressor = zstd.ZstdDecompressor()
        with zstd.open(zst_path, "rt", dctx=decompressor, encoding="utf-8") as file:
            data = json.load(file)

        # Reconstruct Tweet objects and dictionaries
        tweets = [tweepy.Tweet(data=tweet) for tweet in data.get("account_tweets", [])]
        original_posts = {
            int(k): tweepy.Tweet(data=v) if v else None for k, v in data.get("original_posts", {}).items()
        }
        user_map = {int(k): v for k, v in data.get("user_map", {}).items()}
    except (json.JSONDecodeError, zstd.ZstdError):
        logging.exception("Invalid JSON or zstd file in %s", path)
        return [], {}, {}
    except Exception:
        logging.exception("Error loading data from %s", path)
        return [], {}, {}
    else:
        logging.info("Data successfully loaded from %s", zst_path)
        return tweets, original_posts, user_map


def process_tweets(tweets: TweetList, chunk_size: int = 1000) -> pd.DataFrame:
    """
    Process a list of tweets into a pandas DataFrame efficiently with from_records.

    Args:
        tweets: List of tweepy.Tweet objects to process.
        chunk_size: Number of tweets to process per chunk (default: 1000).

    Returns:
        pd.DataFrame: DataFrame with columns: id, text, created_at, mentions, hashtags, referenced_tweets.

    Raises:
        ValueError: If chunk_size is not a positive integer.

    """
    if not tweets:
        logging.debug("No tweets provided to process_tweets.")
        return pd.DataFrame()

    if not isinstance(chunk_size, int) or chunk_size <= 0:
        error_message = f"chunk_size ust be a positive integer, got {chunk_size}"
        raise ValueError(error_message)

    logging.info("Processing %d tweets in chunks of %d", len(tweets), chunk_size)

    def tweet_to_dict(tweet: tweepy.Tweet) -> dict[str, Any]:
        """Extract tweet attributes into a dictionary in a single pass."""
        entities = getattr(tweet, "entities", {})
        return {
            "id": tweet.id,
            "text": tweet.text,
            "created_at": tweet.created_at,
            "mentions": [m.get("username", "") for m in entities.get("mentions", [])],
            "hashtags": [h.get("tag", "") for h in entities.get("hashtags", [])],
            "referenced_tweets": [
                {"id": ref.id, "type": ref.type} for ref in getattr(tweet, "referenced_tweets", []) or []
            ],
        }

    # Use iterator for lazy chunking
    tweet_iter = iter(tweets)
    dataframes: list[pd.DataFrame] = []

    chunk_start = 0
    while True:
        chunk = list(islice(tweet_iter, chunk_size))
        if not chunk:
            break

        try:
            # Extract data and create DataFrame
            chunk_data = [tweet_to_dict(tweet) for tweet in chunk]
            chunk_df = (
                pd.DataFrame.from_records(chunk_data)
                .dropna(subset=["id", "text"])
                .assign(
                    id=lambda df: df["id"].astype("int64"),
                    created_at=lambda df: pd.to_datetime(df["created_at"], errors="coerce", utc=True),
                )
            )

            if chunk_df.empty:
                logging.warning("Chunk starting at index %d has no valid data", chunk_start)
                chunk_start += chunk_size
                continue

            dataframes.append(chunk_df)
            logging.debug("Processed %d valid tweets in chunk starting at %d", len(chunk_df), chunk_start)
            chunk_start += chunk_size
        except AttributeError:
            logging.exception("Invalid tweet data in chunk starting at %d", chunk_start)
            chunk_start += chunk_size
            continue
        except Exception:
            logging.exception("Unexpected error in chunk starting at %d", chunk_start)
            chunk_start += chunk_size
            continue

    if dataframes:
        try:
            final_df = pd.concat(dataframes, ignore_index=True, copy=False)
        except ValueError:
            logging.exception("Failed to concatenate DataFrames")
            return pd.DataFrame()
        else:
            logging.info("Processed %d valid tweets into final DataFrame.", len(final_df))
            return final_df

    logging.debug("No valid data processed.")
    return pd.DataFrame()


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


def plot_temporal_patterns(
    temporal_data: TemporalAnalysisResult,
    path: Path,
    style: str = "seaborn-v0_8",
    dpi: int = 300,
) -> bool:
    """
    Plot temporal patterns with optimized plotting and robust error handling.

    Args:
        temporal_data: TemporalAnalysisResult object with temporal statistics.
        path: Path to save the plot.
        style: Matplotlib style context (default: "seaborn-v0_8").
        dpi: Resolution for saving the plot (default: 300).

    Returns:
        bool: True if plot saved successfully, False otherwise.

    """
    # Validate inputs via pydantic
    hour_counts = temporal_data.hour_counts
    day_counts = temporal_data.day_counts

    if hour_counts.empty and day_counts.empty:
        logging.warning("No temporal data to plot (both hour_counts and day_counts are empty).")
        return False

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        # Static day labels
        day_labels = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

        # Create figure with style context
        with plt.style.context(style):
            fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
            ax1, ax2 = axes

            # Plot hourly plots if data exists
            if not hour_counts.empty:
                sns.barplot(
                    x=hour_counts.index,
                    y=hour_counts.to_numpy(),
                    ax=ax1,
                    palette="virdis",
                    hue=hour_counts.index,  # Avoid FutureWarning
                    legend=False,
                )
                ax1.set(
                    title="Tweets per Hour",
                    xlabel="Hour of Day",
                    ylabel="Number of Tweets",
                )
                ax1.tick_params(axis="x", rotation=45)
                ax1.grid(True, linestyle="--", alpha="0.7")

            # Plot daily counts if data exists
            if not day_counts.empty:
                sns.barplot(
                    x=day_counts.index,
                    y=day_counts.to_numpy(),
                    ax=ax2,
                    palette="virdis",
                    hue=day_counts.index,  # Avoid FutureWarning
                    legend=False,
                )
                ax2.set(
                    title="Tweets per Day of Week",
                    xlabel="Day of Week",
                    ylabel="Number of Tweets",
                )
                ax2.set_xticks(range(len(day_counts)))
                ax2.set_xticklabels([day_labels[i] for i in day_counts.index], rotation=45)
                ax2.grid(True, linestyle="--", alpha="0.7")

            # Save and close figure
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            logging.info("Temporal patterns plot saved to %s", path)
            return True
    except OSError:
        logging.exception("Failed to save plot to %s", path)
        return False
    except Exception:
        logging.exception("Error plotting temporal data")
        return False


def plot_content_visualizations(
    content_data: ContentAnalysisResult,
    path: Path,
    style: str = "seaborn-v0_8",
    dpi: int = 300,
    wordcloud_params: dict[str, Any] | None = None,
) -> bool:
    """
    Generate and save a word cloud for top terms in content analysis.

    Args:
        content_data: ContentAnalysisResult object with top terms.
        path: Path to save the word cloud image.
        style: Matplotlib style context (default: "seaborn-v0_8").
        dpi: Resolution for saving the plot (default: 300).
        wordcloud_params: Optional dict of additional WordCloud parameters.

    Returns:
        bool: True if plot saved successfully, False otherwise.

    """
    top_terms = content_data.top_terms
    if not top_terms:
        logging.warning("No top terms provided for word cloud plotting.")
        return False

    try:
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Default WordCloud parameters with option to override
        default_params: dict[str, Any] = {
            "width": 800,
            "height": 400,
            "background_color": "white",
            "max_words": 50,
        }
        if wordcloud_params:
            default_params.update(wordcloud_params)

        # Validate top_terms are strings
        if not all(isinstance(term, str) for term in top_terms):
            error_message = "All top_terms must be strings."
            raise ValueError(error_message)

        # Generate word cloud
        wordcloud = WordCloud(**default_params).generate(" ".join(top_terms))

        # Plot with style context
        with plt.style.context(style):
            fig = plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.title("Word Cloud of Top Terms", pad=10)  # Add padding to avoid overlap
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

        logging.info("Word cloud plot saved to %s", path)

    except OSError:
        logging.exception("Failed to save word cloud to %s", path)
        return False
    except ValueError:
        logging.exception("Invalid data for word cloud")
        return False
    except Exception:
        logging.exception("Error generating word cloud: %s")
        return False
    else:
        return True


async def main(account: str, config: ConfigModel, log_level: str = "INFO") -> int:
    """
    Main execution function orchestrating Twitter data analysis with concurrent operations.

    Args:
        account: Twitter username to analyze (e.g., "example_user").
        config: Configuration object with analysis parameters.
        log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR").

    Returns:
        int: Exit code (0 for success, 1 for failure).

    """
    # Configure logging with the provided level
    setup_logging(log_level)
    logging.info("Starting analysis for @%s with log_level %s", account, log_level)

    try:
        # Step 1: Initialize components
        paths = Paths(account)
        rate_limiter = RateLimiter(max_requests=config.rate_limits, reset_interval=config.reset_interval)

        # Step 2: Load NLP model with fallback
        nlp = load_nlp_model(config.nlp_model)
        if nlp is None:
            logging.warning("Proceeding with basic tokenization due to NLP model failure.")

        # Step 3: Fetch or load tweet data
        tweets, original_posts, user_map = load_from_json(paths.DATA_FILE)
        if not tweets or (
            paths.DATA_FILE.exists() and (time.time() - paths.DATA_FILE.stat().st_mtime > 3600)
        ):
            logging.info("Cache missing or outdated; fetching new data.")
            bearer_token = get_bearer_token()
            client = AsyncClient(bearer_token)

            # Get user ID
            try:
                response = await client.get_user(username=account)
                if not response.data:
                    logging.error("User @%s account not found.", account)
                    return 1
                user_id = response.data.id
            except tweepy.errors.TweepyException:
                logging.exception("Failed to fetch user ID for @%s", account)
                return 1

            # Fetch tweets and original posts
            tweets = await fetch_tweets(client, user_id, rate_limiter, config.max_tweets)
            if not tweets:
                logging.warning("No tweets retrieved for @%s", account)
                return 1
            original_posts, user_map = await fetch_original_posts(client, tweets, rate_limiter)
            save_to_json(tweets, original_posts, user_map, paths.DATA_FILE)

        # Step 4: Process tweets into DataFrame
        df = process_tweets(tweets)
        if df.empty:
            logging.error("Processed tweet data is empty.")
            return 1

        # Step 5: Perform analyses
        temporal_data = analyze_temporal_patterns(df)
        content_data = analyze_content_dynamic(df, nlp)

        # Step 6: Generate visualizations concurently
        async with asyncio.TaskGroup() as tg:
            _ = tg.create_task(
                asyncio.to_thread(plot_temporal_patterns, temporal_data, paths.TEMPORAL_PLOT)
            )
            _ = tg.create_task(
                asyncio.to_thread(plot_content_visualizations, content_data, paths.WORDCLOUD_PLOT)
            )

        # Step 7: Save analysis results
        temporal_scalars = {
            "avg_time_between": temporal_data.avg_time_between,
            "std_time_between": temporal_data.std_time_between,
            "cv_time_between": temporal_data.cv_time_between,
            "num_bursts": temporal_data.num_bursts,
            "avg_burst_size": temporal_data.avg_burst_size,
        }
        content_scalars = {
            "type_token_ratio": content_data.type_token_ratio,
            "avg_sentence_length": content_data.avg_sentence_length,
            "text_similarity": content_data.text_similarity,
            "sentiment_compound": content_data.sentiment_stats["compound"],
            "sentiment_positive": content_data.sentiment_stats["positive"],
            "sentiment_negative": content_data.sentiment_stats["negative"],
            "sentiment_neutral": content_data.sentiment_stats["neutral"],
        }
        results_df = pd.DataFrame(
            {
                "metric": list(temporal_scalars.keys()) + list(content_scalars.keys()),
                "value": list(temporal_scalars.values()) + list(content_scalars.values()),
            }
        )
        results_df.to_csv(paths.OUTPUT_CSV, index=False)
        logging.info("Results saved to %s", str(Paths.OUTPUT_CSV))

        # Step 8: Log summary
        summary_table = tabulate(
            results_df[["metric", "value"]].values, headers=["Metric", "Value"], tablefmt="pretty"
        )
        logging.info(f"Analysis Summary:\n{summary_table}")

        logging.info("Analysis completed successfully for @%s", account)

    except* Exception as errors:
        for error in errors.exceptions:
            logging.exception("Unhandled exception occurred", exc_info=error)
    else:
        return 0

    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Twitter account activity.")
    parser.add_argument("account", type=str, help="Twitter account username")
    parser.add_argument("--config", "-c", type=str, help="Path to configuration file")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level (default: INFO)",
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)

    # Run the main function with the parsed log level
    try:
        exit_code = asyncio.run(main(args.account, config, args.log_level))
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logging.info("Analysis interrupted by user.")
        sys.exit(130)
    except Exception:
        logging.exception("Fatal error")
        sys.exit(1)
