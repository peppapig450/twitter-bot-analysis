import logging
from itertools import islice
from typing import Any

import pandas as pd
import tweepy

from twitter_analysis.models.tweet import TweetList


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
