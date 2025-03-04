import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import tweepy
import zstandard as zstd

from twitter_analysis.models.tweet import TweetDict, TweetList, UserMap


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
