import asyncio
import logging
from collections.abc import AsyncGenerator
from os import getenv
from pathlib import Path

import tweepy
from dotenv import load_dotenv
from tweepy.asynchronous import AsyncClient, AsyncPaginator
from tweepy.errors import TooManyRequests, TweepyException

from twitter_analysis.models.tweet import TweetDict, TweetList, UserMap
from twitter_analysis.utils.rate_limiter import RateLimiter


def get_bearer_token() -> str:
    """
    Retrieve bearer token securely from environment or .env file.

    Returns:
        str: The retrieved bearer token.

    Raises:
        ValueError: If bearer token cannot be found in environment or .env file.

    """

    def _get_token_from_env() -> str | None:
        """
        Helper function to get token from environment variables.

        Returns:
            str | None: The token if found, None otherwise.

        """
        return getenv("X_BEARER_TOKEN")

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


async def fetch_tweets(
    client: AsyncClient, user_id: int, rate_limiter: RateLimiter, max_tweets: int = 500
) -> TweetList:
    """
    Fetch recent tweets asynchronously with proper error handling and backoff.

    Args:
        client (AsyncClient): Tweepy asynchronous client instance.
        user_id (int): Twitter user ID to fetch tweets for.
        rate_limiter (RateLimiter): Rate limiting handler instance.
        max_tweets (int): Maximum number of tweets to fetch (default: 500, max: 500).

    Returns:
        TweetList: List of retrieved tweets, limited to max_tweets.

    Notes:
        - Implements exponential backoff for rate limiting.
        - Handles various response types from the Twitter API.
        - Logs errors and rate limit events.

    """
    tweets: TweetList = []
    tweet_fields = ["text", "created_at", "entities", "in_reply_to_user_id", "referenced_tweets"]
    max_tweets = min(max_tweets, 500)
    endpoint = "users_tweets"

    # Define an async generator for paginated tweets
    async def paginated_tweets() -> AsyncGenerator[TweetList, None]:
        """
        Generate batches of tweets through pagination.

        Yields:
            TweetList: Batch of tweets from each pagination page.

        """
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
    except TweepyException as e:
        match e:
            case TooManyRequests():
                logging.warning("Rate limit hit during pagination. Consider retrying later.")
            case _:
                logging.exception("Error fetching tweets")
    except Exception:
        logging.exception("Unexpected error fetching tweets")

    return tweets[:max_tweets]


async def fetch_original_posts(
    client: AsyncClient, tweets: TweetList, rate_limiter: RateLimiter
) -> tuple[TweetDict, UserMap]:
    """
    Fetch original posts referenced in the given tweets asynchronously.

    Args:
        client (AsyncClient): Tweepy asynchronous client instance.
        tweets (TweetList): List of tweets to check for referenced posts.
        rate_limiter (RateLimiter): Rate limiting handler instance.

    Returns:
        tuple[TweetDict, UserMap]: Tuple containing:
            - TweetDict: Dictionary mapping tweet IDs to tweet objects.
            - UserMap: Dictionary mapping user IDs to usernames.

    Notes:
        - Limited to 100 referenced tweets per API call per Twitter API restrictions.
        - Handles rate limiting and various error conditions.

    """
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
    except TooManyRequests:
        logging.warning("Too many requests for %s. Consider retrying later.", endpoint)
        return {}, {}
    except TweepyException:
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
