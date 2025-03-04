import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import tweepy.errors
from tabulate import tabulate
from tweepy.asynchronous import AsyncClient

from twitter_analysis.analysis import analyze_content_dynamic, analyze_sentiment, analyze_temporal_patterns
from twitter_analysis.config import ConfigModel, load_config, setup_logging
from twitter_analysis.data import (
    fetch_original_posts,
    fetch_tweets,
    get_bearer_token,
    load_from_json,
    process_tweets,
    save_to_json,
)
from twitter_analysis.utils import Paths, RateLimiter, load_nlp_model
from twitter_analysis.visualization import plot_content_visualizations, plot_temporal_patterns


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
        nlp = load_nlp_model(config.nlp_model)
        if nlp is None:
            logging.warning("Proceeding with basic tokenization due to NLP model failure.")

        # Step 2: Fetch or load tweet data
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

        # Step 3: Process tweets into DataFrame
        df = process_tweets(tweets)
        if df.empty:
            logging.error("Processed tweet data is empty.")
            return 1

        # Step 4: Perform analyses
        temporal_data = analyze_temporal_patterns(df)
        content_data = analyze_content_dynamic(df, nlp)
        sentiment_stats = analyze_sentiment(df)  # Separated sentiment analysis
        content_data.sentiment_stats = sentiment_stats  # Attach sentiment results

        # Step 5: Generate visualizations concurently
        async with asyncio.TaskGroup() as tg:
            tg.create_task(asyncio.to_thread(plot_temporal_patterns, temporal_data, paths.TEMPORAL_PLOT))
            tg.create_task(
                asyncio.to_thread(plot_content_visualizations, content_data, paths.WORDCLOUD_PLOT)
            )

        # Step 6: Save analysis results
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
