import asyncio
import logging
from dataclasses import dataclass, field

from twitter_analysis.config.settings import RateLimits


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
