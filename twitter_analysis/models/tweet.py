from typing import TYPE_CHECKING, TypedDict

import pandas as pd

if TYPE_CHECKING:
    from tweepy import Tweet


class TweetData(TypedDict):
    """
    Define the structure of tweet data using TypedDict for type safety.
    """

    id: int
    text: str
    created_at: pd.Timestamp | None
    mentions: list[str]
    hashtags: list[str]
    referenced_tweets: list[dict[str, str | int]]


# Type aliases for clarity
type TweetList = list[Tweet]
type TweetDict = dict[int, Tweet | None]
type UserMap = dict[int, str]
