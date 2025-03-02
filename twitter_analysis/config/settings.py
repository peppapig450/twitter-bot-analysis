import json
import logging
from pathlib import Path
from typing import Literal, Required, TypedDict, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class RateLimits(TypedDict):
    """
    Define the structure of rate_limits.
    """

    users_tweets: Required[int]
    tweets: Required[int]


type RateLimitEndpoint = Literal["users_tweets", "tweets"]


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
    model_config = ConfigDict(
        str_strip_whitespace=True,  # Automatically strip whitespace from strings
        frozen=True,  # Make the model immutable after creation
        extra="forbid",  # Prevent extra fields
    )


def load_config(config_path: Path | None = None) -> ConfigModel:
    """
    Load and validate configuration from a JSON file or return defaults.

    Args:
        config_path: Optional path to the configuration file

    Returns:
        ConfigModel instance with validated configuration

    Raises:
        ValueError: If the config file exists but cannot be parsed

    """
    if config_path is None or not config_path.is_file():
        logging.info("No config file provided or found, using default configuration")
        return ConfigModel()

    try:
        config_text = config_path.read_text(encoding="utf-8")
        config = ConfigModel.model_validate_json(config_text)
        logging.info("Successfully loaded configuration from %s", config_path)
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in config file {config_path}"
        logging.exception(error_msg)
        raise ValueError(error_msg) from e
    except ValidationError as e:
        error_msg = f"Configuration validation failed for {config_path}"
        logging.exception(error_msg)
        raise ValueError(error_msg) from e
    except Exception:
        error_msg = f"Unexpected error loading config from {config_path}"
        logging.exception(error_msg)
        raise
    else:
        return config
