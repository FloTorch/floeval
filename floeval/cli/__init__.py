from typing import Any

from pydantic import BaseModel


class ConfigError(Exception):
    """Custom exception for configuration-related errors."""

    pass


class CliConfig(BaseModel):
    gateway_config: dict[str, Any]
    evaluation_config: dict[str, Any]
