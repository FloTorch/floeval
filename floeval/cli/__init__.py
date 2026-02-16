from typing import Any, TypedDict

from pydantic import BaseModel


class ConfigError(Exception):
    """Custom exception for configuration-related errors."""

    pass


class CLIEvaluationConfig(BaseModel):
    llm_config: dict[str, Any]
    evaluation_config: dict[str, Any]


class DatasetGenConfig(TypedDict):
    generator_model: str


class CLIGenerationConfig(BaseModel):
    llm_config: dict[str, Any]
    dataset_generation_config: DatasetGenConfig
