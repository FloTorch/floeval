from typing import Any

from pydantic import BaseModel


class ConfigError(Exception):
    """Custom exception for configuration-related errors."""

    pass


class CLIEvaluationConfig(BaseModel):
    llm_config: dict[str, Any]
    evaluation_config: dict[str, Any]
    dataset_generation_config: dict[str, Any] | None = None
    agent_workflow_config: dict[str, Any] | None = None


class DatasetGenConfig(BaseModel):
    """Configuration for dataset generation from partial datasets."""

    generator_model: str
    batch_size: int = 20
    max_concurrency: int = 10


class CLIGenerationConfig(BaseModel):
    llm_config: dict[str, Any]
    dataset_generation_config: DatasetGenConfig
