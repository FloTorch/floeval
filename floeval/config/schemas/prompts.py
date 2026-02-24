"""Prompt schemas for prompt-based response generation."""

from pydantic import BaseModel, Field


class Prompt(BaseModel):
    """Single prompt definition for response generation."""

    template: str = Field(..., description="System prompt template text")
    temperature: float | None = Field(default=None, description="Temperature override")
    max_tokens: int | None = Field(default=None, description="Max tokens override")


class PromptFile(BaseModel):
    """Collection of prompts loaded from YAML/JSON file."""

    prompts: dict[str, Prompt] = Field(
        ..., description="Mapping of prompt_id to Prompt"
    )
