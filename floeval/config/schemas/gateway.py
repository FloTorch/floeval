"""
Unified gateway configuration for all metric providers (RAGAS, DeepEval, etc.).

A single config type allows mixed metrics from multiple providers without
provider-specific gateway types at the Evaluation API.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class GatewayConfig(BaseModel):
    """
    Single gateway configuration used by RAGAS, DeepEval, and future providers.

    All fields are optional to support environment-driven defaults and partial overrides.
    Providers use only the fields they need (e.g. RAGAS ignores temperature/max_tokens).
    """

    base_url: Optional[str] = Field(
        default=None,
        description="Base URL for custom API gateway (OpenAI-compatible)",
    )
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    chat_model: Optional[str] = Field(default=None, description="LLM model identifier")
    embedding_model: Optional[str] = Field(
        default=None,
        description="Embedding model identifier",
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Temperature for LLM generation (provider will use its default if not provided)",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Max tokens for LLM generation (provider will use its default if not provided)",
    )
