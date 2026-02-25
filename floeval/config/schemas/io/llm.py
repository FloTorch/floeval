from pydantic import BaseModel, Field


def _normalize_openai_base_url(url: str) -> str:
    """Return OpenAI-compatible API base URL (adds /openai/v1 if needed).

    Ensures URL has http/https, strips trailing paths, and ends with
    /openai/v1 or /v1 for OpenAI-compatible clients.

    Args:
        url: Raw base URL string.

    Returns:
        Normalized URL suitable for openai.OpenAI(base_url=...).

    Raises:
        ValueError: If url is empty.
    """
    raw = url.strip()
    if not raw:
        raise ValueError("base_url cannot be empty.")
    if not raw.startswith(("http://", "https://")):
        raw = f"https://{raw}"
    raw = raw.rstrip("/")
    for suffix in ("/chat/completions", "/embeddings"):
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)]
            raw = raw.rstrip("/")
    if raw.endswith("/openai/v1") or raw.endswith("/v1"):
        return raw
    return f"{raw}/openai/v1"


class LLMProviderConfig(BaseModel):
    """Generic LLM provider configuration for LLM-based evaluation execution."""

    provider_type: str = Field(
        default="openai",
        description="Provider type identifier. Currently supports 'openai' and any "
        "OpenAI-compatible API (e.g., Azure OpenAI, vLLM, LiteLLM).",
    )
    base_url: str = Field(..., description="Base URL for the LLM provider API.")
    api_key: str = Field(
        ..., description="API key or token used to authenticate with the provider."
    )
    chat_model: str = Field(
        ..., description="Model name or ID to use for chat/completion requests."
    )
    chat_endpoint: str = Field(
        ..., description="Endpoint/path for chat/completion requests on the provider."
    )
    embedding_model: str | None = Field(
        default=None, description="Optional model name/ID to use for embeddings."
    )
    embedding_endpoint: str | None = Field(
        default=None, description="Optional endpoint/path for embedding requests."
    )
    system_prompt: str | None = Field(
        default=None,
        description="Optional default system prompt to include in chat requests.",
    )
    extra_kwargs: dict | None = Field(
        default=None,
        description="Optional additional provider-specific keyword arguments.",
    )


class OpenAIProviderConfig(LLMProviderConfig):
    """OpenAI-compatible provider configuration with sensible defaults."""

    base_url: str = Field(
        "https://api.openai.com/v1",
        description="Base URL for the OpenAI API.",
    )
    api_key: str = Field(
        ...,
        description="OpenAI API key (prefer using the OPENAI_API_KEY environment variable).",
    )
    chat_model: str = Field(
        "gpt-3.5-turbo",
        description="Default chat model for OpenAI requests.",
    )
    chat_endpoint: str = Field(
        default="chat/completions",
        description="Endpoint path for chat/completion requests on OpenAI.",
    )
    embedding_model: str | None = Field(
        default="text-embedding-3-small",
        description="Default embedding model for OpenAI embeddings.",
    )
    embedding_endpoint: str | None = Field(
        default="embeddings",
        description="Endpoint path for embedding requests on OpenAI.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Optional default system prompt to include in chat requests.",
    )
    extra_kwargs: dict | None = Field(
        default=None,
        description="Additional provider-specific keyword arguments passed to the OpenAI client.",
    )
