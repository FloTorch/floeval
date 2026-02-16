from pydantic import BaseModel, Field


class LLMProviderConfig(BaseModel):
    """Generic LLM provider configuration for LLM-based evaluation execution."""

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
        None, description="Optional model name/ID to use for embeddings."
    )
    embedding_endpoint: str | None = Field(
        None, description="Optional endpoint/path for embedding requests."
    )
    system_prompt: str | None = Field(
        None, description="Optional default system prompt to include in chat requests."
    )
    extra_kwargs: dict | None = Field(
        None, description="Optional additional provider-specific keyword arguments."
    )


class OpenAIProviderConfig(BaseModel):
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
