"""Unit tests for LLM schema helpers and defaults."""

import pytest

from floeval.config.schemas.io.llm import OpenAIProviderConfig, _normalize_openai_base_url

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("api.openai.com", "https://api.openai.com/openai/v1"),
        ("https://api.openai.com/v1", "https://api.openai.com/v1"),
        ("https://x.example.com/chat/completions", "https://x.example.com/openai/v1"),
        ("https://x.example.com/embeddings", "https://x.example.com/openai/v1"),
    ],
)
def test_normalize_openai_base_url(raw: str, expected: str) -> None:
    assert _normalize_openai_base_url(raw) == expected


def test_normalize_openai_base_url_rejects_empty() -> None:
    with pytest.raises(ValueError, match="base_url cannot be empty"):
        _normalize_openai_base_url("  ")


def test_openai_provider_defaults() -> None:
    cfg = OpenAIProviderConfig(api_key="unit-test-key")
    assert cfg.chat_model == "gpt-3.5-turbo"
    assert cfg.chat_endpoint == "chat/completions"
    assert cfg.embedding_model == "text-embedding-3-small"
