"""Fixtures for backend integration tests."""

import pytest


@pytest.fixture
def backend_integration_env(requires_llm_credentials) -> None:
    """Gate LLM-backed backend tests on credentials."""
    return None
