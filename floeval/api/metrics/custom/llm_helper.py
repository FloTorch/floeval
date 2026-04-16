"""Shared LLM helper for custom metric execution paths."""

import logging
import threading
from typing import Any

import openai

from floeval.config.schemas.io.llm import OpenAIProviderConfig, _normalize_openai_base_url

logger = logging.getLogger(__name__)


class SimpleLLMHelper:
    """OpenAI wrapper with dedicated sync and async clients."""

    def __init__(self, config: OpenAIProviderConfig, chat_model: str | None = None):
        """Create helper; clients are initialized lazily."""
        self._config = config
        self._chat_model = chat_model or config.chat_model
        self._base_url = _normalize_openai_base_url(config.base_url)
        self._api_key = config.api_key

        # Sync and async clients are intentionally independent.
        self._sync_client: openai.OpenAI | None = None
        self._async_client: openai.AsyncOpenAI | None = None
        self._sync_lock: threading.Lock = threading.Lock()

    @property
    def sync_client(self) -> openai.OpenAI:
        """Return a cached sync client."""
        if self._sync_client is None:
            with self._sync_lock:
                if self._sync_client is None:
                    self._sync_client = openai.OpenAI(
                        base_url=self._base_url,
                        api_key=self._api_key,
                    )
        return self._sync_client

    @property
    def async_client(self) -> openai.AsyncOpenAI:
        """Return a cached async client."""
        if self._async_client is None:
            self._async_client = openai.AsyncOpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
            )
        return self._async_client

    def _build_params(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Build common completion parameters."""
        params: dict[str, Any] = {
            "model": self._chat_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        return params

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """Run a synchronous chat completion request."""
        params = self._build_params(prompt, temperature, max_tokens)
        try:
            response = self.sync_client.chat.completions.create(**params)
            content = response.choices[0].message.content
            return content or ""
        except openai.APITimeoutError as e:
            logger.error("LLM call timed out: %s", e)
            raise
        except openai.APIError as e:
            logger.error("LLM API error (status=%s): %s", getattr(e, "status_code", "N/A"), e)
            raise

    async def agenerate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """Run an asynchronous chat completion request."""
        params = self._build_params(prompt, temperature, max_tokens)
        try:
            response = await self.async_client.chat.completions.create(**params)
            content = response.choices[0].message.content
            return content or ""
        except openai.APITimeoutError as e:
            logger.error("Async LLM call timed out: %s", e)
            raise
        except openai.APIError as e:
            logger.error(
                "Async LLM API error (status=%s): %s",
                getattr(e, "status_code", "N/A"),
                e,
            )
            raise

    def close(self) -> None:
        """Clean up sync client."""
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None

    async def aclose(self) -> None:
        """Clean up async client."""
        if self._async_client:
            await self._async_client.close()
            self._async_client = None
