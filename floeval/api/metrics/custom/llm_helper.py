"""LLM helper for custom metrics with separate sync and async clients.

Provides SimpleLLMHelper, a lightweight wrapper around the OpenAI API
for use within custom metrics (criteria, decorator). Exposes:
- generate(): Synchronous API using openai.OpenAI (httpx.Client — no threads)
- agenerate(): Asynchronous API using openai.AsyncOpenAI (httpx.AsyncClient — native async)

No ThreadPoolExecutor, no asyncio.new_event_loop(), no shared mutable state.
"""

import logging
import threading
from typing import Any

import openai

from floeval.config.schemas.io.llm import OpenAIProviderConfig, _normalize_openai_base_url

logger = logging.getLogger(__name__)


class SimpleLLMHelper:
    """LLM helper with truly separate sync and async paths.

    Design:
    - generate() → openai.OpenAI (sync httpx.Client) — no threads needed
    - agenerate() → openai.AsyncOpenAI (async httpx.AsyncClient) — native async
    - No shared client state — each path has its own client instance
    - Lazy initialization — clients created on first use
    - Thread-safe — no mutable shared state between paths

    Args:
        config: OpenAI provider configuration.
        chat_model: Override model name (defaults to config.chat_model).
    """

    def __init__(self, config: OpenAIProviderConfig, chat_model: str | None = None):
        """Initialize with config. Clients are created lazily on first use."""
        self._config = config
        self._chat_model = chat_model or config.chat_model
        self._base_url = _normalize_openai_base_url(config.base_url)
        self._api_key = config.api_key

        # Separate clients — lazily initialized, independent lifecycle
        self._sync_client: openai.OpenAI | None = None
        self._async_client: openai.AsyncOpenAI | None = None
        self._sync_lock: threading.Lock = threading.Lock()

    @property
    def sync_client(self) -> openai.OpenAI:
        """Get or create sync client (thread-safe via double-checked locking)."""
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
        """Get or create async client."""
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
        """Build request parameters (shared between sync and async)."""
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
        """Synchronous LLM call using openai.OpenAI (sync httpx.Client).

        This is a true sync call — no ThreadPoolExecutor, no event loops.
        Safe to call from any context (sync functions, threads, workers).

        Args:
            prompt: The prompt text for the LLM.
            temperature: Sampling temperature (default 0.0 for deterministic).
            max_tokens: Maximum tokens in response (None = provider default).

        Returns:
            Generated text response.

        Raises:
            openai.APITimeoutError: If the request timed out.
            openai.APIError: For other API errors.
        """
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
        """Asynchronous LLM call using openai.AsyncOpenAI (async httpx.AsyncClient).

        Natively awaitable — no threads, no sync bridges.
        Use this inside async metric functions and aevaluate() paths.

        Args:
            prompt: The prompt text for the LLM.
            temperature: Sampling temperature (default 0.0 for deterministic).
            max_tokens: Maximum tokens in response (None = provider default).

        Returns:
            Generated text response.

        Raises:
            openai.APITimeoutError: If the request timed out.
            openai.APIError: For other API errors.
        """
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
