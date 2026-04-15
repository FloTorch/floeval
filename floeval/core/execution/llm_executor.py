"""OpenAI-compatible LLM provider with separate sync and async clients.

Uses native openai SDK (not LangChain) for direct, lightweight LLM access.
- Sync path: openai.OpenAI (httpx.Client) — no threads, no event loops
- Async path: openai.AsyncOpenAI (httpx.AsyncClient) — native async, no bridges
"""

import logging
from typing import Any

import openai

from floeval.config.schemas.io.llm import (
    LLMProviderConfig,
    OpenAIProviderConfig,
    _normalize_openai_base_url,
)
from floeval.core.execution.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI-compatible LLM provider with separate sync and async clients.

    Works with any OpenAI-compatible API: OpenAI, Azure OpenAI, vLLM, LiteLLM, etc.

    Key design:
    - No ThreadPoolExecutor — sync client is natively sync (httpx.Client)
    - No sync/async loop bridging — async client is natively async (httpx.AsyncClient)
    - Lazy client initialization — clients created on first use
    - Separate client instances — no shared state between sync and async paths
    """

    def __init__(
        self,
        config: LLMProviderConfig | None = None,
        extra_headers: dict[str, str] | None = None,
        **kwargs,
    ):
        """Initialize with LLM config.

        Args:
            config: LLMProviderConfig or OpenAIProviderConfig instance.
                If None, creates OpenAIProviderConfig from kwargs.
            extra_headers: Optional headers forwarded on every API request
                (e.g. gateway run-context headers for log correlation).
        """
        if config is None:
            config = OpenAIProviderConfig(**kwargs)
        self.config = config
        self._base_url = _normalize_openai_base_url(config.base_url)
        self._chat_model = config.chat_model
        self._system_prompt = config.system_prompt
        self._extra_headers: dict[str, str] = dict(extra_headers or {})

        # Lazy-initialized, separate clients — no shared state
        self._sync_client: openai.OpenAI | None = None
        self._async_client: openai.AsyncOpenAI | None = None

    @property
    def sync_client(self) -> openai.OpenAI:
        """Lazily create sync OpenAI client."""
        if self._sync_client is None:
            self._sync_client = openai.OpenAI(
                base_url=self._base_url,
                api_key=self.config.api_key,
                default_headers=self._extra_headers or None,
            )
        return self._sync_client

    @property
    def async_client(self) -> openai.AsyncOpenAI:
        """Lazily create async OpenAI client."""
        if self._async_client is None:
            self._async_client = openai.AsyncOpenAI(
                base_url=self._base_url,
                api_key=self.config.api_key,
                default_headers=self._extra_headers or None,
            )
        return self._async_client

    def _build_messages(
        self, prompt: str, system_prompt: str | None = None
    ) -> list[dict[str, str]]:
        """Build message list for chat completion."""
        messages: list[dict[str, str]] = []
        sys_prompt = system_prompt or self._system_prompt
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(
        self, prompt: str, system_prompt: str | None = None, **kwargs: Any
    ) -> str:
        """Sync generation using openai.OpenAI (no threads, no event loops).

        Args:
            prompt: User prompt text.
            system_prompt: Optional system prompt override.
            **kwargs: Additional params passed to chat.completions.create
                (e.g., temperature, max_tokens).

        Returns:
            Generated response text.
        """
        messages = self._build_messages(prompt, system_prompt)
        logger.debug(
            "LLM request: model=%s, messages=%d", self._chat_model, len(messages)
        )

        params: dict[str, Any] = {
            "model": self._chat_model,
            "messages": messages,
        }
        params.update(kwargs)

        response = self.sync_client.chat.completions.create(**params)
        content = response.choices[0].message.content
        logger.debug("LLM response: %s chars", len(content) if content else 0)
        return content or ""

    async def agenerate(
        self, prompt: str, system_prompt: str | None = None, **kwargs: Any
    ) -> str:
        """Async generation using openai.AsyncOpenAI (natively awaitable).

        Args:
            prompt: User prompt text.
            system_prompt: Optional system prompt override.
            **kwargs: Additional params passed to chat.completions.create.

        Returns:
            Generated response text.
        """
        messages = self._build_messages(prompt, system_prompt)
        logger.debug(
            "LLM async request: model=%s, messages=%d",
            self._chat_model,
            len(messages),
        )

        params: dict[str, Any] = {
            "model": self._chat_model,
            "messages": messages,
        }
        params.update(kwargs)

        response = await self.async_client.chat.completions.create(**params)
        content = response.choices[0].message.content
        logger.debug("LLM async response: %s chars", len(content) if content else 0)
        return content or ""

    async def agenerate_with_messages(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> str:
        """Generate with full message list (e.g. for few-shot prompting).

        Args:
            messages: Full chat message list, e.g. [system, user, assistant, user, ...].
            **kwargs: Additional params passed to chat.completions.create.

        Returns:
            Generated response text.
        """
        params: dict[str, Any] = {
            "model": self._chat_model,
            "messages": messages,
        }
        params.update(kwargs)
        response = await self.async_client.chat.completions.create(**params)
        content = response.choices[0].message.content
        return content or ""

    def generate_embedding(self, text: str, **kwargs: Any) -> list[float]:
        """Sync embedding generation.

        Args:
            text: Text to embed.
            **kwargs: Additional params.

        Returns:
            Embedding vector.

        Raises:
            NotImplementedError: If no embedding_model configured.
        """
        if not self.config.embedding_model:
            raise NotImplementedError("No embedding_model configured.")
        response = self.sync_client.embeddings.create(
            model=self.config.embedding_model,
            input=text,
            **kwargs,
        )
        return response.data[0].embedding

    async def agenerate_embedding(self, text: str, **kwargs: Any) -> list[float]:
        """Async embedding generation.

        Args:
            text: Text to embed.
            **kwargs: Additional params.

        Returns:
            Embedding vector.

        Raises:
            NotImplementedError: If no embedding_model configured.
        """
        if not self.config.embedding_model:
            raise NotImplementedError("No embedding_model configured.")
        response = await self.async_client.embeddings.create(
            model=self.config.embedding_model,
            input=text,
            **kwargs,
        )
        return response.data[0].embedding

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
