"""Base LLM provider interface with sync and async paths.

All providers MUST implement both sync and async methods.
This enables the framework to use the correct path based on
whether the evaluation is running in sync or async mode.
"""

from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.

    Providers implement separate sync and async methods using
    separate client instances (e.g., openai.OpenAI vs openai.AsyncOpenAI).
    No ThreadPoolExecutor or event-loop bridging needed.
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response synchronously."""
        ...

    @abstractmethod
    async def agenerate(self, prompt: str, **kwargs) -> str:
        """Generate a response asynchronously."""
        ...

    @abstractmethod
    def generate_embedding(self, text: str, **kwargs) -> list[float]:
        """Generate embeddings synchronously."""
        ...

    @abstractmethod
    async def agenerate_embedding(self, text: str, **kwargs) -> list[float]:
        """Generate embeddings asynchronously."""
        ...

    def close(self) -> None:
        """Clean up sync resources. Override if needed."""
        pass

    async def aclose(self) -> None:
        """Clean up async resources. Override if needed."""
        pass
