from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    """Abstract base class for execution engines."""

    name: str

    @abstractmethod
    def generate(self, *args, **kwargs) -> str:
        """Generate output based on input arguments."""

    @abstractmethod
    def generate_embedding(self, *args, **kwargs) -> list[float]:
        """Generate an embedding based on input arguments."""
