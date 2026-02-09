"""
Base metric abstract class and result model
"""

from abc import ABC, abstractmethod
from typing import Any, Mapping


class MetricResult:
    """Result container for metric evaluation."""

    def __init__(self, score: float | None, metadata: Mapping[str, Any] | None = None):
        self.score = score
        self.metadata = dict(metadata) if metadata else {}


class BaseMetric(ABC):
    """Abstract base class for all metrics."""

    def __init__(self, name: str, *args, **kwargs):
        self.name: str = name
        self.provider: str | None = None

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> MetricResult:
        """Evaluate metric synchronously."""
        pass

    async def aevaluate(self, *args, **kwargs) -> MetricResult:
        """Evaluate metric asynchronously. Default runs evaluate() in executor."""
        import asyncio
        
        if type(self).aevaluate is not BaseMetric.aevaluate:
            return await type(self).aevaluate(self, *args, **kwargs)
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.evaluate, *args, **kwargs)
        return result
