"""
Base metric abstract class and result model
"""

import asyncio
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
        """Default: run evaluate() in executor. Override for async."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.evaluate(*args, **kwargs))
