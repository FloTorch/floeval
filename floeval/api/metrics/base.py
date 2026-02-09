"""
Base metric abstract class and result model
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any


class MetricResult:
    """
    Result container for metric evaluation.
    
    score=None indicates evaluation failed (differentiates from actual 0.0 scores).
    """

    def __init__(self, score: float | None, metadata: Mapping[str, Any] | None = None):
        self.score = score
        self.metadata = dict(metadata) if metadata else {}


class BaseMetric(ABC):
    """
    Abstract base class for all metrics.
    
    Supports both synchronous and asynchronous evaluation.
    All metrics must implement compute(). Async support is optional.
    """

    def __init__(self, name: str, *args, **kwargs):
        self.name: str = name
        self.provider: str | None = None

    @abstractmethod
    def compute(self, *args, **kwargs) -> MetricResult:
        """
        Synchronous evaluation method.
        
        All metrics MUST implement this method.

        Returns:
            MetricResult: The result of the metric computation
        """
        pass

    async def acompute(self, *args, **kwargs) -> MetricResult:
        """
        Asynchronous evaluation method (optional).
        
        Default implementation: Runs compute() in thread pool executor.
        Subclasses can override this for true async evaluation (e.g., async LLM calls).

        Returns:
            MetricResult: The result of the metric computation
        """
        import asyncio
        
        # Check if subclass override this method
        if type(self).acompute is not BaseMetric.acompute:
            # Subclass implemented custom async, call it directly
            return await type(self).acompute(self, *args, **kwargs)
        
        # Default: Run sync compute() in thread pool executor
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.compute, *args, **kwargs)
        return result

    def evaluate(self, *args, **kwargs) -> MetricResult:
        """
        Public synchronous evaluation method.
        
        Calls compute() by default. Subclasses typically don't override this.
        """
        return self.compute(*args, **kwargs)

    async def aevaluate(self, *args, **kwargs) -> MetricResult:
        """
        Public asynchronous evaluation method.
        
        Calls acompute() by default. Subclasses typically don't override this.
        """
        return await self.acompute(*args, **kwargs)
