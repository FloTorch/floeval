"""
Base metric abstract class and result model
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class MetricResult:
    """
    Result container for metric evaluation.
    """
    
    def __init__(self, score: float, metadata: Dict[str, Any] = None):
        self.score = score
        self.metadata = metadata or {}


class BaseMetric(ABC):
    """
    Abstract base class for all metrics.
    """
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def compute(self, *args, **kwargs) -> MetricResult:
        """
        Compute the metric score.
        
        Returns:
            MetricResult: The result of the metric computation
        """
        pass

    def evaluate(self, *args, **kwargs) -> MetricResult:
        return self.compute(*args, **kwargs)

