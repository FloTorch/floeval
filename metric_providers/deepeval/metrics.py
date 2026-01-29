"""
DeepEval metric implementations
"""

from ...api.metrics.base import BaseMetric, MetricResult


class DeepEvalMetric(BaseMetric):
    """
    Base class for DeepEval metrics.
    """
    
    def __init__(self, name: str):
        super().__init__(name=name)
    
    def compute(self, *args, **kwargs) -> MetricResult:
        """
        Compute DeepEval metric score.
        """
        pass

