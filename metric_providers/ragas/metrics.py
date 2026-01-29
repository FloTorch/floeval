"""
RAGAS metric implementations
"""

from ...api.metrics.base import BaseMetric, MetricResult


class RAGASMetric(BaseMetric):
    """
    Base class for RAGAS metrics.
    """
    
    def __init__(self, name: str):
        super().__init__(name=name)
    
    def compute(self, *args, **kwargs) -> MetricResult:
        """
        Compute RAGAS metric score.
        """
        pass

