"""
Built-in/core metrics: ExactMatch, SemanticSim, etc.
"""

from floeval.api.metrics.base import BaseMetric, MetricResult


class ExactMatch(BaseMetric):
    """
    Exact match metric for comparing strings exactly.
    """

    def __init__(self):
        super().__init__(name="exact_match")

    def compute(self, *args, **kwargs) -> MetricResult:
        """
        Compute exact match score.
        """
        pass


class SemanticSim(BaseMetric):
    """
    Semantic similarity metric for comparing semantic meaning.
    """

    def __init__(self):
        super().__init__(name="semantic_sim")

    def compute(self, *args, **kwargs) -> MetricResult:
        """
        Compute semantic similarity score.
        """
        pass
