"""
DeepEval provider module

This module registers DeepEval metrics (answer_relevancy, faithfulness)
with the global metric registry.
"""

from floeval.api.metrics.registry import MetricRegistry
from floeval.metric_providers.deepeval.metrics import FaithfulnessDeepEvalMetric

# Create a module-level registry instance for registration
# This will be used to register metrics when the module is imported
_registry = MetricRegistry()
_registry.register("deepeval", "faithfulness", FaithfulnessDeepEvalMetric)

__all__ = [
    "FaithfulnessDeepEvalMetric"
]
