"""
RAGAS provider module.

This module registers RAGAS metrics (answer_relevancy, faithfulness)
with the global metric registry.
"""

from floeval.metric_providers.ragas.metrics import RAGASAnswerRelevancy, RAGASFaithfulness
from floeval.metric_providers.ragas.adapter import RAGASAdapter
from floeval.api.metrics.registry import MetricRegistry

# Create a module-level registry instance for registration
# This will be used to register metrics when the module is imported
_registry = MetricRegistry()

# Register RAGAS metrics
_registry.register("ragas", "answer_relevancy", RAGASAnswerRelevancy)
_registry.register("ragas", "faithfulness", RAGASFaithfulness)

__all__ = [
    "RAGASAnswerRelevancy",
    "RAGASFaithfulness",
    "RAGASAdapter",
]
