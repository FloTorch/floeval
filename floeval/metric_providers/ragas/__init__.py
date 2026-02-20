"""
RAGAS provider module.

This module registers RAGAS metrics (answer_relevancy, faithfulness,
agent_goal_accuracy, tool_call_accuracy) with the global metric registry.
"""

from floeval.api.metrics.registry import MetricRegistry
from floeval.metric_providers.ragas.adapter import RAGASAdapter
from floeval.metric_providers.ragas.agent_metrics import (
    RAGASAgentGoalAccuracy,
    RAGASToolCallAccuracy,
)
from floeval.metric_providers.ragas.metrics import (
    RAGASAnswerRelevancy,
    RAGASFaithfulness,
)

# Create a module-level registry instance for registration
# This will be used to register metrics when the module is imported
_registry = MetricRegistry()

# Register RAGAS metrics
_registry.register("ragas", "answer_relevancy", RAGASAnswerRelevancy)
_registry.register("ragas", "faithfulness", RAGASFaithfulness)
_registry.register("ragas", "agent_goal_accuracy", RAGASAgentGoalAccuracy)
_registry.register("ragas", "tool_call_accuracy", RAGASToolCallAccuracy)

__all__ = [
    "RAGASAnswerRelevancy",
    "RAGASFaithfulness",
    "RAGASAgentGoalAccuracy",
    "RAGASToolCallAccuracy",
    "RAGASAdapter",
]
