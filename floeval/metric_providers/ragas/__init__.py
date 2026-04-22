"""RAGAS provider module.

This module registers RAGAS metrics:
    - answer_relevancy
    - faithfulness
    - aspect_critic
    - context_precision
    - context_recall
    - context_entity_recall
    - noise_sensitivity
    - agent_goal_accuracy
    - tool_call_accuracy
with the global metric registry.
"""

from floeval.api.metrics.registry import MetricRegistry
from floeval.metric_providers.ragas.adapter import RAGASAdapter
from floeval.metric_providers.ragas.agent_metrics import (
    RAGASAgentGoalAccuracy,
    RAGASToolCallAccuracy,
    RAGASTopicAdherence,
)
from floeval.metric_providers.ragas.metrics import (
    RAGASAnswerRelevancy,
    RAGASAspectCritic,
    RAGASContextEntityRecall,
    RAGASContextPrecision,
    RAGASContextRecall,
    RAGASFaithfulness,
    RAGASMultiTurnTopicAdherence,
    RAGASNoiseSensitivity,
)

# Create a module-level registry instance for registration
# This will be used to register metrics when the module is imported
_registry = MetricRegistry()

# Register RAGAS metrics
_registry.register("ragas", "answer_relevancy", RAGASAnswerRelevancy)
_registry.register("ragas", "faithfulness", RAGASFaithfulness)
_registry.register("ragas", "context_precision", RAGASContextPrecision)
_registry.register("ragas", "context_recall", RAGASContextRecall)
_registry.register("ragas", "context_entity_recall", RAGASContextEntityRecall)
_registry.register("ragas", "noise_sensitivity", RAGASNoiseSensitivity)
_registry.register("ragas", "aspect_critic", RAGASAspectCritic)
_registry.register("ragas", "agent_goal_accuracy", RAGASAgentGoalAccuracy)
_registry.register("ragas", "tool_call_accuracy", RAGASToolCallAccuracy)
_registry.register("ragas", "topic_adherence", RAGASTopicAdherence)
# TODO(dev): merge multi_turn_topic_adherence into topic_adherence when AgentSample
# schema stabilizes with reference_topics support.
_registry.register("ragas", "multi_turn_topic_adherence", RAGASMultiTurnTopicAdherence)

__all__ = [
    "RAGASAnswerRelevancy",
    "RAGASFaithfulness",
    "RAGASContextPrecision",
    "RAGASContextRecall",
    "RAGASContextEntityRecall",
    "RAGASNoiseSensitivity",
    "RAGASAspectCritic",
    "RAGASAgentGoalAccuracy",
    "RAGASToolCallAccuracy",
    "RAGASTopicAdherence",
    "RAGASMultiTurnTopicAdherence",
    "RAGASAdapter",
]
