"""DeepEval provider module.

This module registers DeepEval metrics (answer_relevancy, faithfulness,
contextual_precision, contextual_recall, contextual_relevancy, hallucination,
toxicity, exact_match, pattern_match, json_correctness, conversational_g_eval)
with the global metric registry.
"""

from floeval.api.metrics.registry import MetricRegistry
from floeval.metric_providers.deepeval.conversational_multiturn_metrics import (
    ConversationCompletenessDeepEvalMetric,
    GoalAccuracyDeepEvalMetric,
    KnowledgeRetentionDeepEvalMetric,
    RoleAdherenceDeepEvalMetric,
    ToolUseDeepEvalMetric,
    TopicAdherenceDeepEvalMetric,
    TurnContextualPrecisionDeepEvalMetric,
    TurnContextualRecallDeepEvalMetric,
    TurnContextualRelevancyDeepEvalMetric,
    TurnFaithfulnessDeepEvalMetric,
    TurnRelevancyDeepEvalMetric,
)
from floeval.metric_providers.deepeval.conversational_metrics import (
    ConversationalGEvalDeepEvalMetric,
)
from floeval.metric_providers.deepeval.metrics import (
    AnswerRelevancyDeepEvalMetric,
    ContextualPrecisionDeepEvalMetric,
    ContextualRecallDeepEvalMetric,
    ContextualRelevancyDeepEvalMetric,
    ExactMatchDeepEvalMetric,
    FaithfulnessDeepEvalMetric,
    HallucinationDeepEvalMetric,
    JsonCorrectnessDeepEvalMetric,
    PatternMatchDeepEvalMetric,
    ToxicityDeepEvalMetric,
)

# Create a module-level registry instance for registration
# This will be used to register metrics when the module is imported
_registry = MetricRegistry()
_registry.register("deepeval", "faithfulness", FaithfulnessDeepEvalMetric)
_registry.register("deepeval", "answer_relevancy", AnswerRelevancyDeepEvalMetric)
_registry.register(
    "deepeval", "contextual_precision", ContextualPrecisionDeepEvalMetric
)
_registry.register("deepeval", "contextual_recall", ContextualRecallDeepEvalMetric)
_registry.register(
    "deepeval", "contextual_relevancy", ContextualRelevancyDeepEvalMetric
)
_registry.register("deepeval", "hallucination", HallucinationDeepEvalMetric)
_registry.register("deepeval", "toxicity", ToxicityDeepEvalMetric)
_registry.register("deepeval", "exact_match", ExactMatchDeepEvalMetric)
_registry.register("deepeval", "pattern_match", PatternMatchDeepEvalMetric)
_registry.register("deepeval", "json_correctness", JsonCorrectnessDeepEvalMetric)
_registry.register("deepeval", "conversational_g_eval", ConversationalGEvalDeepEvalMetric)
_registry.register("deepeval", "turn_relevancy", TurnRelevancyDeepEvalMetric)
_registry.register("deepeval", "role_adherence", RoleAdherenceDeepEvalMetric)
_registry.register("deepeval", "knowledge_retention", KnowledgeRetentionDeepEvalMetric)
_registry.register(
    "deepeval", "conversation_completeness", ConversationCompletenessDeepEvalMetric
)
_registry.register("deepeval", "goal_accuracy", GoalAccuracyDeepEvalMetric)
_registry.register("deepeval", "tool_use", ToolUseDeepEvalMetric)
_registry.register("deepeval", "topic_adherence", TopicAdherenceDeepEvalMetric)
_registry.register("deepeval", "turn_faithfulness", TurnFaithfulnessDeepEvalMetric)
_registry.register(
    "deepeval", "turn_contextual_precision", TurnContextualPrecisionDeepEvalMetric
)
_registry.register("deepeval", "turn_contextual_recall", TurnContextualRecallDeepEvalMetric)
_registry.register(
    "deepeval", "turn_contextual_relevancy", TurnContextualRelevancyDeepEvalMetric
)

__all__ = [
    "FaithfulnessDeepEvalMetric",
    "AnswerRelevancyDeepEvalMetric",
    "ContextualPrecisionDeepEvalMetric",
    "ContextualRecallDeepEvalMetric",
    "ContextualRelevancyDeepEvalMetric",
    "HallucinationDeepEvalMetric",
    "ToxicityDeepEvalMetric",
    "ExactMatchDeepEvalMetric",
    "PatternMatchDeepEvalMetric",
    "JsonCorrectnessDeepEvalMetric",
    "ConversationalGEvalDeepEvalMetric",
    "TurnRelevancyDeepEvalMetric",
    "RoleAdherenceDeepEvalMetric",
    "KnowledgeRetentionDeepEvalMetric",
    "ConversationCompletenessDeepEvalMetric",
    "GoalAccuracyDeepEvalMetric",
    "ToolUseDeepEvalMetric",
    "TopicAdherenceDeepEvalMetric",
    "TurnFaithfulnessDeepEvalMetric",
    "TurnContextualPrecisionDeepEvalMetric",
    "TurnContextualRecallDeepEvalMetric",
    "TurnContextualRelevancyDeepEvalMetric",
]
