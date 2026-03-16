"""DeepEval provider module.

This module registers DeepEval metrics (answer_relevancy, faithfulness,
contextual_precision, contextual_recall, contextual_relevancy, hallucination,
toxicity, exact_match, pattern_match, json_correctness) with the global
metric registry.
"""

from floeval.api.metrics.registry import MetricRegistry
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
]
