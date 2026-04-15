"""DeepEval multi-turn conversational metric implementations."""

from typing import Any, ClassVar, Literal

from deepeval.evaluate import evaluate
from deepeval.metrics import (
    ConversationCompletenessMetric,
    GoalAccuracyMetric,
    KnowledgeRetentionMetric,
    RoleAdherenceMetric,
    ToolUseMetric,
    TopicAdherenceMetric,
    TurnContextualPrecisionMetric,
    TurnContextualRecallMetric,
    TurnContextualRelevancyMetric,
    TurnFaithfulnessMetric,
    TurnRelevancyMetric,
)

from floeval.api.metrics.base import MetricResult
from floeval.config.schemas.io.agent_dataset import AgentSample
from floeval.config.schemas.io.conversational_dataset import ConversationalSample
from floeval.metric_providers.deepeval.metrics import DeepEvalMetric
from floeval.metric_providers.deepeval.multiturn_adapter import (
    conversational_row_to_deepeval,
)


class _ConversationalDeepEvalMetric(DeepEvalMetric):
    """Shared implementation for conversational DeepEval metrics."""

    execute_via: ClassVar[str] = "deepeval"
    deepeval_test_case_kind: ClassVar[Literal["conversational"]] = "conversational"
    metric_name: ClassVar[str]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, name=self.metric_name, **kwargs)

    def _extract_metric_params(
        self, params: dict[str, Any], kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        metric_params = super()._extract_metric_params(params, kwargs)
        for key in ("window_size", "relevant_topics", "available_tools"):
            if key in kwargs and key not in metric_params:
                metric_params[key] = kwargs[key]
        return metric_params

    def _llm_metric_kwargs(self) -> dict[str, Any]:
        if self._llm_adapter is None:
            raise ValueError(
                f"llm_config is required for {self.name}. Pass llm_config into Evaluation()."
            )
        return self._metric_params | {"model": self._llm_adapter}

    def create_deepeval_metric_instance(self) -> Any:
        raise NotImplementedError

    def evaluate(
        self, sample: ConversationalSample | AgentSample, **kwargs: Any
    ) -> MetricResult:
        """Score a single conversational sample using DeepEval evaluate()."""
        inst = self.create_deepeval_metric_instance()
        tc = conversational_row_to_deepeval(sample)
        agg = evaluate(metrics=[inst], test_cases=[tc])
        if not agg.test_results:
            return MetricResult(
                score=None,
                metadata={
                    "passed": False,
                    "provider": "deepeval",
                    "metric_name": self.name,
                    "error": "DeepEval returned no test_results.",
                },
            )
        return self._extract_metric_result(agg.test_results[0], self.name)


class TurnRelevancyDeepEvalMetric(_ConversationalDeepEvalMetric):
    metric_name = "turn_relevancy"

    def create_deepeval_metric_instance(self) -> TurnRelevancyMetric:
        return TurnRelevancyMetric(**self._llm_metric_kwargs())


class RoleAdherenceDeepEvalMetric(_ConversationalDeepEvalMetric):
    metric_name = "role_adherence"

    def create_deepeval_metric_instance(self) -> RoleAdherenceMetric:
        return RoleAdherenceMetric(**self._llm_metric_kwargs())


class KnowledgeRetentionDeepEvalMetric(_ConversationalDeepEvalMetric):
    metric_name = "knowledge_retention"

    def create_deepeval_metric_instance(self) -> KnowledgeRetentionMetric:
        return KnowledgeRetentionMetric(**self._llm_metric_kwargs())


class ConversationCompletenessDeepEvalMetric(_ConversationalDeepEvalMetric):
    metric_name = "conversation_completeness"

    def create_deepeval_metric_instance(self) -> ConversationCompletenessMetric:
        return ConversationCompletenessMetric(**self._llm_metric_kwargs())


class GoalAccuracyDeepEvalMetric(_ConversationalDeepEvalMetric):
    metric_name = "goal_accuracy"

    def create_deepeval_metric_instance(self) -> GoalAccuracyMetric:
        return GoalAccuracyMetric(**self._llm_metric_kwargs())


class ToolUseDeepEvalMetric(_ConversationalDeepEvalMetric):
    metric_name = "tool_use"

    def create_deepeval_metric_instance(self) -> ToolUseMetric:
        metric_kwargs = self._llm_metric_kwargs()
        metric_kwargs.setdefault("available_tools", [])
        return ToolUseMetric(**metric_kwargs)


class TopicAdherenceDeepEvalMetric(_ConversationalDeepEvalMetric):
    metric_name = "topic_adherence"

    def create_deepeval_metric_instance(self) -> TopicAdherenceMetric:
        metric_kwargs = self._llm_metric_kwargs()
        metric_kwargs.setdefault("relevant_topics", [])
        return TopicAdherenceMetric(**metric_kwargs)


class TurnFaithfulnessDeepEvalMetric(_ConversationalDeepEvalMetric):
    metric_name = "turn_faithfulness"

    def create_deepeval_metric_instance(self) -> TurnFaithfulnessMetric:
        return TurnFaithfulnessMetric(**self._llm_metric_kwargs())


class TurnContextualPrecisionDeepEvalMetric(_ConversationalDeepEvalMetric):
    metric_name = "turn_contextual_precision"

    def create_deepeval_metric_instance(self) -> TurnContextualPrecisionMetric:
        return TurnContextualPrecisionMetric(**self._llm_metric_kwargs())


class TurnContextualRecallDeepEvalMetric(_ConversationalDeepEvalMetric):
    metric_name = "turn_contextual_recall"

    def create_deepeval_metric_instance(self) -> TurnContextualRecallMetric:
        return TurnContextualRecallMetric(**self._llm_metric_kwargs())


class TurnContextualRelevancyDeepEvalMetric(_ConversationalDeepEvalMetric):
    metric_name = "turn_contextual_relevancy"

    def create_deepeval_metric_instance(self) -> TurnContextualRelevancyMetric:
        return TurnContextualRelevancyMetric(**self._llm_metric_kwargs())
