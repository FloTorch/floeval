"""DeepEval multi-turn conversational metric implementations."""

from typing import Any, ClassVar, Literal, cast

from deepeval.evaluate import evaluate
from deepeval.metrics import (
    BaseConversationalMetric,
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
from deepeval.test_case import ToolCall as DeepEvalToolCall

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
    sample_param_aliases: ClassVar[dict[str, tuple[str, ...]]] = {
        "available_tools": ("available_tools", "reference_tool_calls"),
        "relevant_topics": ("relevant_topics", "reference_topics"),
    }

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

    @staticmethod
    def _to_deepeval_tool_call(tool: Any) -> DeepEvalToolCall:
        if isinstance(tool, DeepEvalToolCall):
            return tool

        if hasattr(tool, "name"):
            return DeepEvalToolCall(
                name=str(tool.name),
                input_parameters=getattr(tool, "args", {}) or {},
                output=getattr(tool, "output", None),
            )

        if isinstance(tool, dict):
            name = tool.get("name")
            if not name:
                raise ValueError("Tool call is missing required field: 'name'.")
            args = (
                tool.get("args")
                or tool.get("input")
                or tool.get("input_parameters")
                or {}
            )
            return DeepEvalToolCall(
                name=str(name),
                input_parameters=args,
                output=tool.get("output"),
            )

        raise TypeError(
            "Unsupported tool call format for available_tools. "
            "Expected DeepEval ToolCall, object with name/args, or dict."
        )

    def _normalize_required_metric_param(self, param: str, value: Any) -> Any:
        if param == "available_tools":
            return [self._to_deepeval_tool_call(tool) for tool in value]
        if param == "relevant_topics":
            return list(value)
        return value

    def _resolve_metric_param_from_sample(
        self, sample: ConversationalSample | AgentSample, param: str
    ) -> Any | None:
        aliases = self.sample_param_aliases.get(param, (param,))
        for alias in aliases:
            value = getattr(sample, alias, None)
            if value:
                return value
        return None

    def _metric_kwargs_for_sample(
        self, sample: ConversationalSample | AgentSample
    ) -> dict[str, Any]:
        return dict(self._llm_metric_kwargs())

    def _metric_kwargs_with_required(
        self, sample: ConversationalSample | AgentSample, required_params: tuple[str, ...]
    ) -> dict[str, Any]:
        metric_kwargs = dict(self._llm_metric_kwargs())
        for param in required_params:
            value = metric_kwargs.get(param)
            if value:
                metric_kwargs[param] = self._normalize_required_metric_param(param, value)
                continue

            sample_value = self._resolve_metric_param_from_sample(sample, param)
            if sample_value:
                metric_kwargs[param] = self._normalize_required_metric_param(
                    param, sample_value
                )
                continue

            aliases = ", ".join(self.sample_param_aliases.get(param, (param,)))
            raise ValueError(
                f"deepeval:{self.name} requires `{param}`. "
                f"Provide metric param `{param}` or sample field alias ({aliases})."
            )
        return metric_kwargs

    def create_deepeval_metric_instance(
        self, sample: ConversationalSample | AgentSample, **kwargs: Any
    ) -> Any:
        raise NotImplementedError

    def evaluate(
        self, sample: ConversationalSample | AgentSample, **kwargs: Any
    ) -> MetricResult:
        """Score a single conversational sample using DeepEval evaluate."""
        inst = self.create_deepeval_metric_instance(sample)
        tc = conversational_row_to_deepeval(sample)
        # ``list`` is invariant: concrete metric types need an explicit cast for ``evaluate``.
        metrics = cast(list[BaseConversationalMetric], [inst])
        agg = evaluate(metrics=metrics, test_cases=[tc])
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

    def create_deepeval_metric_instance(
        self, sample: ConversationalSample | AgentSample, **kwargs: Any
    ) -> TurnRelevancyMetric:
        return TurnRelevancyMetric(**self._metric_kwargs_for_sample(sample))


class RoleAdherenceDeepEvalMetric(_ConversationalDeepEvalMetric):
    metric_name = "role_adherence"

    def create_deepeval_metric_instance(
        self, sample: ConversationalSample | AgentSample, **kwargs: Any
    ) -> RoleAdherenceMetric:
        return RoleAdherenceMetric(**self._metric_kwargs_for_sample(sample))


class KnowledgeRetentionDeepEvalMetric(_ConversationalDeepEvalMetric):
    metric_name = "knowledge_retention"

    def create_deepeval_metric_instance(
        self, sample: ConversationalSample | AgentSample, **kwargs: Any
    ) -> KnowledgeRetentionMetric:
        return KnowledgeRetentionMetric(**self._metric_kwargs_for_sample(sample))


class ConversationCompletenessDeepEvalMetric(_ConversationalDeepEvalMetric):
    metric_name = "conversation_completeness"

    def create_deepeval_metric_instance(
        self, sample: ConversationalSample | AgentSample, **kwargs: Any
    ) -> ConversationCompletenessMetric:
        return ConversationCompletenessMetric(**self._metric_kwargs_for_sample(sample))


class GoalAccuracyDeepEvalMetric(_ConversationalDeepEvalMetric):
    metric_name = "goal_accuracy"

    def create_deepeval_metric_instance(
        self, sample: ConversationalSample | AgentSample, **kwargs: Any
    ) -> GoalAccuracyMetric:
        return GoalAccuracyMetric(**self._metric_kwargs_for_sample(sample))


class ToolUseDeepEvalMetric(_ConversationalDeepEvalMetric):
    metric_name = "tool_use"

    def create_deepeval_metric_instance(
        self, sample: ConversationalSample | AgentSample, **kwargs: Any
    ) -> ToolUseMetric:
        metric_kwargs = self._metric_kwargs_with_required(
            sample, required_params=("available_tools",)
        )
        return ToolUseMetric(**metric_kwargs)


class TopicAdherenceDeepEvalMetric(_ConversationalDeepEvalMetric):
    metric_name = "topic_adherence"

    def create_deepeval_metric_instance(
        self, sample: ConversationalSample | AgentSample, **kwargs: Any
    ) -> TopicAdherenceMetric:
        metric_kwargs = self._metric_kwargs_with_required(
            sample, required_params=("relevant_topics",)
        )
        return TopicAdherenceMetric(**metric_kwargs)


class TurnFaithfulnessDeepEvalMetric(_ConversationalDeepEvalMetric):
    metric_name = "turn_faithfulness"

    def create_deepeval_metric_instance(
        self, sample: ConversationalSample | AgentSample, **kwargs: Any
    ) -> TurnFaithfulnessMetric:
        return TurnFaithfulnessMetric(**self._metric_kwargs_for_sample(sample))


class TurnContextualPrecisionDeepEvalMetric(_ConversationalDeepEvalMetric):
    metric_name = "turn_contextual_precision"

    def create_deepeval_metric_instance(
        self, sample: ConversationalSample | AgentSample, **kwargs: Any
    ) -> TurnContextualPrecisionMetric:
        return TurnContextualPrecisionMetric(**self._metric_kwargs_for_sample(sample))


class TurnContextualRecallDeepEvalMetric(_ConversationalDeepEvalMetric):
    metric_name = "turn_contextual_recall"

    def create_deepeval_metric_instance(
        self, sample: ConversationalSample | AgentSample, **kwargs: Any
    ) -> TurnContextualRecallMetric:
        return TurnContextualRecallMetric(**self._metric_kwargs_for_sample(sample))


class TurnContextualRelevancyDeepEvalMetric(_ConversationalDeepEvalMetric):
    metric_name = "turn_contextual_relevancy"

    def create_deepeval_metric_instance(
        self, sample: ConversationalSample | AgentSample, **kwargs: Any
    ) -> TurnContextualRelevancyMetric:
        return TurnContextualRelevancyMetric(**self._metric_kwargs_for_sample(sample))
