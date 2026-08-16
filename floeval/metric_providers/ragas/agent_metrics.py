"""RAGAS agent evaluation metrics.

Follows the same pattern as RAGASAnswerRelevancy/RAGASFaithfulness:
- Use RAGASAdapter for LLM (adapter.agent_llm for agent metrics)
- Transform sample to RAGAS format
- Sync evaluate() uses notebook-safe coroutine runner; async aevaluate() awaits natively
- Return MetricResult
"""

import logging
from typing import Any

from ragas.messages import AIMessage as RAGASAIMessage, ToolCall as RAGASToolCall
from ragas.metrics.collections import TopicAdherence

from floeval.api.metrics.base import BaseMetric, MetricResult
from floeval.config.schemas.io.agent_dataset import (
    AgentSample,
    ToolCall as AgentToolCall,
    _to_display_str,
)
from floeval.config.schemas.io.conversation import ToolCallPayload
from floeval.config.schemas.io.conversational_dataset import ConversationalSample
from floeval.config.schemas.io.llm import LLMProviderConfig
from floeval.metric_providers.ragas.adapter import (
    RAGASAdapter,
    transform_sample_to_ragas_messages,
)
from floeval.utils.asyncio_compat import run_coroutine_sync

logger = logging.getLogger(__name__)


def _to_ragas_tool_call(
    tool: AgentToolCall | ToolCallPayload | dict[str, Any],
) -> RAGASToolCall:
    """Normalize a tool-call payload from either sample type to RAGAS ToolCall."""
    if isinstance(tool, dict):
        tool = ToolCallPayload.model_validate(tool)
    return RAGASToolCall(name=tool.name, args=tool.args)


class RAGASAgentGoalAccuracy(BaseMetric):
    """RAGAS AgentGoalAccuracyWithReference - same pattern as RAGASAnswerRelevancy."""

    ragas_sample_kind = "multi_turn"

    def __init__(
        self,
        llm_config: LLMProviderConfig | None = None,
        adapter: RAGASAdapter | None = None,
        **kwargs: Any,
    ):
        super().__init__(name="agent_goal_accuracy", **kwargs)
        self.provider = "ragas"
        self.llm_config = llm_config
        self._extra_headers: dict[str, str] = dict(kwargs.get("extra_headers") or {})
        self.adapter = adapter or RAGASAdapter(
            config=llm_config, extra_headers=self._extra_headers or None
        )
        from ragas.metrics.collections import AgentGoalAccuracyWithReference

        # Use the standard RAGAS LLM wrapper for compatibility with
        # AgentGoalAccuracyWithReference across ragas versions.
        self._metric = AgentGoalAccuracyWithReference(llm=self.adapter.agent_llm)

    @property
    def ragas_multiturn_metric(self):
        """Native RAGAS metric instance for batch evaluation routing."""
        return self._metric

    def evaluate(
        self, sample: AgentSample | ConversationalSample, **kwargs: Any
    ) -> MetricResult:
        try:
            messages = transform_sample_to_ragas_messages(sample)
            if not messages:
                return MetricResult(
                    score=None,
                    metadata={"error": "No messages in trace", "provider": "ragas"},
                )
            # Ensure final response content is explicitly present in the conversation.
            if isinstance(sample, AgentSample):
                final = sample.trace.final_response
                if final is not None and str(final).strip():
                    messages = list(messages) + [
                        RAGASAIMessage(
                            content=f"[Agent's final response to user: {final}]"
                        )
                    ]
            reference = _to_display_str(sample.reference_outcome)
            result = run_coroutine_sync(
                lambda: self._metric.ascore(
                    user_input=messages,
                    reference=reference,
                )
            )
            return MetricResult(
                score=float(result.value),
                metadata={"provider": "ragas", "metric_name": "agent_goal_accuracy"},
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("RAGASAgentGoalAccuracy failed: %s", e)
            return MetricResult(
                score=None,
                metadata={"error": str(e), "provider": "ragas"},
            )

    async def aevaluate(
        self, sample: AgentSample | ConversationalSample, **kwargs: Any
    ) -> MetricResult:
        """Evaluate agent goal accuracy asynchronously."""
        try:
            messages = transform_sample_to_ragas_messages(sample)
            if not messages:
                return MetricResult(
                    score=None,
                    metadata={"error": "No messages in trace", "provider": "ragas"},
                )
            if isinstance(sample, AgentSample):
                final = sample.trace.final_response
                if final is not None and str(final).strip():
                    messages = list(messages) + [
                        RAGASAIMessage(content=f"[Agent's final response to user: {final}]")
                    ]
            reference = _to_display_str(sample.reference_outcome)
            result = await self._metric.ascore(user_input=messages, reference=reference)
            return MetricResult(
                score=float(result.value),
                metadata={"provider": "ragas", "metric_name": "agent_goal_accuracy"},
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("RAGASAgentGoalAccuracy async failed: %s", e)
            return MetricResult(
                score=None,
                metadata={"error": str(e), "provider": "ragas"},
            )


class RAGASToolCallAccuracy(BaseMetric):
    """RAGAS ToolCallAccuracy - same pattern as RAGASAnswerRelevancy."""

    ragas_sample_kind = "multi_turn"

    def __init__(
        self,
        llm_config: LLMProviderConfig | None = None,
        adapter: RAGASAdapter | None = None,
        **kwargs: Any,
    ):
        super().__init__(name="tool_call_accuracy", **kwargs)
        self.provider = "ragas"
        self.llm_config = llm_config
        self._extra_headers: dict[str, str] = dict(kwargs.get("extra_headers") or {})
        self.adapter = adapter or RAGASAdapter(
            config=llm_config, extra_headers=self._extra_headers or None
        )
        from ragas.metrics.collections import ToolCallAccuracy

        self._metric = ToolCallAccuracy()

    @property
    def ragas_multiturn_metric(self):
        """Native RAGAS metric instance for batch evaluation routing."""
        return self._metric

    def evaluate(
        self, sample: AgentSample | ConversationalSample, **kwargs: Any
    ) -> MetricResult:
        if sample.reference_tool_calls is None:
            return MetricResult(
                score=None,
                metadata={"error": "reference_tool_calls required", "provider": "ragas"},
            )
        try:
            messages = transform_sample_to_ragas_messages(sample)
            if not messages:
                return MetricResult(
                    score=None,
                    metadata={"error": "No messages in trace", "provider": "ragas"},
                )
            ref_calls = [_to_ragas_tool_call(tc) for tc in sample.reference_tool_calls]
            result = run_coroutine_sync(
                lambda: self._metric.ascore(
                    user_input=messages,
                    reference_tool_calls=ref_calls,
                )
            )
            return MetricResult(
                score=float(result.value),
                metadata={"provider": "ragas", "metric_name": "tool_call_accuracy"},
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("RAGASToolCallAccuracy failed: %s", e)
            return MetricResult(
                score=None,
                metadata={"error": str(e), "provider": "ragas"},
            )

    async def aevaluate(
        self, sample: AgentSample | ConversationalSample, **kwargs: Any
    ) -> MetricResult:
        """Evaluate tool call accuracy asynchronously."""
        if sample.reference_tool_calls is None:
            return MetricResult(
                score=None,
                metadata={"error": "reference_tool_calls required", "provider": "ragas"},
            )
        try:
            messages = transform_sample_to_ragas_messages(sample)
            if not messages:
                return MetricResult(
                    score=None,
                    metadata={"error": "No messages in trace", "provider": "ragas"},
                )
            ref_calls = [_to_ragas_tool_call(tc) for tc in sample.reference_tool_calls]
            result = await self._metric.ascore(user_input=messages, reference_tool_calls=ref_calls)
            return MetricResult(
                score=float(result.value),
                metadata={"provider": "ragas", "metric_name": "tool_call_accuracy"},
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("RAGASToolCallAccuracy async failed: %s", e)
            return MetricResult(
                score=None,
                metadata={"error": str(e), "provider": "ragas"},
            )


class RAGASTopicAdherence(BaseMetric):
    """RAGAS TopicAdherence — evaluates whether agent stays on assigned topic.

    Evaluates topic adherence across a multi-turn conversation / trace.
    Registered as: ragas:topic_adherence

    Requires: trace with multiple turns.
    """

    ragas_sample_kind = "multi_turn"

    def __init__(
        self,
        llm_config: LLMProviderConfig | None = None,
        adapter: RAGASAdapter | None = None,
        **kwargs: Any,
    ):
        super().__init__(name="topic_adherence", **kwargs)
        self.provider = "ragas"
        self.llm_config = llm_config
        self._extra_headers: dict[str, str] = dict(kwargs.get("extra_headers") or {})
        self.adapter = adapter or RAGASAdapter(
            config=llm_config, extra_headers=self._extra_headers or None
        )
        self._metric = TopicAdherence(llm=self.adapter.agent_llm)

    @property
    def ragas_multiturn_metric(self):
        """Native RAGAS metric instance for batch evaluation routing."""
        return self._metric

    def evaluate(
        self, sample: AgentSample | ConversationalSample, **kwargs: Any
    ) -> MetricResult:
        try:
            messages = transform_sample_to_ragas_messages(sample)
            if not messages:
                return MetricResult(
                    score=None,
                    metadata={"error": "No messages in trace", "provider": "ragas"},
                )
            reference_topics: list[str] = []
            if isinstance(sample, ConversationalSample):
                reference_topics = sample.reference_topics or []
            result = run_coroutine_sync(
                lambda: self._metric.ascore(
                    user_input=messages, reference_topics=reference_topics
                )
            )
            return MetricResult(
                score=float(result.value),
                metadata={"provider": "ragas", "metric_name": "topic_adherence"},
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("RAGASTopicAdherence failed: %s", e)
            return MetricResult(
                score=None,
                metadata={"error": str(e), "provider": "ragas"},
            )

    async def aevaluate(
        self, sample: AgentSample | ConversationalSample, **kwargs: Any
    ) -> MetricResult:
        try:
            messages = transform_sample_to_ragas_messages(sample)
            if not messages:
                return MetricResult(
                    score=None,
                    metadata={"error": "No messages in trace", "provider": "ragas"},
                )
            reference_topics: list[str] = []
            if isinstance(sample, ConversationalSample):
                reference_topics = sample.reference_topics or []
            result = await self._metric.ascore(
                user_input=messages, reference_topics=reference_topics
            )
            return MetricResult(
                score=float(result.value),
                metadata={"provider": "ragas", "metric_name": "topic_adherence"},
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("RAGASTopicAdherence async failed: %s", e)
            return MetricResult(
                score=None,
                metadata={"error": str(e), "provider": "ragas"},
            )


