"""RAGAS agent evaluation metrics.

Follows the same pattern as RAGASAnswerRelevancy/RAGASFaithfulness:
- Use RAGASAdapter for LLM (adapter.agent_llm for agent metrics)
- Transform sample to RAGAS format
- Sync evaluate() uses notebook-safe coroutine runner; async aevaluate() awaits natively
- Return MetricResult
"""

import logging
from typing import Any

from ragas.messages import AIMessage as RAGASAIMessage
from ragas.metrics.collections import TopicAdherence

from floeval.api.metrics.base import BaseMetric, MetricResult
from floeval.config.schemas.io.agent_dataset import AgentSample, _to_display_str
from floeval.config.schemas.io.llm import LLMProviderConfig
from floeval.metric_providers.ragas.adapter import RAGASAdapter, transform_agent_sample_to_ragas_messages
from floeval.utils.asyncio_compat import run_coroutine_sync

logger = logging.getLogger(__name__)


class RAGASAgentGoalAccuracy(BaseMetric):
    """RAGAS AgentGoalAccuracyWithReference - same pattern as RAGASAnswerRelevancy."""

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

    def evaluate(self, sample: AgentSample, **kwargs: Any) -> MetricResult:
        try:
            messages = transform_agent_sample_to_ragas_messages(sample)
            if not messages:
                return MetricResult(
                    score=None,
                    metadata={"error": "No messages in trace", "provider": "ragas"},
                )
            # Ensure final response content is explicitly present in the conversation.
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

    async def aevaluate(self, sample: AgentSample, **kwargs: Any) -> MetricResult:
        """Evaluate agent goal accuracy asynchronously."""
        try:
            messages = transform_agent_sample_to_ragas_messages(sample)
            if not messages:
                return MetricResult(
                    score=None,
                    metadata={"error": "No messages in trace", "provider": "ragas"},
                )
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

    def evaluate(self, sample: AgentSample, **kwargs: Any) -> MetricResult:
        if sample.reference_tool_calls is None:
            return MetricResult(
                score=None,
                metadata={"error": "reference_tool_calls required", "provider": "ragas"},
            )
        try:
            messages = transform_agent_sample_to_ragas_messages(sample)
            if not messages:
                return MetricResult(
                    score=None,
                    metadata={"error": "No messages in trace", "provider": "ragas"},
                )
            from ragas.messages import ToolCall as RAGASToolCall

            ref_calls = [
                RAGASToolCall(name=tc.name, args=tc.args) for tc in sample.reference_tool_calls
            ]
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

    async def aevaluate(self, sample: AgentSample, **kwargs: Any) -> MetricResult:
        """Evaluate tool call accuracy asynchronously."""
        if sample.reference_tool_calls is None:
            return MetricResult(
                score=None,
                metadata={"error": "reference_tool_calls required", "provider": "ragas"},
            )
        try:
            messages = transform_agent_sample_to_ragas_messages(sample)
            if not messages:
                return MetricResult(
                    score=None,
                    metadata={"error": "No messages in trace", "provider": "ragas"},
                )
            from ragas.messages import ToolCall as RAGASToolCall

            ref_calls = [
                RAGASToolCall(name=tc.name, args=tc.args) for tc in sample.reference_tool_calls
            ]
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

    def evaluate(self, sample: AgentSample, **kwargs: Any) -> MetricResult:
        try:
            messages = transform_agent_sample_to_ragas_messages(sample)
            if not messages:
                return MetricResult(
                    score=None,
                    metadata={"error": "No messages in trace", "provider": "ragas"},
                )
            result = run_coroutine_sync(lambda: self._metric.ascore(user_input=messages))
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

    async def aevaluate(self, sample: AgentSample, **kwargs: Any) -> MetricResult:
        try:
            messages = transform_agent_sample_to_ragas_messages(sample)
            if not messages:
                return MetricResult(
                    score=None,
                    metadata={"error": "No messages in trace", "provider": "ragas"},
                )
            result = await self._metric.ascore(user_input=messages)
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


