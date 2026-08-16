"""Builtin efficiency metrics — all deterministic, zero LLM cost.

Metrics registered:
- builtin:turn_count      (count of AI turns in trace)
- builtin:tool_call_count (count of tool result messages in trace)
- builtin:latency_seconds (wall-clock seconds; requires trace.start_time and trace.end_time)
- builtin:token_usage     (total tokens; requires trace.total_tokens)

Score is the raw value (not normalised to 0-1). Useful for dashboards and
SLO tracking. Efficiency metrics are exempt from the [0, 1] score constraint.
"""

import logging

from floeval.api.metrics.base import BaseMetric, MetricResult
from floeval.api.metrics.registry import MetricRegistry
from floeval.config.schemas.io.agent_dataset import AgentSample, ToolMessage

logger = logging.getLogger(__name__)


class TurnCountMetric(BaseMetric):
    """Number of AI turns in the trace. Score = raw count (float)."""

    workflow_aggregate = "sum"  # total turns across all agents in workflow

    def __init__(self, max_turns: int | None = None, **kwargs):
        super().__init__(name="turn_count", **kwargs)
        self.provider = "builtin"
        self.max_turns = max_turns

    def evaluate(self, sample: AgentSample, **kwargs) -> MetricResult:
        if not sample.trace:
            return MetricResult(
                score=None,
                metadata={"error": "trace required", "provider": "builtin"},
            )
        count = sample.trace.turn_count
        passed = (count <= self.max_turns) if self.max_turns is not None else None
        return MetricResult(
            score=float(count),
            metadata={
                "turn_count": count,
                "max_turns": self.max_turns,
                "passed": passed,
                "provider": "builtin",
                "metric_name": "turn_count",
            },
        )


class ToolCallCountMetric(BaseMetric):
    """Number of tool result messages in the trace. Score = raw count (float)."""

    workflow_aggregate = "sum"  # total tool calls across all agents in workflow

    def __init__(self, max_tool_calls: int | None = None, **kwargs):
        super().__init__(name="tool_call_count", **kwargs)
        self.provider = "builtin"
        self.max_tool_calls = max_tool_calls

    def evaluate(self, sample: AgentSample, **kwargs) -> MetricResult:
        if not sample.trace:
            return MetricResult(
                score=None,
                metadata={"error": "trace required", "provider": "builtin"},
            )
        count = sum(1 for m in sample.trace.messages if isinstance(m, ToolMessage))
        passed = (count <= self.max_tool_calls) if self.max_tool_calls is not None else None
        return MetricResult(
            score=float(count),
            metadata={
                "tool_call_count": count,
                "max_tool_calls": self.max_tool_calls,
                "passed": passed,
                "provider": "builtin",
                "metric_name": "tool_call_count",
            },
        )


class LatencyMetric(BaseMetric):
    """Wall-clock latency in seconds per agent. Summed across agents in workflow.

    Requires AgentTrace.start_time and AgentTrace.end_time.
    Returns score=None otherwise.
    """

    workflow_aggregate = "sum"  # total latency across agents in workflow

    def __init__(self, max_latency_seconds: float | None = None, **kwargs):
        super().__init__(name="latency_seconds", **kwargs)
        self.provider = "builtin"
        self.max_latency_seconds = max_latency_seconds

    def evaluate(self, sample: AgentSample, **kwargs) -> MetricResult:
        if not sample.trace:
            return MetricResult(
                score=None,
                metadata={"error": "trace required", "provider": "builtin"},
            )
        latency = sample.trace.latency_seconds
        if latency is None:
            return MetricResult(
                score=None,
                metadata={
                    "reason": "trace.start_time and trace.end_time required",
                    "provider": "builtin",
                },
            )
        passed = (
            (latency <= self.max_latency_seconds) if self.max_latency_seconds is not None else None
        )
        return MetricResult(
            score=latency,
            metadata={
                "latency_seconds": latency,
                "max_latency_seconds": self.max_latency_seconds,
                "passed": passed,
                "provider": "builtin",
                "metric_name": "latency_seconds",
            },
        )


class TokenUsageMetric(BaseMetric):
    """Total token consumption per agent. Summed across agents in workflow.

    Requires AgentTrace.total_tokens. Returns score=None otherwise.
    """

    workflow_aggregate = "sum"  # total tokens across all agents in workflow

    def __init__(self, max_tokens: int | None = None, **kwargs):
        super().__init__(name="token_usage", **kwargs)
        self.provider = "builtin"
        self.max_tokens = max_tokens

    def evaluate(self, sample: AgentSample, **kwargs) -> MetricResult:
        if not sample.trace:
            return MetricResult(
                score=None,
                metadata={"error": "trace required", "provider": "builtin"},
            )
        total = sample.trace.total_tokens
        if total is None:
            return MetricResult(
                score=None,
                metadata={
                    "reason": "trace.total_tokens required",
                    "provider": "builtin",
                },
            )
        passed = (total <= self.max_tokens) if self.max_tokens is not None else None
        return MetricResult(
            score=float(total),
            metadata={
                "total_tokens": total,
                "prompt_tokens": sample.trace.prompt_tokens,
                "completion_tokens": sample.trace.completion_tokens,
                "max_tokens": self.max_tokens,
                "passed": passed,
                "provider": "builtin",
                "metric_name": "token_usage",
            },
        )


MetricRegistry.register("builtin", "turn_count", TurnCountMetric)
MetricRegistry.register("builtin", "tool_call_count", ToolCallCountMetric)
MetricRegistry.register("builtin", "latency_seconds", LatencyMetric)
MetricRegistry.register("builtin", "token_usage", TokenUsageMetric)
