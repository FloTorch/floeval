"""Builtin tool usage metrics — all deterministic, zero LLM cost.

Metrics registered:
- builtin:tool_call_success_rate    (heuristic error detection on ToolMessage content)
- builtin:tool_selection_accuracy   (F1 score on tool names; requires reference_tool_calls)
- builtin:tool_argument_correctness (key-value match on args; requires reference_tool_calls)

All return score=None gracefully when required inputs are missing.
"""

from __future__ import annotations

import logging

from floeval.api.metrics.base import BaseMetric, MetricResult
from floeval.api.metrics.registry import MetricRegistry
from floeval.config.schemas.io.agent_dataset import AgentSample, ToolMessage

logger = logging.getLogger(__name__)

# Prefixes that unambiguously signal an error regardless of content length
_ERROR_PREFIXES = ("error:", "exception:", "failed:", "traceback", "http error")

# Short-content keywords — only treated as errors when content is brief (likely
# a bare error message, not a scraped page that happens to mention the phrase)
_SHORT_ERROR_KEYWORDS = frozenset([
    "not found", "unauthorized", "forbidden", "timeout",
    "connection refused", "internal server error", "bad gateway",
    "service unavailable",
])

# HTTP status codes that unambiguously indicate failure
_HTTP_ERROR_CODES = frozenset(["400", "401", "403", "404", "429", "500", "502", "503", "504"])

# Content shorter than this is treated as a bare error message
_SHORT_CONTENT_THRESHOLD = 200


def _is_tool_error(content: str) -> bool:
    """Return True only when the tool output is clearly an error, not normal content.

    Strategy:
    - Short responses (< 200 chars): any error keyword or HTTP code triggers failure.
    - Long responses: only explicit error prefixes at the start trigger failure.
      This avoids false positives from crawled pages that happen to mention
      phrases like 'not found' inside regular text.
    """
    text = (content or "").strip()
    if not text:
        return False

    lower = text.lower()
    is_short = len(text) < _SHORT_CONTENT_THRESHOLD

    # Explicit error prefix at the very start — always an error
    if any(lower.startswith(prefix) for prefix in _ERROR_PREFIXES):
        return True

    # Bare HTTP status code as the entire content
    if text.strip() in _HTTP_ERROR_CODES:
        return True

    # Short content: looser keyword matching
    if is_short:
        return any(kw in lower for kw in _SHORT_ERROR_KEYWORDS)

    return False


class ToolCallSuccessRateMetric(BaseMetric):
    """Deterministic: fraction of tool result messages that appear successful.

    Uses heuristic error detection on ToolMessage content strings.
    Returns score=None if no tool calls exist in the trace.
    Score: 0.0–1.0. 1.0 = all tool calls succeeded.
    """

    def __init__(self, threshold: float = 0.8, **kwargs):
        super().__init__(name="tool_call_success_rate", **kwargs)
        self.provider = "builtin"
        self.threshold = threshold

    def evaluate(self, sample: AgentSample, **kwargs) -> MetricResult:
        if not sample.trace:
            return MetricResult(
                score=None,
                metadata={"error": "trace required", "provider": "builtin"},
            )

        tool_messages = [m for m in sample.trace.messages if isinstance(m, ToolMessage)]
        if not tool_messages:
            return MetricResult(
                score=None,
                metadata={"reason": "No tool calls in trace", "provider": "builtin"},
            )

        success_count = sum(1 for m in tool_messages if not _is_tool_error(m.content))
        rate = success_count / len(tool_messages)

        return MetricResult(
            score=round(rate, 4),
            metadata={
                "passed": rate >= self.threshold,
                "threshold": self.threshold,
                "total_tool_calls": len(tool_messages),
                "successful_calls": success_count,
                "failed_calls": len(tool_messages) - success_count,
                "provider": "builtin",
                "metric_name": "tool_call_success_rate",
            },
        )


class ToolSelectionAccuracyMetric(BaseMetric):
    """Deterministic: F1 score of expected vs actual tool names.

    Compares tool names called by the agent against reference_tool_calls.
    Returns score=None if reference_tool_calls not provided.
    Score: 0.0–1.0 (F1 of tool name sets).
    """

    def __init__(self, threshold: float = 0.8, **kwargs):
        super().__init__(name="tool_selection_accuracy", **kwargs)
        self.provider = "builtin"
        self.threshold = threshold

    def evaluate(self, sample: AgentSample, **kwargs) -> MetricResult:
        if not sample.trace:
            return MetricResult(
                score=None,
                metadata={"error": "trace required", "provider": "builtin"},
            )
        if not sample.reference_tool_calls:
            return MetricResult(
                score=None,
                metadata={
                    "reason": "reference_tool_calls required for tool_selection_accuracy",
                    "provider": "builtin",
                },
            )

        actual_names = {tc.name for tc in sample.trace.tool_calls_made}
        expected_names = {tc.name for tc in sample.reference_tool_calls}

        if not expected_names and not actual_names:
            return MetricResult(
                score=1.0,
                metadata={"reason": "No tools expected or called", "provider": "builtin"},
            )

        tp = len(actual_names & expected_names)
        precision = tp / len(actual_names) if actual_names else 0.0
        recall = tp / len(expected_names) if expected_names else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return MetricResult(
            score=round(f1, 4),
            metadata={
                "passed": f1 >= self.threshold,
                "threshold": self.threshold,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "expected_tools": sorted(expected_names),
                "actual_tools": sorted(actual_names),
                "missing_tools": sorted(expected_names - actual_names),
                "extra_tools": sorted(actual_names - expected_names),
                "provider": "builtin",
                "metric_name": "tool_selection_accuracy",
            },
        )


class ToolArgumentCorrectnessMetric(BaseMetric):
    """Deterministic: argument-level correctness for matched tool calls.

    For each tool that appears in both actual and reference, computes the
    fraction of reference argument keys whose values match (case-insensitive
    string comparison).
    Returns score=None if reference_tool_calls not provided.
    Score: 0.0–1.0 (average across matched tool calls).
    """

    def __init__(self, threshold: float = 0.7, **kwargs):
        super().__init__(name="tool_argument_correctness", **kwargs)
        self.provider = "builtin"
        self.threshold = threshold

    def evaluate(self, sample: AgentSample, **kwargs) -> MetricResult:
        if not sample.trace:
            return MetricResult(
                score=None,
                metadata={"error": "trace required", "provider": "builtin"},
            )
        if not sample.reference_tool_calls:
            return MetricResult(
                score=None,
                metadata={"reason": "reference_tool_calls required", "provider": "builtin"},
            )

        ref_by_name = {tc.name: tc for tc in sample.reference_tool_calls}
        actual_by_name = {tc.name: tc for tc in sample.trace.tool_calls_made}
        matched = set(ref_by_name) & set(actual_by_name)

        if not matched:
            return MetricResult(
                score=0.0,
                metadata={
                    "reason": "No matching tool names between actual and reference",
                    "provider": "builtin",
                },
            )

        per_tool_scores: list[float] = []
        per_tool_detail: dict[str, dict] = {}

        for name in matched:
            ref_args = ref_by_name[name].args or {}
            actual_args = actual_by_name[name].args or {}

            if not ref_args:
                per_tool_scores.append(1.0)
                per_tool_detail[name] = {"score": 1.0, "note": "no args to check"}
                continue

            correct = sum(
                1
                for k, v in ref_args.items()
                if str(actual_args.get(k, "")).strip().lower() == str(v).strip().lower()
            )
            tool_score = correct / len(ref_args)
            per_tool_scores.append(tool_score)
            per_tool_detail[name] = {
                "score": round(tool_score, 4),
                "ref_args": ref_args,
                "actual_args": actual_args,
            }

        avg = sum(per_tool_scores) / len(per_tool_scores)
        return MetricResult(
            score=round(avg, 4),
            metadata={
                "passed": avg >= self.threshold,
                "threshold": self.threshold,
                "per_tool": per_tool_detail,
                "provider": "builtin",
                "metric_name": "tool_argument_correctness",
            },
        )


MetricRegistry.register("builtin", "tool_call_success_rate", ToolCallSuccessRateMetric)
MetricRegistry.register("builtin", "tool_selection_accuracy", ToolSelectionAccuracyMetric)
MetricRegistry.register("builtin", "tool_argument_correctness", ToolArgumentCorrectnessMetric)
