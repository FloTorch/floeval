"""Agent evaluation metrics."""

import json
import logging
import re

from floeval.api.metrics.base import BaseMetric, MetricResult
from floeval.api.metrics.registry import MetricRegistry
from floeval.config.schemas.io.agent_dataset import AgentSample, _to_display_str
from floeval.core.execution.llm_executor import OpenAIProvider

logger = logging.getLogger(__name__)


def _fix_json_escapes(s: str) -> str:
    """Fix invalid JSON escape sequences (e.g. \\(, \\times) so json.loads succeeds."""
    # Backslash not followed by valid JSON escape char: " \ / b f n r t u
    return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", s)


def _parse_llm_json(raw: str) -> dict:
    """Parse JSON from LLM response, tolerating invalid escapes in strings."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(_fix_json_escapes(raw))


class GoalAchievementMetric(BaseMetric):
    """LLM-as-judge: Did agent achieve the goal?

    Receives llm_provider via constructor (dependency injection).
    AgentEvaluation creates and injects the provider.
    """

    PROMPT = """You are an expert evaluator assessing an AI agent's goal achievement.

USER REQUEST:
{user_input}

AGENT'S FINAL RESPONSE:
{final_response}

{reference}

Did the agent successfully complete the request?

Respond ONLY with JSON:
{{
    "score": between 0 and 1,
    "reasoning": "explanation"
}}
"""

    def __init__(
        self,
        llm_provider: OpenAIProvider,
        threshold: float = 0.5,
        **kwargs,
    ):
        super().__init__(name="goal_achievement", **kwargs)
        self.provider = "builtin"
        self.threshold = threshold
        self._llm_provider = llm_provider

    def evaluate(self, sample: AgentSample, **kwargs) -> MetricResult:
        """Evaluate whether agent achieved the goal."""
        ref = ""
        if sample.reference_outcome:
            ref = f"EXPECTED OUTCOME:\n{_to_display_str(sample.reference_outcome)}\n"

        prompt = self.PROMPT.format(
            user_input=_to_display_str(sample.user_input),
            final_response=sample.trace.final_response,
            reference=ref,
        )

        try:
            response = self._llm_provider.generate(
                prompt,
                **{"temperature": 0.0} | kwargs,
            )

            match = re.search(r"\{.*\}", response, re.DOTALL)
            if not match:
                return MetricResult(
                    score=None,
                    metadata={"error": f"No JSON in response: {response[:200]}"},
                )

            data = _parse_llm_json(match.group())
            score = float(data.get("score", 0))
            score = max(0.0, min(1.0, score))

            return MetricResult(
                score=score,
                metadata={
                    "passed": score >= self.threshold,
                    "threshold": self.threshold,
                    "reasoning": data.get("reasoning", ""),
                    "provider": "builtin",
                },
            )

        except Exception as e:
            logger.error("GoalAchievementMetric failed: %s", e, exc_info=True)
            return MetricResult(
                score=None,
                metadata={"error": str(e), "provider": "builtin"},
            )


class ResponseCoherenceMetric(BaseMetric):
    """LLM-as-judge: Is final response consistent with conversation trace?

    Receives llm_provider via constructor (dependency injection).
    No reference needed.
    """

    PROMPT = """You are an expert evaluator assessing response coherence.

Analyze the conversation trace and the final response below.

CONVERSATION TRACE (messages in order):
{trace_summary}

FINAL RESPONSE:
{final_response}

Is the final response consistent with the conversation trace? Does it logically follow from the preceding exchange?

Respond ONLY with JSON:
{{
    "score": between 0 and 1 (1 = fully coherent and consistent),
    "reasoning": "explanation"
}}
"""

    def __init__(
        self,
        llm_provider: OpenAIProvider,
        threshold: float = 0.5,
        **kwargs,
    ):
        super().__init__(name="response_coherence", **kwargs)
        self.provider = "builtin"
        self.threshold = threshold
        self._llm_provider = llm_provider

    def evaluate(self, sample: AgentSample, **kwargs) -> MetricResult:
        """Evaluate whether final response is coherent with trace."""
        trace_parts = []
        for msg in sample.trace.messages:
            role = getattr(msg, "role", "unknown")
            content = getattr(msg, "content", "") or ""
            trace_parts.append(f"[{role}] {content[:500]}{'...' if len(content) > 500 else ''}")
        trace_summary = "\n".join(trace_parts) if trace_parts else "(empty trace)"

        prompt = self.PROMPT.format(
            trace_summary=trace_summary,
            final_response=sample.trace.final_response,
        )

        try:
            response = self._llm_provider.generate(
                prompt,
                **{"temperature": 0.0} | kwargs,
            )

            match = re.search(r"\{.*\}", response, re.DOTALL)
            if not match:
                return MetricResult(
                    score=None,
                    metadata={"error": f"No JSON in response: {response[:200]}"},
                )

            data = _parse_llm_json(match.group())
            score = float(data.get("score", 0))
            score = max(0.0, min(1.0, score))

            return MetricResult(
                score=score,
                metadata={
                    "passed": score >= self.threshold,
                    "threshold": self.threshold,
                    "reasoning": data.get("reasoning", ""),
                    "provider": "builtin",
                },
            )

        except Exception as e:
            logger.error("ResponseCoherenceMetric failed: %s", e, exc_info=True)
            return MetricResult(
                score=None,
                metadata={"error": str(e), "provider": "builtin"},
            )


MetricRegistry.register("builtin", "goal_achievement", GoalAchievementMetric)
MetricRegistry.register("builtin", "response_coherence", ResponseCoherenceMetric)
