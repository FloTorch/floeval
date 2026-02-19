"""Agent evaluation metrics."""

import json
import logging
import re

from floeval.api.metrics.base import BaseMetric, MetricResult
from floeval.api.metrics.registry import MetricRegistry
from floeval.config.schemas.io.agent_dataset import AgentSample
from floeval.core.execution.llm_executor import OpenAIProvider

logger = logging.getLogger(__name__)


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
            ref = f"EXPECTED OUTCOME:\n{sample.reference_outcome}\n"

        prompt = self.PROMPT.format(
            user_input=sample.user_input,
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

            data = json.loads(match.group())
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


MetricRegistry.register("builtin", "goal_achievement", GoalAchievementMetric)
