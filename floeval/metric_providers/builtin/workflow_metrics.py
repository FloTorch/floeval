"""Builtin workflow-level metrics.

Metrics registered:
- builtin:workflow_completion_rate  (deterministic — fraction of agents that succeeded)
- builtin:agent_handoff_quality     (LLM-based — evaluates inter-agent information transfer)
- builtin:cross_agent_consistency   (LLM-based — detects contradictions across agent outputs)

IMPORTANT: These metrics have execution_scope = "per_workflow".
WorkflowEvaluation detects this attribute and passes a WorkflowExecution object
to evaluate(), not an AgentSample. Do NOT use these metrics inside AgentEvaluation;
they will return score=None with a warning if called with an AgentSample.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from floeval.api.metrics.base import BaseMetric, MetricResult
from floeval.api.metrics.registry import MetricRegistry
from floeval.config.schemas.io.agent_dataset import WorkflowExecution, _to_display_str
from floeval.core.execution.llm_executor import OpenAIProvider

logger = logging.getLogger(__name__)


def _parse_json_response(response: str) -> dict:
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {}


class WorkflowCompletionRateMetric(BaseMetric):
    """Deterministic: fraction of agents that completed with SUCCESS status.

    Computed from WorkflowExecution.node_results. An agent node is considered
    successful when its result dict contains status == "SUCCESS".
    Score: 0.0–1.0. 1.0 = all agents succeeded.
    """

    execution_scope = "per_workflow"

    def __init__(self, threshold: float = 1.0, **kwargs):
        super().__init__(name="workflow_completion_rate", **kwargs)
        self.provider = "builtin"
        self.threshold = threshold

    def evaluate(self, execution: WorkflowExecution, **kwargs) -> MetricResult:  # type: ignore[override]
        if not isinstance(execution, WorkflowExecution):
            logger.warning(
                "WorkflowCompletionRateMetric received %s instead of WorkflowExecution. "
                "This metric requires execution_scope='per_workflow' and must be used "
                "inside WorkflowEvaluation.",
                type(execution).__name__,
            )
            return MetricResult(
                score=None,
                metadata={"error": "WorkflowExecution required", "provider": "builtin"},
            )
        rate = execution.workflow_completion_rate
        return MetricResult(
            score=round(rate, 4),
            metadata={
                "passed": rate >= self.threshold,
                "threshold": self.threshold,
                "completed_agents": execution.completed_agents,
                "total_agents": execution.total_agents,
                "provider": "builtin",
                "metric_name": "workflow_completion_rate",
            },
        )


class AgentHandoffQualityMetric(BaseMetric):
    """LLM-as-judge: evaluates information quality at each agent-to-agent handoff.

    For each consecutive pair of agent traces, evaluates whether the upstream
    agent's output was a useful and complete input for the downstream agent.
    Score = average across all handoff pairs.

    Returns score=None if fewer than 2 agent traces exist.
    """

    execution_scope = "per_workflow"

    _PROMPT = """You are evaluating an agent-to-agent handoff in a multi-agent workflow.

WORKFLOW GOAL:
{workflow_goal}

UPSTREAM AGENT OUTPUT (agent {upstream_idx}):
{upstream_output}

DOWNSTREAM AGENT'S FIRST INPUT (agent {downstream_idx}):
{downstream_input}

Was the upstream agent's output a useful and complete input for the downstream agent
given the workflow goal?

Score 1.0 if output was complete, relevant, and ready to use.
Score 0.5 if output was usable but had gaps or needed interpretation.
Score 0.0 if output was incomplete, irrelevant, or harmful to downstream.

Respond ONLY with JSON:
{{"score": <float 0.0 to 1.0>, "issues": "<brief note if any, else empty string>"}}"""

    def __init__(self, llm_provider: OpenAIProvider, threshold: float = 0.6, **kwargs):
        super().__init__(name="agent_handoff_quality", **kwargs)
        self.provider = "builtin"
        self.threshold = threshold
        self._llm_provider = llm_provider

    def evaluate(self, execution: WorkflowExecution, **kwargs) -> MetricResult:  # type: ignore[override]
        if not isinstance(execution, WorkflowExecution):
            logger.warning(
                "AgentHandoffQualityMetric received %s instead of WorkflowExecution.",
                type(execution).__name__,
            )
            return MetricResult(
                score=None,
                metadata={"error": "WorkflowExecution required", "provider": "builtin"},
            )
        pairs = execution.get_handoff_pairs()
        if not pairs:
            return MetricResult(
                score=None,
                metadata={
                    "reason": "Fewer than 2 agent traces; no handoffs to evaluate",
                    "provider": "builtin",
                },
            )

        scores: list[float] = []
        handoff_details: list[dict] = []

        for i, (upstream, downstream) in enumerate(pairs):
            upstream_output = (upstream.final_response or "")[:1500]
            downstream_input = (
                (downstream.messages[0].content or "")[:800]
                if downstream.messages
                else "(not captured)"
            )

            prompt = self._PROMPT.format(
                workflow_goal=_to_display_str(execution.user_input)[:300],
                upstream_idx=i + 1,
                upstream_output=upstream_output,
                downstream_idx=i + 2,
                downstream_input=downstream_input,
            )
            try:
                response = self._llm_provider.generate(prompt)
                data = _parse_json_response(response)
                if data:
                    score = float(max(0.0, min(1.0, data.get("score", 0))))
                    scores.append(score)
                    handoff_details.append({
                        "handoff_index": i,
                        "score": score,
                        "issues": data.get("issues", ""),
                    })
            except Exception as e:
                logger.warning("Handoff %d evaluation failed: %s", i, e)

        if not scores:
            return MetricResult(
                score=None,
                metadata={"error": "All handoff evaluations failed", "provider": "builtin"},
            )

        avg = sum(scores) / len(scores)
        return MetricResult(
            score=round(avg, 4),
            metadata={
                "passed": avg >= self.threshold,
                "threshold": self.threshold,
                "handoff_count": len(pairs),
                "handoff_details": handoff_details,
                "provider": "builtin",
                "metric_name": "agent_handoff_quality",
            },
        )

    async def aevaluate(self, execution: WorkflowExecution, **kwargs) -> MetricResult:  # type: ignore[override]
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.evaluate(execution, **kwargs))


class CrossAgentConsistencyMetric(BaseMetric):
    """LLM-as-judge: detects factual contradictions across agent outputs.

    Evaluates whether agents in the same workflow produce contradictory factual
    claims (e.g., agent 1 says price is $100, agent 3 says $150).
    Score: 1.0 = no contradictions. 0.0 = significant contradictions.
    Requires at least 2 agent traces.
    """

    execution_scope = "per_workflow"

    _PROMPT = """You are evaluating factual consistency across multiple AI agent outputs in a workflow.

WORKFLOW GOAL:
{workflow_goal}

AGENT OUTPUTS (in execution order):
{agent_outputs}

Are there any factual contradictions between agent outputs?
For example: different numbers, different names, different dates for the same entity.

Score 1.0 if all outputs are factually consistent with each other.
Score 0.5 if minor inconsistencies exist but don't affect the workflow outcome.
Score 0.0 if major contradictions exist that would mislead the user.

Respond ONLY with JSON:
{{"score": <float 0.0 to 1.0>, "contradictions": ["<describe any contradiction found>"]}}"""

    def __init__(self, llm_provider: OpenAIProvider, threshold: float = 0.7, **kwargs):
        super().__init__(name="cross_agent_consistency", **kwargs)
        self.provider = "builtin"
        self.threshold = threshold
        self._llm_provider = llm_provider

    def evaluate(self, execution: WorkflowExecution, **kwargs) -> MetricResult:  # type: ignore[override]
        if not isinstance(execution, WorkflowExecution):
            logger.warning(
                "CrossAgentConsistencyMetric received %s instead of WorkflowExecution.",
                type(execution).__name__,
            )
            return MetricResult(
                score=None,
                metadata={"error": "WorkflowExecution required", "provider": "builtin"},
            )
        if execution.total_agents < 2:
            return MetricResult(
                score=None,
                metadata={
                    "reason": "At least 2 agent traces required",
                    "provider": "builtin",
                },
            )

        names = execution.agent_names or [
            f"agent_{i}" for i in range(len(execution.agent_traces))
        ]
        agent_outputs_text = "\n\n".join(
            f"Agent {i + 1} ({name}):\n{trace.final_response[:400]}"
            for i, (name, trace) in enumerate(zip(names, execution.agent_traces))
        )

        prompt = self._PROMPT.format(
            workflow_goal=_to_display_str(execution.user_input)[:300],
            agent_outputs=agent_outputs_text,
        )
        try:
            response = self._llm_provider.generate(prompt)
            data = _parse_json_response(response)
            if not data:
                return MetricResult(
                    score=None,
                    metadata={"error": "No JSON in response", "provider": "builtin"},
                )
            score = float(max(0.0, min(1.0, data.get("score", 0))))
            return MetricResult(
                score=score,
                metadata={
                    "passed": score >= self.threshold,
                    "threshold": self.threshold,
                    "contradictions": data.get("contradictions", []),
                    "provider": "builtin",
                    "metric_name": "cross_agent_consistency",
                },
            )
        except Exception as e:
            logger.error("CrossAgentConsistencyMetric failed: %s", e, exc_info=True)
            return MetricResult(score=None, metadata={"error": str(e), "provider": "builtin"})

    async def aevaluate(self, execution: WorkflowExecution, **kwargs) -> MetricResult:  # type: ignore[override]
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.evaluate(execution, **kwargs))


MetricRegistry.register("builtin", "workflow_completion_rate", WorkflowCompletionRateMetric)
MetricRegistry.register("builtin", "agent_handoff_quality", AgentHandoffQualityMetric)
MetricRegistry.register("builtin", "cross_agent_consistency", CrossAgentConsistencyMetric)
