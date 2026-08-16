"""Builtin agent evaluation metrics.

Metrics registered by this module:
- builtin:task_completion          (NEW — preferred for agent task evaluation)
- builtin:trajectory_faithfulness  (NEW — trace-based coherence evaluation)
- builtin:goal_achievement         (DEPRECATED — kept for backward compatibility)
- builtin:response_coherence       (DEPRECATED — kept for backward compatibility)

Why task_completion instead of ragas:agent_goal_accuracy as default:
    RAGAS agent_goal_accuracy uses an LLM to first "infer" an end_state from
    the trace, then compares it to the reference via NLI. With custom gateway
    LLMs (smaller than GPT-4 quality), the inference step produces generic
    summaries ("The AI answered the question"), making the comparison meaningless.

    task_completion calls the judge LLM directly on user_input + final_response
    without an inference intermediary. This works correctly across all LLM sizes.
"""

import asyncio
import json
import logging
import re
import warnings

from floeval.api.metrics.base import BaseMetric, MetricResult
from floeval.api.metrics.registry import MetricRegistry
from floeval.config.schemas.io.agent_dataset import AgentSample, _to_display_str
from floeval.core.execution.llm_executor import OpenAIProvider

logger = logging.getLogger(__name__)


def _fix_json_escapes(s: str) -> str:
    """Fix invalid JSON escape sequences so json.loads succeeds."""
    return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", s)


def _parse_llm_json(raw: str) -> dict:
    """Parse JSON from LLM response, tolerating invalid escapes."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(_fix_json_escapes(raw))


def _format_trace_for_prompt(sample: AgentSample) -> str:
    """Format AgentTrace messages into readable text for LLM judge prompts."""
    lines = []
    for i, msg in enumerate(sample.trace.messages):
        role = msg.role.upper()
        content = (msg.content or "")[:400]
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            calls_str = ", ".join(f"{tc.name}({json.dumps(tc.args)[:80]})" for tc in msg.tool_calls)
            lines.append(f"[Step {i + 1}] {role}: {content} | TOOL_CALLS: {calls_str}")
        elif hasattr(msg, "tool_name") and msg.tool_name:
            lines.append(f"[Step {i + 1}] TOOL_RESULT ({msg.tool_name}): {content}")
        else:
            lines.append(f"[Step {i + 1}] {role}: {content}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# TaskCompletionMetric — preferred for agent task evaluation
# ─────────────────────────────────────────────────────────────────────────────


class TaskCompletionMetric(BaseMetric):
    """LLM-as-judge: did the agent's final response complete the user's task?

    Evaluates whether the agent's final_response successfully addresses the
    user's request. Does NOT require a trace (uses final output only), but
    optionally uses reference_outcome for reference-based scoring.

    Preferred over ragas:agent_goal_accuracy because it:
    - Works with any LLM quality (no trace inference intermediary step)
    - Provides direct, deterministic prompting
    - Surfaces reasoning in metadata for debugging

    Required inputs: user_input, trace.final_response
    Optional inputs: reference_outcome
    Score: 0.0–1.0. Default threshold: 0.5.
    """

    _PROMPT = """You are an expert evaluator assessing whether an AI agent successfully completed a task.

USER REQUEST:
{user_input}

AGENT'S FINAL RESPONSE:
{final_response}

{reference_section}Evaluate: Did the agent successfully and completely address the user's request?

Score 1.0 if fully completed with correct information.
Score 0.7-0.9 if mostly complete with minor gaps.
Score 0.4-0.6 if partially complete.
Score 0.1-0.3 if attempted but largely failed.
Score 0.0 if failed entirely or off-topic.

Respond ONLY with valid JSON:
{{"score": <float 0.0 to 1.0>, "reasoning": "<one concise sentence>"}}"""

    def __init__(
        self,
        llm_provider: OpenAIProvider,
        threshold: float = 0.5,
        **kwargs,
    ):
        super().__init__(name="task_completion", **kwargs)
        self.provider = "builtin"
        self.threshold = threshold
        self._llm_provider = llm_provider

    def _build_prompt(self, sample: AgentSample) -> str:
        ref_section = ""
        if sample.reference_outcome:
            ref_section = f"EXPECTED OUTCOME:\n{_to_display_str(sample.reference_outcome)}\n\n"
        final_response = sample.trace.final_response if sample.trace else "No response captured."
        return self._PROMPT.format(
            user_input=_to_display_str(sample.user_input),
            final_response=final_response,
            reference_section=ref_section,
        )

    def _parse_response(self, response: str) -> MetricResult:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            return MetricResult(
                score=None,
                metadata={"error": f"No JSON in response: {response[:200]}"},
            )
        data = _parse_llm_json(match.group())
        score = float(max(0.0, min(1.0, data.get("score", 0))))
        return MetricResult(
            score=score,
            metadata={
                "passed": score >= self.threshold,
                "threshold": self.threshold,
                "reasoning": data.get("reasoning", ""),
                "provider": "builtin",
                "metric_name": "task_completion",
            },
        )

    def evaluate(self, sample: AgentSample, **kwargs) -> MetricResult:
        if not sample.trace:
            return MetricResult(score=None, metadata={"error": "AgentSample.trace is required"})
        try:
            response = self._llm_provider.generate(self._build_prompt(sample))
            return self._parse_response(response)
        except Exception as e:
            logger.error("TaskCompletionMetric failed: %s", e, exc_info=True)
            return MetricResult(score=None, metadata={"error": str(e), "provider": "builtin"})

    async def aevaluate(self, sample: AgentSample, **kwargs) -> MetricResult:
        if not sample.trace:
            return MetricResult(score=None, metadata={"error": "AgentSample.trace is required"})
        try:
            if hasattr(self._llm_provider, "agenerate"):
                response = await self._llm_provider.agenerate(self._build_prompt(sample))
            else:
                loop = asyncio.get_running_loop()
                prompt = self._build_prompt(sample)
                response = await loop.run_in_executor(
                    None, lambda: self._llm_provider.generate(prompt)
                )
            return self._parse_response(response)
        except Exception as e:
            logger.error("TaskCompletionMetric async failed: %s", e, exc_info=True)
            return MetricResult(score=None, metadata={"error": str(e), "provider": "builtin"})


# ─────────────────────────────────────────────────────────────────────────────
# TrajectoryFaithfulnessMetric
# ─────────────────────────────────────────────────────────────────────────────


class TrajectoryFaithfulnessMetric(BaseMetric):
    """LLM-as-judge: is the agent's execution trace logically coherent?

    Evaluates whether each step in the agent's trace is logically consistent
    with the user's request and previous steps.

    Unlike ragas:faithfulness (which checks RAG context attribution), this
    metric evaluates whether the agent's reasoning chain is coherent — i.e.,
    no contradictions, no irrelevant tool calls, no circular reasoning.

    Required inputs: AgentSample with trace containing multiple messages.
    Score: 0.0–1.0. 1.0 = all steps logically coherent.
    """

    _PROMPT = """You are an expert evaluator assessing an AI agent's execution trace coherence.

USER REQUEST:
{user_input}

AGENT EXECUTION TRACE:
{trace_text}

Evaluate whether each step in the trace logically follows from the previous steps
and is relevant to the user's request. Look for: contradictions, irrelevant tool
calls, steps that ignore previous results, or circular reasoning.

Score 1.0 if all steps are logically coherent and directly relevant.
Score 0.7-0.9 if mostly coherent with minor unnecessary steps.
Score 0.4-0.6 if some steps are incoherent or irrelevant.
Score 0.0-0.3 if major logical inconsistencies or many irrelevant steps.

Respond ONLY with valid JSON:
{{"score": <float 0.0 to 1.0>, "issues": ["<issue if any>"], "reasoning": "<brief explanation>"}}"""

    def __init__(self, llm_provider: OpenAIProvider, threshold: float = 0.5, **kwargs):
        super().__init__(name="trajectory_faithfulness", **kwargs)
        self.provider = "builtin"
        self.threshold = threshold
        self._llm_provider = llm_provider

    def _build_prompt(self, sample: AgentSample) -> str:
        return self._PROMPT.format(
            user_input=_to_display_str(sample.user_input),
            trace_text=_format_trace_for_prompt(sample),
        )

    def _parse_response(self, response: str) -> MetricResult:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            return MetricResult(score=None, metadata={"error": "No JSON in response"})
        data = _parse_llm_json(match.group())
        score = float(max(0.0, min(1.0, data.get("score", 0))))
        return MetricResult(
            score=score,
            metadata={
                "passed": score >= self.threshold,
                "threshold": self.threshold,
                "issues": data.get("issues", []),
                "reasoning": data.get("reasoning", ""),
                "provider": "builtin",
                "metric_name": "trajectory_faithfulness",
            },
        )

    def evaluate(self, sample: AgentSample, **kwargs) -> MetricResult:
        if not sample.trace or len(sample.trace.messages) < 2:
            return MetricResult(
                score=None,
                metadata={"error": "Trace with multiple messages required"},
            )
        try:
            response = self._llm_provider.generate(self._build_prompt(sample))
            return self._parse_response(response)
        except Exception as e:
            logger.error("TrajectoryFaithfulnessMetric failed: %s", e, exc_info=True)
            return MetricResult(score=None, metadata={"error": str(e), "provider": "builtin"})

    async def aevaluate(self, sample: AgentSample, **kwargs) -> MetricResult:
        if not sample.trace or len(sample.trace.messages) < 2:
            return MetricResult(
                score=None,
                metadata={"error": "Trace with multiple messages required"},
            )
        try:
            if hasattr(self._llm_provider, "agenerate"):
                response = await self._llm_provider.agenerate(self._build_prompt(sample))
            else:
                loop = asyncio.get_running_loop()
                prompt = self._build_prompt(sample)
                response = await loop.run_in_executor(
                    None, lambda: self._llm_provider.generate(prompt)
                )
            return self._parse_response(response)
        except Exception as e:
            logger.error("TrajectoryFaithfulnessMetric async failed: %s", e, exc_info=True)
            return MetricResult(score=None, metadata={"error": str(e), "provider": "builtin"})


# ─────────────────────────────────────────────────────────────────────────────
# GoalAchievementMetric — DEPRECATED
# ─────────────────────────────────────────────────────────────────────────────


class GoalAchievementMetric(BaseMetric):
    """DEPRECATED: Use builtin:task_completion for better reliability.

    Kept for backward compatibility. Will be removed in a future release.
    """

    _PROMPT = """You are an expert evaluator assessing an AI agent's goal achievement.

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
        warnings.warn(
            "builtin:goal_achievement will be deprecated. Use builtin:task_completion for better reliability.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(name="goal_achievement", **kwargs)
        self.provider = "builtin"
        self.threshold = threshold
        self._llm_provider = llm_provider

    def evaluate(self, sample: AgentSample, **kwargs) -> MetricResult:
        ref = ""
        if sample.reference_outcome:
            ref = f"EXPECTED OUTCOME:\n{_to_display_str(sample.reference_outcome)}\n"

        prompt = self._PROMPT.format(
            user_input=_to_display_str(sample.user_input),
            final_response=sample.trace.final_response if sample.trace else "",
            reference=ref,
        )

        try:
            response = self._llm_provider.generate(prompt, **{"temperature": 0.0} | kwargs)
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if not match:
                return MetricResult(
                    score=None,
                    metadata={"error": f"No JSON in response: {response[:200]}"},
                )
            data = _parse_llm_json(match.group())
            score = float(max(0.0, min(1.0, data.get("score", 0))))
            return MetricResult(
                score=score,
                metadata={
                    "passed": score >= self.threshold,
                    "threshold": self.threshold,
                    "reasoning": data.get("reasoning", ""),
                    "provider": "builtin",
                    "deprecated": True,
                },
            )
        except Exception as e:
            logger.error("GoalAchievementMetric failed: %s", e, exc_info=True)
            return MetricResult(
                score=None,
                metadata={"error": str(e), "provider": "builtin"},
            )


# ─────────────────────────────────────────────────────────────────────────────
# ResponseCoherenceMetric — DEPRECATED
# ─────────────────────────────────────────────────────────────────────────────


class ResponseCoherenceMetric(BaseMetric):
    """DEPRECATED: Use builtin:trajectory_faithfulness.

    Kept for backward compatibility. Will be removed in a future release.
    """

    _PROMPT = """You are an expert evaluator assessing response coherence.

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
        warnings.warn(
            "builtin:response_coherence is deprecated. Use builtin:trajectory_faithfulness.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(name="response_coherence", **kwargs)
        self.provider = "builtin"
        self.threshold = threshold
        self._llm_provider = llm_provider

    def evaluate(self, sample: AgentSample, **kwargs) -> MetricResult:
        trace_parts = []
        for msg in sample.trace.messages:
            role = getattr(msg, "role", "unknown")
            content = getattr(msg, "content", "") or ""
            trace_parts.append(f"[{role}] {content[:500]}{'...' if len(content) > 500 else ''}")
        trace_summary = "\n".join(trace_parts) if trace_parts else "(empty trace)"

        prompt = self._PROMPT.format(
            trace_summary=trace_summary,
            final_response=sample.trace.final_response,
        )

        try:
            response = self._llm_provider.generate(prompt, **{"temperature": 0.0} | kwargs)
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if not match:
                return MetricResult(
                    score=None,
                    metadata={"error": f"No JSON in response: {response[:200]}"},
                )
            data = _parse_llm_json(match.group())
            score = float(max(0.0, min(1.0, data.get("score", 0))))
            return MetricResult(
                score=score,
                metadata={
                    "passed": score >= self.threshold,
                    "threshold": self.threshold,
                    "reasoning": data.get("reasoning", ""),
                    "provider": "builtin",
                    "deprecated": True,
                },
            )
        except Exception as e:
            logger.error("ResponseCoherenceMetric failed: %s", e, exc_info=True)
            return MetricResult(
                score=None,
                metadata={"error": str(e), "provider": "builtin"},
            )


MetricRegistry.register("builtin", "task_completion", TaskCompletionMetric)
MetricRegistry.register("builtin", "trajectory_faithfulness", TrajectoryFaithfulnessMetric)
MetricRegistry.register("builtin", "goal_achievement", GoalAchievementMetric)
MetricRegistry.register("builtin", "response_coherence", ResponseCoherenceMetric)
