"""Strict per-sample requirements for conversational DeepEval metrics."""

import logging
from collections.abc import Callable, Sequence

from floeval.api.metrics.base import BaseMetric
from floeval.config.schemas.io.agent_dataset import AgentSample
from floeval.config.schemas.io.conversational_dataset import ConversationalSample
from floeval.metric_providers.deepeval.multiturn_adapter import conversational_row_to_deepeval

logger = logging.getLogger(__name__)
RequirementCheck = Callable[[ConversationalSample | AgentSample], bool]


def _has_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _has_any(values: object) -> bool:
    return isinstance(values, list) and len(values) > 0


def _has_assistant_tools(sample: ConversationalSample | AgentSample) -> bool:
    test_case = conversational_row_to_deepeval(sample)
    return any(t.role == "assistant" and bool(t.tools_called) for t in test_case.turns)


def _has_assistant_retrieval_context(sample: ConversationalSample | AgentSample) -> bool:
    test_case = conversational_row_to_deepeval(sample)
    return any(t.role == "assistant" and bool(t.retrieval_context) for t in test_case.turns)


def _missing_requirement(metric_name: str, sample_index: int, requirement: str) -> str:
    return f"deepeval:{metric_name} requires {requirement}; missing at sample index {sample_index}"


def _raise_aggregated_errors(errors: list[str]) -> None:
    if not errors:
        return
    lines = ["DeepEval conversational validation failed:", *[f"- {error}" for error in errors]]
    message = "\n".join(lines)
    logger.error(message)
    raise ValueError(message)


def _has_chatbot_role(sample: ConversationalSample | AgentSample) -> bool:
    return _has_text(getattr(sample, "chatbot_role", None))


def _has_reference_topics(sample: ConversationalSample | AgentSample) -> bool:
    return _has_any(getattr(sample, "reference_topics", None))


def _has_reference_tool_calls(sample: ConversationalSample | AgentSample) -> bool:
    return _has_any(getattr(sample, "reference_tool_calls", None))


def _has_reference_outcome(sample: ConversationalSample | AgentSample) -> bool:
    return _has_text(getattr(sample, "reference_outcome", None))


METRIC_REQUIREMENTS: dict[str, list[tuple[str, RequirementCheck]]] = {
    "role_adherence": [("chatbot_role", _has_chatbot_role)],
    "topic_adherence": [("reference_topics", _has_reference_topics)],
    "tool_use": [
        ("reference_tool_calls", _has_reference_tool_calls),
        ("assistant turn tools_called", _has_assistant_tools),
    ],
    "turn_faithfulness": [("assistant turn retrieval_context", _has_assistant_retrieval_context)],
    "turn_contextual_precision": [
        ("reference_outcome", _has_reference_outcome),
        ("assistant turn retrieval_context", _has_assistant_retrieval_context),
    ],
    "turn_contextual_recall": [
        ("reference_outcome", _has_reference_outcome),
        ("assistant turn retrieval_context", _has_assistant_retrieval_context),
    ],
    "turn_contextual_relevancy": [
        ("assistant turn retrieval_context", _has_assistant_retrieval_context)
    ],
}


def validate_conversational_metric_requirements(
    metrics: Sequence[BaseMetric], rows: Sequence[ConversationalSample | AgentSample]
) -> None:
    """Validate required per-sample fields before conversational DeepEval batch execution."""
    metric_names = {m.name for m in metrics}
    if not metric_names:
        return

    active_requirements = {
        metric_name: METRIC_REQUIREMENTS[metric_name]
        for metric_name in metric_names
        if metric_name in METRIC_REQUIREMENTS
    }
    if not active_requirements:
        return

    errors: list[str] = []
    for i, row in enumerate(rows):
        for metric_name, requirements in active_requirements.items():
            for requirement_name, requirement_check in requirements:
                if requirement_check(row):
                    continue
                errors.append(_missing_requirement(metric_name, i, requirement_name))

    _raise_aggregated_errors(errors)
