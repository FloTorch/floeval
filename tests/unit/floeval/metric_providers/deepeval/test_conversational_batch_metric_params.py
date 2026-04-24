"""Tests for strict conversational DeepEval metric requirements."""

# ruff: noqa: D103

import pytest

from floeval.config.schemas.io.conversational_dataset import ConversationalSample
from floeval.metric_providers.deepeval.batch_eval import run_deepeval_conversational_batch
from floeval.metric_providers.deepeval.conversational_multiturn_metrics import (
    RoleAdherenceDeepEvalMetric,
    ToolUseDeepEvalMetric,
    TopicAdherenceDeepEvalMetric,
    TurnContextualPrecisionDeepEvalMetric,
    TurnFaithfulnessDeepEvalMetric,
)
from floeval.metric_providers.deepeval.conversational_requirements import (
    validate_conversational_metric_requirements,
)

pytestmark = pytest.mark.unit


def test_topic_adherence_requires_reference_topics_on_every_sample() -> None:
    metric = TopicAdherenceDeepEvalMetric(params={})
    rows = [
        ConversationalSample.model_validate(
            {
                "turns": [
                    {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "a1"},
                ]
            }
        )
    ]

    with pytest.raises(
        ValueError,
        match="deepeval:topic_adherence requires reference_topics; missing at sample index 0",
    ):
        validate_conversational_metric_requirements([metric], rows)


def test_tool_use_requires_reference_tool_calls() -> None:
    metric = ToolUseDeepEvalMetric(params={})
    rows = [
        ConversationalSample.model_validate(
            {
                "turns": [
                    {"role": "user", "content": "u1"},
                    {
                        "role": "assistant",
                        "content": "a1",
                        "tools_called": [{"name": "lookup", "args": {"id": "1"}}],
                    },
                ],
            }
        )
    ]

    with pytest.raises(
        ValueError,
        match="deepeval:tool_use requires reference_tool_calls; missing at sample index 0",
    ):
        validate_conversational_metric_requirements([metric], rows)


def test_tool_use_requires_assistant_tools_called() -> None:
    metric = ToolUseDeepEvalMetric(params={})
    rows = [
        ConversationalSample.model_validate(
            {
                "turns": [
                    {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "a1"},
                ],
                "reference_tool_calls": [{"name": "lookup", "args": {"id": "1"}}],
            }
        )
    ]

    with pytest.raises(
        ValueError,
        match="deepeval:tool_use requires assistant turn tools_called; missing at sample index 0",
    ):
        validate_conversational_metric_requirements([metric], rows)


def test_turn_faithfulness_requires_assistant_retrieval_context() -> None:
    metric = TurnFaithfulnessDeepEvalMetric(params={})
    rows = [
        ConversationalSample.model_validate(
            {
                "turns": [
                    {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "a1"},
                ],
            }
        )
    ]

    with pytest.raises(
        ValueError,
        match=(
            "deepeval:turn_faithfulness requires assistant turn retrieval_context; "
            "missing at sample index 0"
        ),
    ):
        validate_conversational_metric_requirements([metric], rows)


def test_turn_contextual_precision_requires_reference_outcome() -> None:
    metric = TurnContextualPrecisionDeepEvalMetric(params={})
    rows = [
        ConversationalSample.model_validate(
            {
                "turns": [
                    {"role": "user", "content": "u1"},
                    {
                        "role": "assistant",
                        "content": "a1",
                        "retrieval_context": ["ctx"],
                    },
                ]
            }
        )
    ]

    with pytest.raises(
        ValueError,
        match=(
            "deepeval:turn_contextual_precision requires reference_outcome; "
            "missing at sample index 0"
        ),
    ):
        validate_conversational_metric_requirements([metric], rows)


def test_role_adherence_requires_chatbot_role() -> None:
    metric = RoleAdherenceDeepEvalMetric(params={})
    rows = [
        ConversationalSample.model_validate(
            {
                "turns": [
                    {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "a1"},
                ]
            }
        )
    ]

    with pytest.raises(
        ValueError,
        match="deepeval:role_adherence requires chatbot_role; missing at sample index 0",
    ):
        validate_conversational_metric_requirements([metric], rows)


def test_validation_does_not_mutate_metric_params() -> None:
    metric = TopicAdherenceDeepEvalMetric(params={"relevant_topics": ["from-yaml"]})
    rows = [
        ConversationalSample.model_validate(
            {
                "turns": [
                    {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "a1"},
                ],
                "reference_topics": ["row-topic"],
            }
        )
    ]

    validate_conversational_metric_requirements([metric], rows)

    assert metric._metric_params["relevant_topics"] == ["from-yaml"]


def test_validation_reports_all_missing_requirements() -> None:
    metrics = [ToolUseDeepEvalMetric(params={}), TopicAdherenceDeepEvalMetric(params={})]
    rows = [
        ConversationalSample.model_validate(
            {
                "turns": [
                    {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "a1"},
                ]
            }
        )
    ]

    with pytest.raises(ValueError) as exc_info:
        validate_conversational_metric_requirements(metrics, rows)

    msg = str(exc_info.value)
    assert "DeepEval conversational validation failed:" in msg
    assert "deepeval:tool_use requires reference_tool_calls; missing at sample index 0" in msg
    assert (
        "deepeval:tool_use requires assistant turn tools_called; missing at sample index 0"
        in msg
    )
    assert "deepeval:topic_adherence requires reference_topics; missing at sample index 0" in msg


def test_batch_runner_raises_with_aggregated_validation_errors() -> None:
    metrics = [ToolUseDeepEvalMetric(params={}), TopicAdherenceDeepEvalMetric(params={})]
    rows = [
        ConversationalSample.model_validate(
            {
                "turns": [
                    {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "a1"},
                ]
            }
        )
    ]

    with pytest.raises(ValueError) as exc_info:
        run_deepeval_conversational_batch(
            conversational_rows=rows, floeval_metrics=metrics, llm_config=None
        )

    msg = str(exc_info.value)
    assert "DeepEval conversational validation failed:" in msg
    assert "deepeval:tool_use requires reference_tool_calls; missing at sample index 0" in msg
    assert (
        "deepeval:tool_use requires assistant turn tools_called; missing at sample index 0"
        in msg
    )
    assert "deepeval:topic_adherence requires reference_topics; missing at sample index 0" in msg


def test_validation_reports_multiple_sample_indices() -> None:
    metric = TopicAdherenceDeepEvalMetric(params={})
    rows = [
        ConversationalSample.model_validate(
            {
                "turns": [
                    {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "a1"},
                ]
            }
        ),
        ConversationalSample.model_validate(
            {
                "turns": [
                    {"role": "user", "content": "u2"},
                    {"role": "assistant", "content": "a2"},
                ]
            }
        ),
    ]

    with pytest.raises(ValueError) as exc_info:
        validate_conversational_metric_requirements([metric], rows)

    msg = str(exc_info.value)
    assert "deepeval:topic_adherence requires reference_topics; missing at sample index 0" in msg
    assert "deepeval:topic_adherence requires reference_topics; missing at sample index 1" in msg
