"""Tests for conversational DeepEval metric param hydration."""

# ruff: noqa: D103

import pytest

from floeval.config.schemas.io.conversational_dataset import ConversationalSample
from floeval.metric_providers.deepeval.batch_eval import (
    _hydrate_conversational_metric_params,
)
from floeval.metric_providers.deepeval.conversational_multiturn_metrics import (
    ToolUseDeepEvalMetric,
    TopicAdherenceDeepEvalMetric,
)

pytestmark = pytest.mark.unit


def test_topic_adherence_uses_reference_topics_when_missing_relevant_topics() -> None:
    metric = TopicAdherenceDeepEvalMetric(params={})
    rows = [
        ConversationalSample.model_validate(
            {
                "turns": [
                    {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "a1"},
                ],
                "reference_topics": ["billing", "refunds"],
            }
        )
    ]

    _hydrate_conversational_metric_params([metric], rows)

    assert metric._metric_params["relevant_topics"] == ["billing", "refunds"]


def test_tool_use_uses_reference_tool_calls_when_available_tools_missing() -> None:
    metric = ToolUseDeepEvalMetric(params={})
    rows = [
        ConversationalSample.model_validate(
            {
                "turns": [
                    {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "a1"},
                ],
                "reference_tool_calls": [
                    {"name": "search_orders", "args": {"order_id": "A-1"}}
                ],
            }
        )
    ]

    _hydrate_conversational_metric_params([metric], rows)

    assert "available_tools" in metric._metric_params
    assert len(metric._metric_params["available_tools"]) == 1
    assert metric._metric_params["available_tools"][0].name == "search_orders"
