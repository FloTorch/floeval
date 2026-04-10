"""Tests for Floeval → DeepEval conversational test case mapping."""

import pytest
from deepeval.test_case import ConversationalTestCase

from floeval.config.schemas.io.conversational_dataset import ConversationalSample
from floeval.metric_providers.deepeval.conversational_test_case import (
    sample_to_conversational_test_case,
)


def test_sample_to_conversational_test_case_maps_fields() -> None:
    sample = ConversationalSample.model_validate(
        {
            "turns": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
            "scenario": "S",
            "reference_outcome": "E",
            "user_description": "U",
            "chatbot_role": "C",
            "conversation_context": ["ctx"],
        }
    )
    ctc = sample_to_conversational_test_case(sample)
    assert isinstance(ctc, ConversationalTestCase)
    assert len(ctc.turns) == 2
    assert ctc.scenario == "S"
    assert ctc.expected_outcome == "E"
    assert ctc.user_description == "U"
    assert ctc.chatbot_role == "C"
    assert ctc.context == ["ctx"]


def test_sample_without_turns_shape_raises() -> None:
    with pytest.raises(ValueError, match="at least two"):
        ConversationalSample.model_validate({"turns": [{"role": "user", "content": "x"}]})


def test_conversational_sample_with_tools_called_maps_to_deepeval_turn_tools() -> None:
    sample = ConversationalSample.model_validate(
        {
            "turns": [
                {"role": "user", "content": "Please check order status"},
                {
                    "role": "assistant",
                    "content": "Checking now.",
                    "tools_called": [
                        {
                            "name": "get_order_status",
                            "input_parameters": {"order_id": "A123"},
                            "output": "Order is in transit",
                        }
                    ],
                },
            ],
            "reference_outcome": "Assistant shares the order status.",
        }
    )
    ctc = sample_to_conversational_test_case(sample)
    assert len(ctc.turns) == 2
    assert ctc.turns[1].tools_called is not None
    assert len(ctc.turns[1].tools_called) == 1
    tc = ctc.turns[1].tools_called[0]
    assert tc.name == "get_order_status"
    assert tc.input_parameters == {"order_id": "A123"}
    assert tc.output == "Order is in transit"
