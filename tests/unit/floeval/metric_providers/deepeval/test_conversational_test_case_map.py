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
            "expected_outcome": "E",
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
