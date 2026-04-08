"""Unit tests for conversational dataset rows."""

# ruff: noqa: D103 — test names describe behavior

import pytest

from floeval.api.dataset import conversational_dataset_from_dict
from floeval.config.schemas.io.conversation import ConversationTurn
from floeval.config.schemas.io.conversational_dataset import ConversationalSample


def test_conversation_turn_model() -> None:
    t = ConversationTurn(role="user", content="Hi")
    assert t.role == "user"
    assert t.content == "Hi"


def test_conversational_sample_validates_alternation() -> None:
    s = ConversationalSample(
        turns=[
            ConversationTurn(role="user", content="Hello"),
            ConversationTurn(role="assistant", content="Hi there"),
        ],
        scenario="s",
    )
    assert len(s.turns) == 2


def test_conversational_sample_too_few_turns_raises() -> None:
    with pytest.raises(ValueError, match="at least two"):
        ConversationalSample(
            turns=[ConversationTurn(role="user", content="only one")],
        )


def test_conversational_sample_accepts_expected_outcome_alias() -> None:
    s = ConversationalSample.model_validate(
        {
            "turns": [
                {"role": "user", "content": "Q?"},
                {"role": "assistant", "content": "A."},
            ],
            "expected_outcome": "Done",
        }
    )
    assert s.reference_outcome == "Done"


def test_conversational_dataset_from_dict_roundtrip() -> None:
    data = {
        "conversational_samples": [
            {
                "turns": [
                    {"role": "user", "content": "Q?"},
                    {"role": "assistant", "content": "A."},
                ],
                "scenario": "Support",
                "conversation_context": ["policy: be nice"],
            }
        ]
    }
    ds = conversational_dataset_from_dict(data, partial_dataset=False)
    assert len(ds.conversational_samples) == 1
    sp = ds.conversational_samples[0]
    assert isinstance(sp, ConversationalSample)
    assert sp.scenario == "Support"
    assert sp.conversation_context == ["policy: be nice"]
