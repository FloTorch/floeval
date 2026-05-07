"""Unit tests for conversational dataset rows."""

# ruff: noqa: D103 — test names describe behavior

import pytest

from floeval.api.dataset import conversational_dataset_from_dict
from floeval.config.schemas.io.conversation import ConversationTurn, ToolCallPayload
from floeval.config.schemas.io.conversational_dataset import (
    ConversationalDataset,
    ConversationalSample,
)


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
        "samples": [
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
    assert len(ds.samples) == 1
    sp = ds.samples[0]
    assert isinstance(sp, ConversationalSample)
    assert sp.scenario == "Support"
    assert sp.conversation_context == ["policy: be nice"]


def test_conversational_dataset_model_dump_uses_samples_key() -> None:
    dataset = ConversationalDataset(
        samples=[
            ConversationalSample.model_validate(
                {
                    "turns": [
                        {"role": "user", "content": "Q?"},
                        {"role": "assistant", "content": "A."},
                    ]
                }
            )
        ]
    )
    payload = dataset.model_dump()
    assert "samples" in payload
    assert len(payload["samples"]) == 1


def test_tool_call_payload_aliases_are_unified_to_args() -> None:
    s = ConversationalSample.model_validate(
        {
            "turns": [
                {"role": "user", "content": "Q?"},
                {
                    "role": "assistant",
                    "content": "A.",
                    "tools_called": [
                        {
                            "name": "ragas_style",
                            "args": {"k": "v"},
                            "output": "ok",
                        },
                        {
                            "name": "deepeval_style",
                            "input_parameters": {"x": 1},
                            "output": "done",
                        },
                    ],
                },
            ]
        }
    )
    assert s.turns[1].tools_called is not None
    first, second = s.turns[1].tools_called
    assert isinstance(first, ToolCallPayload)
    assert isinstance(second, ToolCallPayload)
    assert first.args == {"k": "v"}
    assert first.output == "ok"
    assert second.args == {"x": 1}
    assert second.output == "done"


def test_conversation_turn_accepts_agent_role_aliases() -> None:
    s = ConversationalSample.model_validate(
        {
            "turns": [
                {"role": "human", "content": "Q?"},
                {"role": "ai", "content": "A."},
            ]
        }
    )
    assert s.turns[0].role == "user"
    assert s.turns[1].role == "assistant"


def test_conversation_turn_accepts_tool_calls_alias() -> None:
    s = ConversationalSample.model_validate(
        {
            "turns": [
                {"role": "user", "content": "Q?"},
                {
                    "role": "assistant",
                    "content": "A.",
                    "tool_calls": [{"name": "lookup", "args": {"id": 1}}],
                },
            ]
        }
    )
    assert s.turns[1].tools_called is not None
    tool_call = s.turns[1].tools_called[0]
    assert isinstance(tool_call, ToolCallPayload)
    assert tool_call.name == "lookup"
    assert tool_call.args == {"id": 1}


def test_conversational_sample_accepts_messages_alias_for_turns() -> None:
    s = ConversationalSample.model_validate(
        {
            "messages": [
                {"role": "user", "content": "Q?"},
                {"role": "assistant", "content": "A."},
            ]
        }
    )
    assert len(s.turns) == 2


def test_conversational_sample_accepts_trace_messages_shape() -> None:
    s = ConversationalSample.model_validate(
        {
            "trace": {
                "messages": [
                    {"role": "human", "content": "Q?"},
                    {"role": "ai", "content": "A."},
                ]
            }
        }
    )
    assert len(s.turns) == 2
    assert s.turns[0].role == "user"
    assert s.turns[1].role == "assistant"
