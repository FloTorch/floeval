"""Tests for ConversationalSample → RAGAS MultiTurnSample."""

from ragas.messages import AIMessage, HumanMessage, ToolMessage

from floeval.config.schemas.io.conversation import ConversationTurn
from floeval.config.schemas.io.conversational_dataset import ConversationalSample
from floeval.metric_providers.ragas.multiturn_adapter import (
    conversational_sample_to_ragas_multiturn,
)


def test_conversational_to_ragas_messages_no_tools() -> None:
    s = ConversationalSample(
        turns=[
            ConversationTurn(role="user", content="Hi"),
            ConversationTurn(role="assistant", content="Hello"),
        ],
        reference_topics=["greeting"],
    )
    m = conversational_sample_to_ragas_multiturn(s)
    assert len(m.user_input) == 2
    assert isinstance(m.user_input[0], HumanMessage)
    assert isinstance(m.user_input[1], AIMessage)
    assert m.reference_topics == ["greeting"]


def test_conversational_to_ragas_splits_tool_output() -> None:
    s = ConversationalSample(
        turns=[
            ConversationTurn(role="user", content="Search flights"),
            ConversationTurn(
                role="assistant",
                content="Searching.",
                tools_called=[
                    {
                        "name": "search",
                        "args": {"q": "nyc"},
                        "output": "found 2",
                    }
                ],
            ),
        ],
    )
    m = conversational_sample_to_ragas_multiturn(s)
    assert len(m.user_input) == 3
    assert isinstance(m.user_input[0], HumanMessage)
    assert isinstance(m.user_input[1], AIMessage)
    assert m.user_input[1].tool_calls
    assert isinstance(m.user_input[2], ToolMessage)
    assert m.user_input[2].content == "found 2"
