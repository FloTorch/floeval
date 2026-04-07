"""Map Floeval conversational rows to DeepEval ``ConversationalTestCase``."""

from typing import Any

from deepeval.test_case import (
    ConversationalTestCase,
    ToolCall as DeepEvalToolCall,
    Turn,
)

from floeval.config.schemas.io.agent_dataset import (
    AgentSample,
    AIMessage,
    HumanMessage,
    ToolMessage,
    _to_display_str,
)
from floeval.config.schemas.io.conversational_dataset import ConversationalSample


def conversational_sample_to_deepeval(
    sample: ConversationalSample,
) -> ConversationalTestCase:
    """Build DeepEval ``ConversationalTestCase`` from ``ConversationalSample``."""
    deepeval_turns: list[Turn] = []
    for t in sample.turns:
        turn_kw: dict[str, object] = {"role": t.role, "content": t.content}
        if t.retrieval_context is not None:
            turn_kw["retrieval_context"] = t.retrieval_context
        if t.tools_called is not None:
            turn_kw["tools_called"] = t.tools_called
        deepeval_turns.append(Turn(**turn_kw))

    return ConversationalTestCase(
        turns=deepeval_turns,
        scenario=sample.scenario,
        expected_outcome=sample.expected_outcome,
        user_description=sample.user_description,
        chatbot_role=sample.chatbot_role,
        context=sample.conversation_context,
    )


def agent_sample_to_conversational_test_case(
    sample: AgentSample,
) -> ConversationalTestCase:
    """Map agent trace (human/ai/tool) to DeepEval turns (user/assistant, inline tools)."""
    msgs = sample.trace.messages
    deepeval_turns: list[Turn] = []
    i = 0
    while i < len(msgs):
        m = msgs[i]
        if isinstance(m, HumanMessage):
            deepeval_turns.append(Turn(role="user", content=m.content))
            i += 1
            continue
        if isinstance(m, ToolMessage):
            i += 1
            continue
        am = m
        if not isinstance(am, AIMessage):
            i += 1
            continue
        if not am.tool_calls:
            deepeval_turns.append(Turn(role="assistant", content=am.content))
            i += 1
            continue
        deepeval_tools: list[DeepEvalToolCall] = []
        k = i + 1
        for tc in am.tool_calls:
            output_val: Any = None
            if k < len(msgs) and isinstance(msgs[k], ToolMessage):
                output_val = msgs[k].content
                k += 1
            deepeval_tools.append(
                DeepEvalToolCall(
                    name=tc.name,
                    input_parameters=tc.args,
                    output=output_val,
                )
            )
        deepeval_turns.append(
            Turn(role="assistant", content=am.content, tools_called=deepeval_tools)
        )
        i = k

    ref_out = _to_display_str(sample.reference_outcome) or None
    return ConversationalTestCase(
        turns=deepeval_turns,
        scenario=sample.scenario,
        expected_outcome=sample.expected_outcome or ref_out,
        user_description=sample.user_description,
        chatbot_role=sample.chatbot_role,
        context=sample.conversation_context,
    )


def conversational_row_to_deepeval(
    sample: ConversationalSample | AgentSample,
) -> ConversationalTestCase:
    """Dispatch conversational vs agent-native row to DeepEval test case."""
    if isinstance(sample, ConversationalSample):
        return conversational_sample_to_deepeval(sample)
    return agent_sample_to_conversational_test_case(sample)


def sample_to_conversational_test_case(
    sample: ConversationalSample | AgentSample,
) -> ConversationalTestCase:
    """Backward-compatible name for tests and callers."""
    return conversational_row_to_deepeval(sample)
