"""Map Floeval conversational rows to RAGAS ``MultiTurnSample``."""

from ragas import MultiTurnSample
from ragas.messages import AIMessage, HumanMessage, ToolCall as RAGASToolCall, ToolMessage

from floeval.config.schemas.io.conversation import ToolCallPayload, ToolCallPayloadInput
from floeval.config.schemas.io.conversational_dataset import ConversationalSample


def _normalize_tool(tool: ToolCallPayloadInput) -> ToolCallPayload:
    return tool if isinstance(tool, ToolCallPayload) else ToolCallPayload.model_validate(tool)


def _to_ragas_tool_call(tool: ToolCallPayloadInput) -> RAGASToolCall:
    tc = _normalize_tool(tool)
    return RAGASToolCall(name=tc.name, args=tc.args)


def conversational_sample_to_ragas_multiturn(
    sample: ConversationalSample,
) -> MultiTurnSample:
    """Convert Floeval ``ConversationalSample`` turns to RAGAS message list + metadata."""
    messages: list[HumanMessage | AIMessage | ToolMessage] = []
    for turn in sample.turns:
        if turn.role == "user":
            messages.append(HumanMessage(content=turn.content))
            continue
        tools_raw = turn.tools_called or []
        if not tools_raw:
            messages.append(AIMessage(content=turn.content))
            continue
        ragas_calls: list[RAGASToolCall] = []
        for tr in tools_raw:
            ragas_calls.append(_to_ragas_tool_call(tr))
        messages.append(AIMessage(content=turn.content, tool_calls=ragas_calls))
        for tr in tools_raw:
            tc = _normalize_tool(tr)
            messages.append(ToolMessage(content=tc.output or ""))

    ref_tools: list[RAGASToolCall] | None = None
    if sample.reference_tool_calls:
        ref_tools = []
        for tr in sample.reference_tool_calls:
            ref_tools.append(_to_ragas_tool_call(tr))

    return MultiTurnSample(
        user_input=messages,
        reference=sample.reference_outcome,
        reference_topics=sample.reference_topics,
        reference_tool_calls=ref_tools,
        rubrics=sample.rubrics,
    )
