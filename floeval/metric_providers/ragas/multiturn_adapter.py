"""Map Floeval conversational rows to RAGAS ``MultiTurnSample``."""

from typing import Any

from ragas import MultiTurnSample
from ragas.messages import AIMessage, HumanMessage, ToolCall as RAGASToolCall, ToolMessage

from floeval.config.schemas.io.conversational_dataset import ConversationalSample


def _tool_args_from_payload(raw: dict[str, Any]) -> dict[str, Any]:
    if "args" in raw and isinstance(raw["args"], dict):
        return raw["args"]
    inp = raw.get("input")
    if isinstance(inp, dict):
        return inp
    ip = raw.get("input_parameters")
    if isinstance(ip, dict):
        return ip
    return {}


def _tool_name_from_payload(raw: dict[str, Any]) -> str:
    name = raw.get("name")
    return str(name) if name is not None else ""


def _tool_output_from_payload(raw: dict[str, Any]) -> str:
    out = raw.get("output")
    if out is None:
        return ""
    return out if isinstance(out, str) else str(out)


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
            if isinstance(tr, dict):
                ragas_calls.append(
                    RAGASToolCall(
                        name=_tool_name_from_payload(tr),
                        args=_tool_args_from_payload(tr),
                    )
                )
            else:
                ragas_calls.append(RAGASToolCall(name="tool", args={}))
        messages.append(AIMessage(content=turn.content, tool_calls=ragas_calls))
        for tr in tools_raw:
            if isinstance(tr, dict):
                messages.append(ToolMessage(content=_tool_output_from_payload(tr)))
            else:
                messages.append(ToolMessage(content=""))

    ref_tools: list[RAGASToolCall] | None = None
    if sample.reference_tool_calls:
        ref_tools = []
        for tr in sample.reference_tool_calls:
            if isinstance(tr, dict):
                ref_tools.append(
                    RAGASToolCall(
                        name=_tool_name_from_payload(tr),
                        args=_tool_args_from_payload(tr),
                    )
                )

    ref = sample.reference
    if ref is None and sample.expected_outcome:
        ref = sample.expected_outcome

    return MultiTurnSample(
        user_input=messages,
        reference=ref,
        reference_topics=sample.reference_topics,
        reference_tool_calls=ref_tools,
        rubrics=sample.rubrics,
    )
