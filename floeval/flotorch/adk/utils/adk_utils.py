"""Utility functions for FloTorch ADK operations."""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def normalize_mcp_schema(schema: Dict[str, Any]) -> None:
    """Normalize MCP JSON schema in-place for GenAI compatibility."""
    if isinstance(schema, dict) and schema.get("type") == "object":
        if schema.get("properties") is None:
            schema["properties"] = {}
        if schema.get("required") is None:
            schema["required"] = []


def tools_to_openai_format(tools: Any) -> List[Dict[str, Any]]:
    """Convert tools to OpenAI format for LLM requests."""
    result: List[Dict[str, Any]] = []
    for tool in tools:
        try:
            name = getattr(
                tool,
                "name",
                getattr(getattr(tool, "func", tool), "__name__", str(tool)),
            )
            description = getattr(
                tool,
                "description",
                getattr(getattr(tool, "func", tool), "__doc__", "")
                or getattr(tool, "__doc__", ""),
            )
            parameters: Dict[str, Any] = {
                "type": "object",
                "properties": {},
                "required": [],
            }

            if hasattr(tool, "_get_declaration"):
                try:
                    decl = tool._get_declaration()
                    if decl and hasattr(decl, "parameters_json_schema") and decl.parameters_json_schema:
                        parameters = (
                            decl.parameters_json_schema.copy()
                            if isinstance(decl.parameters_json_schema, dict)
                            else {}
                        )
                        normalize_mcp_schema(parameters)
                    elif decl and decl.parameters and hasattr(decl.parameters, "properties"):
                        properties = {}
                        for prop_name, prop_schema in decl.parameters.properties.items():
                            prop_type = getattr(prop_schema, "type", None)
                            if prop_type and hasattr(prop_type, "value"):
                                if prop_type.value == "INTEGER":
                                    properties[prop_name] = {"type": "integer"}
                                elif prop_type.value == "NUMBER":
                                    properties[prop_name] = {"type": "number"}
                                elif prop_type.value == "BOOLEAN":
                                    properties[prop_name] = {"type": "boolean"}
                                elif prop_type.value == "ARRAY":
                                    properties[prop_name] = {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    }
                                else:
                                    properties[prop_name] = {"type": "string"}
                            else:
                                properties[prop_name] = {"type": "string"}
                        parameters["properties"] = properties
                        parameters["required"] = getattr(decl.parameters, "required", [])
                        normalize_mcp_schema(parameters)
                except Exception:
                    pass
            elif hasattr(tool, "input_schema"):
                schema = tool.input_schema
                if hasattr(schema, "properties"):
                    parameters["properties"] = {
                        n: {"type": "string"} for n in schema.properties
                    }
                    parameters["required"] = getattr(schema, "required", [])
                    normalize_mcp_schema(parameters)
                elif isinstance(schema, dict):
                    parameters = schema.copy()
                    normalize_mcp_schema(parameters)
            elif hasattr(tool, "func") or hasattr(tool, "__call__"):
                sig = inspect.signature(getattr(tool, "func", tool))
                parameters["properties"] = {
                    n: {"type": "string"} for n in sig.parameters
                }
                parameters["required"] = list(sig.parameters.keys())
                normalize_mcp_schema(parameters)

            result.append({
                "type": "function",
                "function": {"name": name, "description": description, "parameters": parameters},
            })
        except Exception as e:
            logger.warning("Failed to convert tool to OpenAI format: %s", e)
    return result


def parse_function_response(response_content: Any) -> str:
    """Parse function response content to extract text."""
    try:
        if hasattr(response_content, "result") and hasattr(response_content.result, "content"):
            content_list = response_content.result.content
            return (
                content_list[0].text
                if content_list and hasattr(content_list[0], "text")
                else str(response_content.result)
            )
        if hasattr(response_content, "content"):
            content_list = response_content.content
            return (
                content_list[0].text
                if content_list and hasattr(content_list[0], "text")
                else str(response_content)
            )
        if isinstance(response_content, dict):
            if "content" in response_content and isinstance(response_content["content"], list):
                content_list = response_content["content"]
                return (
                    content_list[0]["text"]
                    if content_list
                    and isinstance(content_list[0], dict)
                    and "text" in content_list[0]
                    else str(response_content)
                )
            return json.dumps(response_content) if isinstance(response_content, dict) else str(response_content)
        return str(response_content)
    except Exception:
        return str(response_content)


def parse_llm_response_with_tools(response_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse LLM response that may contain tool calls; return ADK-compatible parts."""
    try:
        if "choices" not in response_data or not response_data["choices"]:
            return []
        choice = response_data["choices"][0]
        parts: List[Dict[str, Any]] = []

        if "message" in choice:
            msg = choice["message"]
            if "tool_calls" in msg and msg["tool_calls"]:
                tool_call = msg["tool_calls"][0]
                try:
                    args_raw = tool_call["function"].get("arguments", "{}")
                    fn_args = (
                        json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    )
                    parts.append({
                        "type": "function_call",
                        "name": tool_call["function"]["name"],
                        "args": fn_args,
                        "id": tool_call.get("id", f"call_{tool_call['function']['name']}"),
                    })
                except Exception as e:
                    logger.warning("Failed to parse tool call arguments: %s", e)
                    parts.append({"type": "text", "content": "Let me help you with that query."})
            elif "function_call" in msg:
                fn_call = msg["function_call"]
                try:
                    args_raw = fn_call.get("arguments", "{}")
                    fn_args = (
                        json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    )
                    parts.append({
                        "type": "function_call",
                        "name": fn_call["name"],
                        "args": fn_args,
                        "id": f"call_{fn_call['name']}",
                    })
                except Exception as e:
                    logger.warning("Failed to parse function call arguments: %s", e)
                    parts.append({"type": "text", "content": "Let me help you with that query."})
            elif "content" in msg and msg["content"]:
                parts.append({"type": "text", "content": msg["content"]})
        elif "text" in choice:
            parts.append({"type": "text", "content": choice["text"]})

        return parts
    except Exception as e:
        logger.error("parse_llm_response_with_tools failed: %s", e)
        return []


def _event_role_to_message_role(author: str) -> str:
    """Map ADK event author to message role for AgentTrace.

    ADK uses author='user' for user, author=agent_name for model (e.g. 'eval_agent').
    AgentTrace expects 'user'|'assistant'|'tool'; model text must map to 'assistant'.
    """
    if not author or author == "user":
        return "user"
    return "assistant"


def process_session_events(session_events: Any) -> List[Dict[str, Any]]:
    """Process ADK session events and convert to message format."""
    messages: List[Dict[str, Any]] = []
    try:
        if not session_events:
            return messages
        recent = session_events[-50:] if len(session_events) > 50 else session_events
        for event in recent:
            if not event.content or not event.content.parts:
                continue
            author = getattr(event, "author", "user") or "user"
            role = _event_role_to_message_role(author)

            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    messages.append({"role": role, "content": part.text})
                elif hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    args = getattr(fc, "args", {})
                    args_str = (
                        json.dumps(args)
                        if isinstance(args, dict)
                        else (args if isinstance(args, str) else "{}")
                    )
                    messages.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": getattr(fc, "id", ""),
                            "type": "function",
                            "function": {"name": getattr(fc, "name", ""), "arguments": args_str},
                        }],
                    })
                elif hasattr(part, "tool_calls") and part.tool_calls:
                    for tc in part.tool_calls:
                        fn = getattr(tc, "function", None) or tc
                        name = getattr(fn, "name", "")
                        args = getattr(fn, "args", {})
                        args_str = (
                            json.dumps(args)
                            if isinstance(args, dict)
                            else (args if isinstance(args, str) else "{}")
                        )
                        messages.append({
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [{
                                "id": getattr(tc, "id", ""),
                                "type": "function",
                                "function": {"name": name, "arguments": args_str},
                            }],
                        })
                elif hasattr(part, "function_response") and part.function_response:
                    fr = part.function_response
                    resp = getattr(fr, "response", None)
                    content = json.dumps(resp) if isinstance(resp, dict) else str(resp)
                    messages.append({
                        "role": "tool",
                        "content": content,
                        "tool_call_id": getattr(fr, "id", ""),
                    })
    except Exception:
        pass
    return messages


def process_content_parts(content: Any) -> List[Dict[str, Any]]:
    """Process content parts and convert to message format."""
    messages: List[Dict[str, Any]] = []
    try:
        if hasattr(content, "role") and hasattr(content, "parts"):
            for part in content.parts:
                if hasattr(part, "text") and part.text:
                    messages.append({"role": content.role, "content": part.text})
                elif hasattr(part, "function_call") and part.function_call:
                    messages.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": part.function_call.id,
                            "type": "function",
                            "function": {
                                "name": part.function_call.name,
                                "arguments": json.dumps(part.function_call.args),
                            },
                        }],
                    })
                elif hasattr(part, "function_response") and part.function_response:
                    response_text = parse_function_response(part.function_response.response)
                    messages.append({
                        "role": "tool",
                        "content": response_text,
                        "tool_call_id": part.function_response.id,
                    })
    except Exception:
        pass
    return messages


def build_messages_from_request(llm_request: Any) -> List[Dict[str, Any]]:
    """Build messages from LLM request including system instruction, session events, and content."""
    messages: List[Dict[str, Any]] = []

    if hasattr(llm_request, "config") and getattr(llm_request.config, "system_instruction", None):
        messages.append({
            "role": "system",
            "content": llm_request.config.system_instruction,
        })

    try:
        ctx = getattr(llm_request, "_invocation_context", None)
        if ctx and (session := getattr(ctx, "session", None)) and session.events:
            messages.extend(process_session_events(session.events))
    except Exception:
        pass

    for content in getattr(llm_request, "contents", []):
        messages.extend(process_content_parts(content))

    return messages
