"""ADK helpers: OpenAI-shaped messages, tool parsing, session → trace messages."""

import json
import inspect
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def normalize_mcp_schema(schema: Dict[str, Any]) -> None:
    """Fix None values in required/properties so GenAI validation doesn't choke."""
    if isinstance(schema, dict) and schema.get("type") == "object":
        if schema.get("properties") is None:
            schema["properties"] = {}
        if schema.get("required") is None:
            schema["required"] = []


def tools_to_openai_format(tools):
    """Convert ADK tools to the OpenAI tools array format."""
    result = []
    for tool in tools:
        try:
            name = getattr(tool, 'name', getattr(tool, 'func', tool).__name__ if hasattr(getattr(tool, 'func', tool), '__name__') else str(tool))
            description = getattr(tool, 'description', getattr(getattr(tool, 'func', tool), '__doc__', '') or getattr(tool, '__doc__', ''))

            parameters = {"type": "object", "properties": {}, "required": []}

            if hasattr(tool, '_get_declaration'):
                try:
                    decl = tool._get_declaration()
                    if decl and hasattr(decl, 'parameters_json_schema') and decl.parameters_json_schema:
                        parameters = decl.parameters_json_schema.copy() if isinstance(decl.parameters_json_schema, dict) else {}
                        normalize_mcp_schema(parameters)
                    elif decl and decl.parameters and hasattr(decl.parameters, 'properties'):
                        properties = {}
                        for prop_name, prop_schema in decl.parameters.properties.items():
                            prop_type = getattr(prop_schema, 'type', None)
                            if prop_type and hasattr(prop_type, 'value'):
                                if prop_type.value == 'INTEGER': properties[prop_name] = {"type": "integer"}
                                elif prop_type.value == 'NUMBER': properties[prop_name] = {"type": "number"}
                                elif prop_type.value == 'BOOLEAN': properties[prop_name] = {"type": "boolean"}
                                elif prop_type.value == 'ARRAY': properties[prop_name] = {"type": "array", "items": {"type": "string"}}
                                else: properties[prop_name] = {"type": "string"}
                            else: properties[prop_name] = {"type": "string"}
                        parameters["properties"], parameters["required"] = properties, getattr(decl.parameters, 'required', [])
                        normalize_mcp_schema(parameters)
                except: pass
            elif hasattr(tool, 'input_schema'):
                schema = tool.input_schema
                if hasattr(schema, 'properties'):
                    parameters["properties"] = {name: {"type": "string"} for name in schema.properties}
                    parameters["required"] = getattr(schema, 'required', [])
                    normalize_mcp_schema(parameters)
                elif isinstance(schema, dict):
                    parameters = schema.copy()
                    normalize_mcp_schema(parameters)
            elif hasattr(tool, 'func') or hasattr(tool, '__call__'):
                sig = inspect.signature(getattr(tool, 'func', tool))
                parameters["properties"] = {name: {"type": "string"} for name in sig.parameters}
                parameters["required"] = list(sig.parameters.keys())
                normalize_mcp_schema(parameters)

            result.append({"type": "function", "function": {"name": name, "description": description, "parameters": parameters}})
        except Exception as e:
            logger.warning("Failed to convert tool to OpenAI format: %s", e)
            continue
    return result


def parse_function_response(response_content):
    """Extract text from tool output (joins multiple MCP content items)."""
    try:
        if hasattr(response_content, 'result') and hasattr(response_content.result, 'content'):
            content_list = response_content.result.content
            if content_list:
                texts = [item.text for item in content_list if hasattr(item, 'text') and item.text]
                return "\n".join(texts) if texts else str(response_content.result)
            return str(response_content.result)
        elif hasattr(response_content, 'content'):
            content_list = response_content.content
            if content_list and hasattr(content_list[0], 'text'):
                texts = [item.text for item in content_list if hasattr(item, 'text') and item.text]
                return "\n".join(texts) if texts else str(response_content)
            return str(response_content)
        elif isinstance(response_content, dict):
            if 'content' in response_content and isinstance(response_content['content'], list):
                content_list = response_content['content']
                texts = [
                    item['text'] for item in content_list
                    if isinstance(item, dict) and 'text' in item and item['text']
                ]
                return "\n".join(texts) if texts else str(response_content)
            else:
                return json.dumps(response_content) if isinstance(response_content, dict) else str(response_content)
        else:
            return str(response_content)
    except:
        return str(response_content)


def parse_llm_response_with_tools(response_data):
    """Parse the raw LLM response JSON into a list of ADK-compatible part dicts."""
    try:
        if 'choices' in response_data and response_data['choices']:
            choice = response_data['choices'][0]
            parts = []

            if "message" in choice:
                msg = choice["message"]
                if "tool_calls" in msg and msg["tool_calls"]:
                    for tool_call in msg["tool_calls"]:
                        try:
                            fn_args = json.loads(tool_call["function"].get("arguments", "{}")) if isinstance(tool_call["function"].get("arguments", "{}"), str) else tool_call["function"].get("arguments", {})
                            parts.append({
                                "type": "function_call",
                                "name": tool_call["function"]["name"],
                                "args": fn_args,
                                "id": tool_call.get("id", f"call_{tool_call['function']['name']}")
                            })
                        except Exception as e:
                            logger.warning("Failed to parse tool call arguments: %s", e)
                            parts.append({"type": "text", "content": "Let me help you with that query."})
                            break
                elif "function_call" in msg:
                    fn_call = msg["function_call"]
                    try:
                        fn_args = json.loads(fn_call.get("arguments", "{}")) if isinstance(fn_call.get("arguments", "{}"), str) else fn_call.get("arguments", {})
                        parts.append({
                            "type": "function_call",
                            "name": fn_call["name"],
                            "args": fn_args,
                            "id": f"call_{fn_call['name']}"
                        })
                    except Exception as e:
                        logger.warning("Failed to parse function call arguments: %s", e)
                        parts.append({"type": "text", "content": "Let me help you with that query."})
                elif "content" in msg and msg["content"]:
                    parts.append({"type": "text", "content": msg["content"]})
            elif "text" in choice:
                parts.append({"type": "text", "content": choice["text"]})

            return parts
        return []
    except Exception as e:
        logger.error("parse_llm_response_with_tools failed: %s", e)
        return []


def process_session_events(session_events):
    """Session events → flat messages (for AgentTrace after a run)."""
    messages = []
    if not session_events:
        return messages
    for event in session_events:
        if not (event.content and event.content.parts):
            continue
        for part in event.content.parts:
            try:
                if hasattr(part, 'text') and part.text:
                    role = "user" if event.author == "user" else "assistant"
                    messages.append({"role": role, "content": part.text})
                    continue
                if hasattr(part, 'function_call') and part.function_call:
                    raw_args = part.function_call.args
                    if isinstance(raw_args, dict):
                        args_json = json.dumps(raw_args)
                    else:
                        try:
                            args_json = json.dumps(dict(raw_args))
                        except Exception:
                            args_json = str(raw_args or "{}")
                    messages.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": part.function_call.id,
                            "type": "function",
                            "function": {
                                "name": part.function_call.name,
                                "arguments": args_json
                            }
                        }]
                    })
                    continue
                if hasattr(part, 'function_response') and part.function_response:
                    resp = part.function_response.response
                    try:
                        content_text = json.dumps(resp)
                    except Exception:
                        content_text = str(resp)
                    messages.append({
                        "role": "tool",
                        "content": content_text,
                        "tool_call_id": part.function_response.id,
                        "tool_name": part.function_response.name or "",
                    })
            except Exception as e:
                print(f"[floeval-debug] process_session_events part parse failed: {e}", flush=True)
                try:
                    import traceback
                    traceback.print_exc()
                except Exception:
                    pass
    return messages


def process_content_parts(content):
    """One ADK Content → OpenAI-style message dicts (model role → assistant)."""
    messages = []
    try:
        if hasattr(content, "role") and hasattr(content, "parts"):
            for part in content.parts:
                if hasattr(part, "text") and part.text:
                    role = "assistant" if content.role == "model" else content.role
                    messages.append({"role": role, "content": part.text})
                elif hasattr(part, "function_call") and part.function_call:
                    messages.append({
                        "role": "assistant", "content": "", "tool_calls": [{
                            "id": part.function_call.id, "type": "function",
                            "function": {"name": part.function_call.name, "arguments": json.dumps(part.function_call.args or {})}
                        }]
                    })
                elif hasattr(part, "function_response") and part.function_response:
                    response_text = parse_function_response(part.function_response.response)
                    messages.append({"role": "tool", "content": response_text, "tool_call_id": part.function_response.id})
    except: pass
    return messages


def build_messages_from_request(llm_request):
    """Messages for this LLM call: prefer llm_request.contents; else session events."""
    messages = []

    if hasattr(llm_request, "config") and getattr(llm_request.config, "system_instruction", None):
        messages.append({"role": "system", "content": llm_request.config.system_instruction})

    contents = getattr(llm_request, "contents", [])

    if contents:
        for content in contents:
            messages.extend(process_content_parts(content))
    else:
        try:
            if (ctx := getattr(llm_request, '_invocation_context', None)) and (session := ctx.session) and session.events:
                messages.extend(process_session_events(session.events))
        except: pass

    return messages
