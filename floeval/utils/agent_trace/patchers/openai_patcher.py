"""Patch OpenAI client for trace capture."""

import json
import logging

from openai.types import CompletionUsage

from floeval.config.schemas.io.agent_dataset import ToolCall
from floeval.utils.agent_trace.trace_context import get_current_trace

logger = logging.getLogger(__name__)

_original_create = None
_original_async_create = None
_patched = False


def _safe_json_loads(s: str) -> dict:
    """Safely parse JSON string to dict."""
    try:
        out = json.loads(s)
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def _extract_tool_calls(msg) -> list[ToolCall]:
    """Extract tool calls from OpenAI message."""
    tool_calls = []
    if not hasattr(msg, "tool_calls") or not msg.tool_calls:
        return tool_calls
    for tc in msg.tool_calls:
        fn = getattr(tc, "function", None)
        if fn is None:
            continue
        name = getattr(fn, "name", "") or ""
        args_raw = getattr(fn, "arguments", "{}")
        args = (
            _safe_json_loads(args_raw)
            if isinstance(args_raw, str)
            else (args_raw if isinstance(args_raw, dict) else {})
        )
        tool_calls.append(ToolCall(name=name, args=args))
    return tool_calls


def _capture_response(response) -> None:
    """Capture LLM response (content + token usage) into trace context."""
    trace = get_current_trace()
    if trace is None:
        return
    try:
        if not hasattr(response, "choices") or not response.choices:
            return
        choice = response.choices[0]
        msg = getattr(choice, "message", None)
        if msg is None:
            return
        content = getattr(msg, "content", None) or ""
        tool_calls = _extract_tool_calls(msg)
        trace.log_ai_turn(
            content=content if isinstance(content, str) else str(content),
            tool_calls=tool_calls if tool_calls else None,
        )
        completion_usage = getattr(response, "usage", None)
        if isinstance(completion_usage, CompletionUsage):
            trace.add_token_usage(
                total_tokens=completion_usage.total_tokens,
                prompt_tokens=completion_usage.prompt_tokens,
                completion_tokens=completion_usage.completion_tokens,
            )
    except Exception as e:
        logger.debug("OpenAI patcher capture failed: %s", e)


def patch_openai() -> None:
    """Monkey-patch OpenAI client for trace capture."""
    global _original_create, _original_async_create, _patched

    if _patched:
        return

    try:
        from openai.resources.chat.completions import AsyncCompletions, Completions
    except ImportError:
        logger.warning("OpenAI not installed - auto-capture unavailable")
        return

    if not hasattr(Completions, "create"):
        logger.warning("OpenAI Completions.create not found - skip patching")
        return

    _original_create = Completions.create
    _original_async_create = getattr(AsyncCompletions, "create", None)

    def traced_create(self, *args, **kwargs):
        response = _original_create(self, *args, **kwargs)
        _capture_response(response)
        return response

    async def traced_async_create(self, *args, **kwargs):
        response = await _original_async_create(self, *args, **kwargs)
        _capture_response(response)
        return response

    Completions.create = traced_create
    if _original_async_create is not None:
        AsyncCompletions.create = traced_async_create
    _patched = True


def unpatch_openai() -> None:
    """Restore original OpenAI client."""
    global _patched, _original_create, _original_async_create

    if not _patched:
        return

    try:
        from openai.resources.chat.completions import AsyncCompletions, Completions

        Completions.create = _original_create
        if _original_async_create is not None:
            AsyncCompletions.create = _original_async_create
        _patched = False
    except ImportError:
        _patched = False
