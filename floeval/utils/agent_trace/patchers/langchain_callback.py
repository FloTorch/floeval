"""LangChain callback for trace capture."""

import logging
from typing import Any

from floeval.config.schemas.io.agent_dataset import ToolCall
from floeval.utils.agent_trace.trace_context import get_current_trace

logger = logging.getLogger(__name__)

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseCallbackHandler = object  # type: ignore[misc, assignment]
    LLMResult = object  # type: ignore[misc, assignment]


class FloevalLangChainCallback(BaseCallbackHandler):
    """LangChain callback that logs to trace context."""

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Called when LLM completes."""
        trace = get_current_trace()
        if trace is None:
            return

        if not hasattr(response, "generations") or not response.generations:
            return
        if not response.generations[0]:
            return

        gen = response.generations[0][0]
        content = getattr(gen, "text", str(gen))

        tool_calls = []
        if hasattr(gen, "message") and hasattr(gen.message, "tool_calls"):
            for tc in gen.message.tool_calls:
                name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                tool_calls.append(ToolCall(name=name, args=args or {}))

        trace.log_ai_turn(content, tool_calls if tool_calls else None)

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        """Called when tool completes."""
        trace = get_current_trace()
        if trace is None:
            return

        tool_name = kwargs.get("name", "unknown")
        trace.log_tool_result(tool_name, str(output))


_callback_instance: FloevalLangChainCallback | None = (
    FloevalLangChainCallback() if LANGCHAIN_AVAILABLE else None
)


def get_langchain_callback() -> FloevalLangChainCallback:
    """Get global callback instance."""
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain not installed - install langchain-core")
    assert _callback_instance is not None
    return _callback_instance


def is_langchain_available() -> bool:
    """Check if LangChain is installed."""
    return LANGCHAIN_AVAILABLE
