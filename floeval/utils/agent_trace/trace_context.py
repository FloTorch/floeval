"""Thread-safe trace context using contextvars."""

import contextvars
import time
from typing import TypedDict

from floeval.config.schemas.io.agent_dataset import (
    AgentMessage,
    AgentTrace,
    AIMessage,
    HumanMessage,
    ToolCall,
    ToolMessage,
)


class _TokenCounts(TypedDict):
    total: int
    prompt: int
    completion: int

_current_trace: contextvars.ContextVar["TraceContext | None"] = contextvars.ContextVar(
    "floeval_trace", default=None
)

# Separate token accumulator used by FlotorchADKLLM (custom HTTP path).
# Lives independently so it works even when no TraceContext is active.
_session_tokens: contextvars.ContextVar["_TokenCounts | None"] = contextvars.ContextVar(
    "floeval_session_tokens", default=None
)


def start_session_token_tracking() -> None:
    _session_tokens.set(_TokenCounts(total=0, prompt=0, completion=0))


def record_llm_token_usage(total: int = 0, prompt: int = 0, completion: int = 0) -> None:
    """Called by FlotorchADKLLM after each LLM response to accumulate tokens."""
    usage = _session_tokens.get()
    if usage is not None:
        usage["total"] += total
        usage["prompt"] += prompt
        usage["completion"] += completion
    # Also push to TraceContext if the TraceCollector path is active
    trace = _current_trace.get()
    if trace is not None:
        trace.add_token_usage(total, prompt, completion)


def collect_session_token_usage() -> "_TokenCounts | None":
    """Read and clear the accumulated token counts for the current session."""
    usage = _session_tokens.get()
    _session_tokens.set(None)
    return usage


class TraceContext:
    """Active trace being captured."""

    def __init__(self, user_input: str):
        self.user_input = user_input
        self.messages: list[AgentMessage] = [HumanMessage(content=user_input)]
        self.final_response: str = ""
        self.metadata: dict = {}
        self._start_time: float = time.time()
        self._total_tokens: int = 0
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0

    def log_ai_turn(
        self,
        content: str,
        tool_calls: list[ToolCall] | None = None,
    ) -> None:
        """Log AI response."""
        self.messages.append(
            AIMessage(
                content=content or "",
                tool_calls=tool_calls or [],
            )
        )
        if content:
            self.final_response = content

    def log_tool_result(self, tool_name: str, result: str) -> None:
        """Log tool execution result."""
        self.messages.append(
            ToolMessage(
                tool_name=tool_name,
                content=result,
            )
        )

    def add_token_usage(
        self,
        total_tokens: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Accumulate token counts across multiple LLM calls in the same trace."""
        self._total_tokens += total_tokens
        self._prompt_tokens += prompt_tokens
        self._completion_tokens += completion_tokens

    def to_trace(self) -> AgentTrace:
        """Convert to immutable trace, including timing and token usage."""
        return AgentTrace(
            messages=self.messages,
            final_response=self.final_response,
            metadata=self.metadata,
            start_time=self._start_time,
            end_time=time.time(),
            total_tokens=self._total_tokens or None,
            prompt_tokens=self._prompt_tokens or None,
            completion_tokens=self._completion_tokens or None,
        )


def get_current_trace() -> TraceContext | None:
    """Get active trace context."""
    return _current_trace.get()


def set_current_trace(trace: TraceContext) -> contextvars.Token:
    """Set active trace context."""
    return _current_trace.set(trace)


def clear_current_trace() -> None:
    """Clear trace context."""
    _current_trace.set(None)
