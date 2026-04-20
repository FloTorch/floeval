"""Thread-safe trace context using contextvars."""

import contextvars

from floeval.config.schemas.io.agent_dataset import (
    AgentMessage,
    AgentTrace,
    AIMessage,
    HumanMessage,
    ToolCall,
    ToolMessage,
)

_current_trace: contextvars.ContextVar["TraceContext | None"] = contextvars.ContextVar(
    "floeval_trace", default=None
)


class TraceContext:
    """Active trace being captured."""

    def __init__(self, user_input: str):
        self.user_input = user_input
        self.messages: list[AgentMessage] = [HumanMessage(content=user_input)]
        self.final_response: str = ""
        self.metadata: dict = {}

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

    def to_trace(self) -> AgentTrace:
        """Convert to immutable trace."""
        return AgentTrace(
            messages=self.messages,
            final_response=self.final_response,
            metadata=self.metadata,
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
