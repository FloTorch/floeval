"""Agent evaluation domain models (pure domain layer - no infrastructure)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

AgentInputOutput = str | dict[str, Any]


def _to_display_str(val: str | dict[str, Any] | None) -> str:
    """Convert user_input or reference_outcome to string for display/prompts."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    return json.dumps(val)


class ToolCall(BaseModel):
    """Tool/function invocation."""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class MessageBase(BaseModel):
    """Base for all message types."""

    role: str
    content: str = ""


class HumanMessage(MessageBase):
    """User message."""

    role: Literal["human"] = "human"


class AIMessage(MessageBase):
    """AI/model message."""

    role: Literal["ai"] = "ai"
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ToolMessage(MessageBase):
    """Tool execution result."""

    role: Literal["tool"] = "tool"
    tool_name: str = ""
    tool_call_id: str | None = None


AgentMessage = HumanMessage | AIMessage | ToolMessage


class AgentTrace(BaseModel):
    """Immutable trace of agent execution.

    Value object - represents what happened, not how to do it.
    """

    messages: list[AgentMessage] = Field(..., min_length=1)
    final_response: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}

    @classmethod
    def from_simple_response(
        cls, user_input: str, response: str, **metadata: Any
    ) -> AgentTrace:
        """Factory: Create trace from simple input/output."""
        return cls(
            messages=[
                HumanMessage(content=user_input),
                AIMessage(content=response),
            ],
            final_response=response,
            metadata=metadata,
        )

    @classmethod
    def from_messages(cls, messages: list[dict[str, Any]], **metadata: Any) -> AgentTrace:
        """Create trace from message list (e.g. from process_session_events).

        Expects messages with keys: role ("user"|"assistant"|"tool"), content,
        optional tool_calls, tool_call_id, tool_name.
        Treats any non-user, non-tool role as assistant (ADK may use agent name).
        """
        agent_messages: list[AgentMessage] = []
        final_response = ""

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "") or ""
            if role != "user" and role != "tool":
                role = "assistant"

            if role == "user":
                agent_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                tool_calls_data = msg.get("tool_calls", [])
                tool_calls = []
                for tc in tool_calls_data:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    if isinstance(fn, dict):
                        name = fn.get("name", "")
                        args_str = fn.get("arguments", "{}")
                        args = (
                            args_str 
                            if isinstance(args_str, dict) 
                            else _safe_json_loads(args_str)
                        )
                        tool_calls.append(ToolCall(name=name, args=args or {}))
                agent_messages.append(
                    AIMessage(content=content, tool_calls=tool_calls)
                )
                if content:
                    final_response = content
            elif role == "tool":
                tool_name = msg.get("tool_name", "")
                tool_call_id = msg.get("tool_call_id")
                agent_messages.append(
                    ToolMessage(
                        content=content,
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                    )
                )

        if not final_response:
            for m in reversed(agent_messages):
                if isinstance(m, AIMessage) and m.content:
                    final_response = m.content
                    break
        if not final_response and agent_messages:
            last = agent_messages[-1]
            if isinstance(last, ToolMessage) and last.content:
                final_response = f"[Tool result: {last.content[:500]}...]" if len(last.content) > 500 else f"[Tool result: {last.content}]"

        return cls(
            messages=agent_messages,
            final_response=final_response,
            metadata=metadata,
        )

    @property
    def tool_calls_made(self) -> list[ToolCall]:
        """Derived property: all tool calls in trace."""
        return [
            tc 
            for msg in self.messages 
            if isinstance(msg, AIMessage) 
            for tc in msg.tool_calls
        ]

    @property
    def turn_count(self) -> int:
        """Derived property: number of AI turns."""
        return sum(1 for m in self.messages if isinstance(m, AIMessage))


def _safe_json_loads(s: str) -> dict:
    """Safely parse JSON string to dict."""
    try:
        out = json.loads(s)
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


class PartialAgentSample(BaseModel):
    """Test case before agent execution."""

    user_input: AgentInputOutput
    reference_outcome: AgentInputOutput | None = None
    reference_tool_calls: list[ToolCall] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentSample(BaseModel):
    """Test case after agent execution (has trace)."""

    user_input: AgentInputOutput
    trace: AgentTrace
    reference_outcome: AgentInputOutput | None = None
    reference_tool_calls: list[ToolCall] | None = None
    agent_traces: list[AgentTrace] | None = None  # One per agent, in execution order (workflow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_partial(
        cls, partial: PartialAgentSample, trace: AgentTrace
    ) -> AgentSample:
        """Factory: Convert partial + trace to full sample."""
        return cls(
            user_input=partial.user_input,
            trace=trace,
            reference_outcome=partial.reference_outcome,
            reference_tool_calls=partial.reference_tool_calls,
            metadata={**partial.metadata, **trace.metadata},
        )


class AgentDataset(BaseModel):
    """Collection of agent samples."""

    samples: list[AgentSample | PartialAgentSample] = Field(..., min_length=1)

    @property
    def is_partial(self) -> bool:
        """True if any sample lacks trace."""
        return any(isinstance(s, PartialAgentSample) for s in self.samples)

    @property
    def all_full(self) -> list[AgentSample]:
        """Only complete samples."""
        return [s for s in self.samples if isinstance(s, AgentSample)]

    @property
    def all_partial(self) -> list[PartialAgentSample]:
        """Only partial samples."""
        return [s for s in self.samples if isinstance(s, PartialAgentSample)]

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)

    @classmethod
    def from_file(cls, path: str | Path) -> AgentDataset:
        """Load dataset from JSON or JSONL file. Delegates to AgentDatasetLoader."""
        from floeval.api.dataset_loaders.agent_file_loader import (
            AgentDatasetLoader,
        )

        return AgentDatasetLoader.from_file(path)
