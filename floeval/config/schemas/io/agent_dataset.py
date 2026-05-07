"""Agent evaluation domain models (pure domain layer - no infrastructure)."""

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

    # Performance metadata — all optional, populated by trace capture infrastructure
    start_time: float | None = Field(default=None, description="Unix timestamp, execution start")
    end_time: float | None = Field(default=None, description="Unix timestamp, execution end")
    total_tokens: int | None = Field(default=None, description="Total tokens (prompt + completion)")
    prompt_tokens: int | None = Field(default=None, description="Prompt tokens")
    completion_tokens: int | None = Field(default=None, description="Completion tokens")
    error_info: dict[str, Any] | None = Field(default=None, description="Error details if failed")
    agent_name: str | None = Field(default=None, description="Name of the agent that produced this trace")

    model_config = {"frozen": True}

    @property
    def latency_seconds(self) -> float | None:
        """Wall-clock execution time in seconds. Requires start_time and end_time."""
        if self.start_time is not None and self.end_time is not None:
            return round(self.end_time - self.start_time, 3)
        return None

    @property
    def tool_error_count(self) -> int:
        """Number of tool messages that appear to contain errors (heuristic)."""
        _error_signals = {"error", "exception", "failed", "traceback", "404", "500", "503", "timeout"}
        return sum(
            1 for msg in self.messages
            if isinstance(msg, ToolMessage)
            and any(sig in (msg.content or "").lower() for sig in _error_signals)
        )

    @classmethod
    def from_simple_response(
        cls, user_input: str, response: str, **metadata: Any
    ) -> "AgentTrace":
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
    def from_messages(
        cls, messages: list[dict[str, Any]], **metadata: Any
    ) -> "AgentTrace":
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
    scenario: str | None = None
    expected_outcome: str | None = None
    user_description: str | None = None
    chatbot_role: str | None = None
    conversation_context: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentSample(BaseModel):
    """Test case after agent execution (has trace)."""

    user_input: AgentInputOutput
    trace: AgentTrace
    reference_outcome: AgentInputOutput | None = None
    reference_tool_calls: list[ToolCall] | None = None
    agent_traces: list[AgentTrace] | None = None  # One per agent, in execution order (workflow)
    scenario: str | None = None
    expected_outcome: str | None = None
    user_description: str | None = None
    chatbot_role: str | None = None
    conversation_context: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_partial(
        cls, partial: PartialAgentSample, trace: AgentTrace
    ) -> "AgentSample":
        """Factory: Convert partial + trace to full sample."""
        return cls(
            user_input=partial.user_input,
            trace=trace,
            reference_outcome=partial.reference_outcome,
            reference_tool_calls=partial.reference_tool_calls,
            scenario=partial.scenario,
            expected_outcome=partial.expected_outcome,
            user_description=partial.user_description,
            chatbot_role=partial.chatbot_role,
            conversation_context=partial.conversation_context,
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
    def from_file(cls, path: str | Path) -> "AgentDataset":
        """Load dataset from JSON or JSONL file. Delegates to AgentDatasetLoader."""
        from floeval.api.dataset_loaders.agent_file_loader import (
            AgentDatasetLoader,
        )

        return AgentDatasetLoader.from_file(path)


class WorkflowExecution(BaseModel):
    """Complete record of a multi-agent workflow run.

    Produced by WorkflowExecutor.execute_and_build().
    Consumed by WorkflowEvaluation.

    Contains the full workflow result: per-agent traces, node statuses,
    final output, and optional reference for comparison.
    """

    user_input: AgentInputOutput
    final_output: str = ""
    agent_traces: list[AgentTrace] = Field(
        default_factory=list,
        description="One trace per agent, in execution order.",
    )
    agent_names: list[str] = Field(
        default_factory=list,
        description="Agent names aligned with agent_traces list.",
    )
    node_results: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="node_id → {status, text, agent_name, tool_calls, turn_count}",
    )
    reference_outcome: AgentInputOutput | None = None
    reference_tool_calls: list[ToolCall] | None = None
    session_id: str | None = None
    dag_graph_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def completed_agents(self) -> int:
        """Number of AGENT nodes (not START/END) that completed with SUCCESS status."""
        return sum(
            1 for r in self.node_results.values()
            if r.get("agent_name") and r.get("status") == "SUCCESS"
        )

    @property
    def total_agents(self) -> int:
        """Total number of AGENT nodes (not START/END) in the workflow."""
        agent_node_count = sum(1 for r in self.node_results.values() if r.get("agent_name"))
        return agent_node_count if agent_node_count > 0 else len(self.agent_traces)

    @property
    def workflow_completion_rate(self) -> float:
        """Fraction of agents that completed successfully. 0.0 if no agents."""
        if self.total_agents == 0:
            return 0.0
        return self.completed_agents / self.total_agents

    def get_handoff_pairs(self) -> list[tuple[AgentTrace, AgentTrace]]:
        """Consecutive (upstream, downstream) trace pairs for handoff evaluation."""
        if len(self.agent_traces) < 2:
            return []
        return [
            (self.agent_traces[i], self.agent_traces[i + 1])
            for i in range(len(self.agent_traces) - 1)
        ]

    def build_agent_sample(self, trace: AgentTrace) -> AgentSample:
        """Build an AgentSample from a single agent's trace for per-agent metric execution."""
        return AgentSample(
            user_input=self.user_input,
            trace=trace,
            reference_outcome=self.reference_outcome,
            reference_tool_calls=self.reference_tool_calls,
            metadata={"agent_name": trace.agent_name, "workflow_session": self.session_id},
        )
