"""Multi-turn conversation payloads for evaluations (e.g. DeepEval ConversationalTestCase)."""

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field


class ToolCallPayload(BaseModel):
    """Unified tool-call payload across dataset formats and adapters."""

    name: str = Field(..., description="Tool/function name.")
    args: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("args", "input", "input_parameters"),
        description="Tool input arguments.",
    )
    output: str | None = Field(
        default=None,
        description="Optional tool output to align assistant tool calls with tool messages.",
    )


type ToolCallPayloadInput = ToolCallPayload | dict[str, Any]

type ConversationRole = Literal["user", "assistant"]


class ConversationTurn(BaseModel):
    """One message in a user–assistant conversation (OpenAI-style roles)."""

    role: ConversationRole = Field(..., description="Speaker role for this turn.")
    content: str = Field(..., description="Message text.")
    retrieval_context: list[str] | None = Field(
        default=None,
        description="Optional retrieval chunks (assistant turns only; maps to DeepEval Turn).",
    )
    tools_called: list[ToolCallPayloadInput] | None = Field(
        default=None,
        description="Optional tool calls (assistant turns only).",
    )
