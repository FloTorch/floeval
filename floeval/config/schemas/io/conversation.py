"""Multi-turn conversation payloads for evaluations (e.g. DeepEval ConversationalTestCase)."""

from typing import Any, Literal

from pydantic import BaseModel, Field

# Opaque JSON-serializable tool payload until DeepEval / callers narrow it further.
type ToolCallPayload = Any

type ConversationRole = Literal["user", "assistant"]


class ConversationTurn(BaseModel):
    """One message in a user–assistant conversation (OpenAI-style roles)."""

    role: ConversationRole = Field(..., description="Speaker role for this turn.")
    content: str = Field(..., description="Message text.")
    retrieval_context: list[str] | None = Field(
        default=None,
        description="Optional retrieval chunks (assistant turns only; maps to DeepEval Turn).",
    )
    tools_called: list[ToolCallPayload] | None = Field(
        default=None,
        description="Optional tool calls (assistant turns only).",
    )
