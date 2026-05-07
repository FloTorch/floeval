"""Conversational (multi-turn) dataset rows — separate from single-turn ``Sample``."""

from typing import Any, Literal, Self

from pydantic import AliasChoices, BaseModel, Field, model_validator

from floeval.config.schemas.io.conversation import (
    ConversationTurn,
    ToolCallPayloadInput,
)

type ConversationalDatasetRow = ConversationalSample


def _normalize_conversation_row_input(value: object) -> object:
    if not isinstance(value, dict):
        return value
    data = dict(value)
    if "turns" in data:
        return data
    trace = data.get("trace")
    if isinstance(trace, dict):
        trace_messages = trace.get("messages")
        if isinstance(trace_messages, list):
            data["turns"] = trace_messages
            return data
    messages = data.get("messages")
    if isinstance(messages, list):
        data["turns"] = messages
    return data


class PartialConversationalSample(BaseModel):
    """Partial conversational row (e.g. before optional fields are filled)."""

    # ------------- Transcript — DeepEval + RAGAS (Turn list / MultiTurnSample.user_input) -------------
    turns: list[ConversationTurn] = Field(
        ...,
        validation_alias=AliasChoices("turns", "messages"),
        description="Transcript turns (user/assistant alternation).",
    )
    # ------------- DeepEval-specific (ConversationalTestCase) -------------
    scenario: str | None = None
    user_description: str | None = None
    chatbot_role: str | None = None
    conversation_context: list[str] | None = None
    # ------------- Expected outcome — DeepEval ``expected_outcome`` + RAGAS ``MultiTurnSample.reference`` -------------
    reference_outcome: str | None = Field(
        default=None,
        validation_alias=AliasChoices("reference_outcome", "reference", "expected_outcome"),
        description="Desired end state of the conversation; maps to DeepEval expected_outcome and RAGAS reference.",
    )
    # ------------- RAGAS-specific (MultiTurnSample) -------------
    reference_topics: list[str] | None = Field(
        default=None,
        description="RAGAS topic_adherence reference topics.",
    )
    reference_tool_calls: list[ToolCallPayloadInput] | None = None
    rubrics: dict[str, str] | None = None
    # ------------- Floeval row metadata (not passed to DeepEval/RAGAS adapters) -------------
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_turns_aliases(cls, value: object) -> object:
        return _normalize_conversation_row_input(value)


class ConversationalSample(BaseModel):
    """Gold conversational transcript for DeepEval/RAGAS multi-turn adapters."""

    # ------------- Transcript — DeepEval + RAGAS (Turn list / MultiTurnSample.user_input) -------------
    turns: list[ConversationTurn] = Field(..., min_length=1)
    # ------------- DeepEval-specific (ConversationalTestCase) -------------
    scenario: str | None = None
    user_description: str | None = None
    chatbot_role: str | None = None
    conversation_context: list[str] | None = None
    # ------------- Expected outcome — DeepEval ``expected_outcome`` + RAGAS ``MultiTurnSample.reference`` -------------
    reference_outcome: str | None = Field(
        default=None,
        validation_alias=AliasChoices("reference_outcome", "reference", "expected_outcome"),
        description="Desired end state of the conversation; maps to DeepEval expected_outcome and RAGAS reference.",
    )
    # ------------- RAGAS-specific (MultiTurnSample) -------------
    reference_topics: list[str] | None = Field(
        default=None,
        description="RAGAS topic_adherence reference topics.",
    )
    reference_tool_calls: list[ToolCallPayloadInput] | None = None
    rubrics: dict[str, str] | None = None
    # ------------- Floeval row metadata (not passed to DeepEval/RAGAS adapters) -------------
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_turns_aliases(cls, value: object) -> object:
        return _normalize_conversation_row_input(value)

    @model_validator(mode="after")
    def _validate_turns_shape(self) -> Self:
        turns = self.turns
        if len(turns) < 2:
            raise ValueError("ConversationalSample requires at least two turns.")
        if turns[0].role != "user":
            raise ValueError("First turn must have role 'user'.")
        expected: Literal["user", "assistant"] = "user"
        for i, t in enumerate(turns):
            if t.role != expected:
                raise ValueError(
                    f"Turn {i}: expected role '{expected}', got '{t.role}' "
                    "(strict user/assistant alternation)."
                )
            expected = "assistant" if expected == "user" else "user"
        has_assistant = any(t.role == "assistant" for t in turns)
        if not has_assistant:
            raise ValueError("ConversationalSample must include at least one assistant turn.")
        return self


class ConversationalDataset(BaseModel):
    """Typed container for conversational evaluation rows."""

    samples: list[ConversationalSample] = Field(
        ..., min_length=1, description="Non-empty list of conversational samples"
    )

    def __len__(self) -> int:
        """Return the number of conversational samples."""
        return len(self.samples)


class PartialConversationalDataset(BaseModel):
    """Partial conversational dataset (no auto-generation in Floeval yet)."""

    samples: list[PartialConversationalSample] = Field(
        ..., min_length=1
    )

    def __len__(self) -> int:
        """Return the number of partial conversational samples."""
        return len(self.samples)
