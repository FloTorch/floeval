"""Dataset and Sample classes.
The RAGAS adapter supports Pydantic models via `model_dump()`.
"""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class Sample(BaseModel):
    """Single evaluation sample."""

    user_input: str = Field(..., description="The input/question/prompt for the LLM")
    contexts: list[str] | None = Field(
        default=None,
        description="Optional retrieved contexts or supporting information",
    )
    llm_response: str = Field(
        ..., description="The actual output/response from the LLM"
    )
    ground_truth: str | None = Field(
        default=None, description="Optional ground truth/reference information"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Optional sample metadata"
    )


class Dataset(BaseModel):
    """Collection of samples."""

    samples: List[Sample] = Field(
        ..., min_length=1, description="Non-empty list of samples"
    )

    def __len__(self) -> int:
        return len(self.samples)
