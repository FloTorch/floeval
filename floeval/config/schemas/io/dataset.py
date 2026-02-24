"""Dataset and Sample model schemas for evaluations."""

from typing import Any

from pydantic import BaseModel, Field


class PartialSample(BaseModel):
    """Schema for partially created sample.

    Args:
        BaseModel: missing contents are set for auto generation.
    """

    user_input: str = Field(..., description="The input/question/prompt for the LLM")
    contexts: list[str] | None = Field(
        default=None,
        description="Optional retrieved contexts or supporting information",
    )
    llm_response: str | None = Field(
        default=None, description="The actual output/response from the LLM"
    )
    ground_truth: str | None = Field(
        default=None, description="Optional ground truth/reference information"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional sample metadata")
    prompt_ids: list[str] | None = Field(
        default=None,
        description="List of prompt IDs to generate responses for (expands to multiple samples)",
    )


class Sample(BaseModel):
    """Single evaluation sample."""

    user_input: str = Field(..., description="The input/question/prompt for the LLM")
    contexts: list[str] | None = Field(
        default=None,
        description="Optional retrieved contexts or supporting information",
    )
    llm_response: str = Field(..., description="The actual output/response from the LLM")
    ground_truth: str | None = Field(
        default=None, description="Optional ground truth/reference information"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional sample metadata")
    prompt_id: str | None = Field(
        default=None,
        description="ID of the prompt used for this response (set after generation)",
    )


class Dataset(BaseModel):
    """Collection of samples."""

    samples: list[Sample] = Field(..., min_length=1, description="Non-empty list of samples")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)


class PartialDataset(BaseModel):
    """Schema for partially created dataset.

    Args:
        BaseModel: missing contents are set for auto generation.
    """

    samples: list[PartialSample] = Field(..., min_length=1, description="Non-empty list of samples")

    def __len__(self) -> int:
        """Return the number of partial samples in the dataset."""
        return len(self.samples)


def convert_partial_to_full_sample(
    partial_sample: PartialSample,
    llm_response: str,
    prompt_id: str | None = None,
) -> Sample:
    """Utility function to convert a PartialSample to a full Sample by filling in the llm_response.

    Args:
        partial_sample: PartialSample instance that needs to be converted to Sample
        llm_response: The generated LLM response that will be filled into the Sample
        prompt_id: Optional ID of the prompt used for this response

    Returns:
        Sample: A fully filled Sample instance with llm_response included
    """
    return Sample(
        user_input=partial_sample.user_input,
        contexts=partial_sample.contexts,
        llm_response=llm_response,
        ground_truth=partial_sample.ground_truth,
        metadata=partial_sample.metadata,
        prompt_id=prompt_id,
    )
