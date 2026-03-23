"""Unit tests for dataset Pydantic schemas."""

import pytest
from pydantic import ValidationError

from floeval.config.schemas.io.dataset import (
    Dataset,
    PartialDataset,
    PartialSample,
    Sample,
    convert_partial_to_full_sample,
)

pytestmark = pytest.mark.unit


def test_sample_requires_llm_response() -> None:
    with pytest.raises(ValidationError):
        Sample.model_validate({"user_input": "hello"})


def test_partial_dataset_requires_non_empty_samples() -> None:
    with pytest.raises(ValidationError):
        PartialDataset(samples=[])


def test_convert_partial_to_full_sample_preserves_fields() -> None:
    partial = PartialSample(
        user_input="hello",
        contexts=["ctx"],
        ground_truth="gt",
        metadata={"k": "v"},
    )
    full = convert_partial_to_full_sample(
        partial_sample=partial,
        llm_response="world",
        prompt_id="p1",
    )
    assert full.user_input == "hello"
    assert full.contexts == ["ctx"]
    assert full.ground_truth == "gt"
    assert full.metadata == {"k": "v"}
    assert full.llm_response == "world"
    assert full.prompt_id == "p1"


def test_dataset_len_matches_samples() -> None:
    ds = Dataset(samples=[Sample(user_input="q", llm_response="a")])
    assert len(ds) == 1
