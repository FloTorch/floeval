"""Evaluation dataset preparation (mocked LLM population)."""

from unittest.mock import patch

import pytest

from floeval.api.evaluation import Evaluation
from floeval.config.schemas.io.dataset import Dataset, PartialDataset, PartialSample, Sample
from floeval.config.schemas.io.llm import OpenAIProviderConfig

pytestmark = pytest.mark.unit


def test_partial_dataset_without_llm_config_raises() -> None:
    partial = PartialDataset(samples=[PartialSample(user_input="hello")])
    with pytest.raises(ValueError, match="llm_config"):
        Evaluation(
            dataset=partial,
            metrics=["deepeval:exact_match"],
        )


def test_partial_dataset_populates_via_mock() -> None:
    partial = PartialDataset(samples=[PartialSample(user_input="hello")])
    filled = Dataset(samples=[Sample(user_input="hello", llm_response="world")])
    llm = OpenAIProviderConfig(api_key="unit-test-key")

    with patch(
        "floeval.api.evaluation.populate_llm_responses",
        return_value=filled,
    ) as mock_pop:
        ev = Evaluation(
            dataset=partial,
            metrics=["deepeval:exact_match"],
            llm_config=llm,
            dataset_generator_model="gpt-4o-mini",
        )

    mock_pop.assert_called_once()
    assert ev.dataset is filled
    assert ev.dataset.samples[0].llm_response == "world"
