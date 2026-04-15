"""Unit tests for RAGAS AspectCritic wrapper."""

from typing import Any, cast

import pytest

from floeval.metric_providers.ragas.metrics import RAGASAspectCritic


class _FakeAdapter:
    def __init__(self) -> None:
        self.llm = object()
        self.embeddings = object()

    def transform_sample(self, sample):
        return sample


def test_aspect_critic_requires_definition() -> None:
    with pytest.raises(ValueError, match="requires a non-empty 'definition'"):
        RAGASAspectCritic(adapter=cast(Any, _FakeAdapter()))


def test_aspect_critic_accepts_definition_from_params() -> None:
    metric = RAGASAspectCritic(
        adapter=cast(Any, _FakeAdapter()),
        params={"definition": "Does the response avoid harmful content?"},
    )
    assert metric.name == "aspect_critic"
