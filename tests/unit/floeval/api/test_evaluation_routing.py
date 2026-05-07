"""Evaluation dataset type vs metric capability guardrails."""

# ruff: noqa: D101,D102,D103

import pytest

from floeval.api import Evaluation
from floeval.api.dataset import conversational_dataset_from_dict, dataset_from_dict
from floeval.api.metrics.base import BaseMetric, MetricResult


def test_conversational_dataset_rejects_single_turn_ragas_metric() -> None:
    data = {
        "samples": [
            {
                "turns": [
                    {"role": "user", "content": "a"},
                    {"role": "assistant", "content": "b"},
                ]
            }
        ]
    }
    cds = conversational_dataset_from_dict(data, partial_dataset=False)
    with pytest.raises(ValueError, match="single-turn RAGAS"):
        Evaluation(
            dataset=cds,
            metrics=[RAGASAnswerRelevancyStub()],
        ).run()


def test_single_turn_dataset_rejects_multi_turn_ragas() -> None:
    ds = dataset_from_dict(
        {
            "samples": [
                {
                    "user_input": "q",
                    "llm_response": "a",
                }
            ]
        },
        partial_dataset=False,
    )
    with pytest.raises(ValueError, match="multi-turn RAGAS"):
        Evaluation(
            dataset=ds,
            metrics=[RAGASTopicAdherenceStub()],
        ).run()


def test_single_turn_dataset_rejects_conversational_deepeval_metric() -> None:
    ds = dataset_from_dict(
        {
            "samples": [
                {
                    "user_input": "q",
                    "llm_response": "a",
                }
            ]
        },
        partial_dataset=False,
    )
    with pytest.raises(ValueError, match="conversational DeepEval metric"):
        Evaluation(
            dataset=ds,
            metrics=[DeepEvalConversationalStub()],
        ).run()


def test_conversational_dataset_rejects_llm_deepeval_metric() -> None:
    data = {
        "conversational_samples": [
            {
                "turns": [
                    {"role": "user", "content": "a"},
                    {"role": "assistant", "content": "b"},
                ]
            }
        ]
    }
    cds = conversational_dataset_from_dict(data, partial_dataset=False)
    with pytest.raises(ValueError, match="LLMTestCase DeepEval metric"):
        Evaluation(
            dataset=cds,
            metrics=[DeepEvalLLMStub()],
        ).run()


class RAGASAnswerRelevancyStub(BaseMetric):
    """Minimal stub with RAGAS routing and default single_turn kind."""

    def __init__(self) -> None:
        super().__init__(name="answer_relevancy")
        self.provider = "ragas"
        self.execute_via = "ragas"
        self.threshold = 0.5

    def evaluate(self, *args, **kwargs) -> MetricResult:
        raise NotImplementedError


class RAGASTopicAdherenceStub(BaseMetric):
    def __init__(self) -> None:
        super().__init__(name="topic_adherence")
        self.provider = "ragas"
        self.execute_via = "ragas"
        self.ragas_sample_kind = "multi_turn"

    def evaluate(self, *args, **kwargs) -> MetricResult:
        raise NotImplementedError


class DeepEvalConversationalStub(BaseMetric):
    def __init__(self) -> None:
        super().__init__(name="turn_relevancy")
        self.provider = "deepeval"
        self.execute_via = "deepeval"
        self.deepeval_test_case_kind = "conversational"

    def evaluate(self, *args, **kwargs) -> MetricResult:
        raise NotImplementedError


class DeepEvalLLMStub(BaseMetric):
    def __init__(self) -> None:
        super().__init__(name="faithfulness")
        self.provider = "deepeval"
        self.execute_via = "deepeval"
        self.deepeval_test_case_kind = "llm"

    def evaluate(self, *args, **kwargs) -> MetricResult:
        raise NotImplementedError
