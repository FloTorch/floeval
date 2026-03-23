"""Custom metrics and criteria (integration; multiple flows, sequential registration)."""

from pathlib import Path

import pytest
import yaml

from floeval.api import DatasetLoader, Evaluation
from floeval.api.metrics.custom import criteria, custom_metric
from floeval.config.schemas.io.dataset import Dataset, PartialDataset
from floeval.config.schemas.io.llm import OpenAIProviderConfig

from tests.conftest import resolve_env_placeholders

pytestmark = pytest.mark.integration


def _define_custom_metrics() -> None:
    @custom_metric(threshold=0.4)
    def response_length(response: str) -> float:
        return min(len(response or "") / 80.0, 1.0)

    @custom_metric(execute_via="ragas", threshold=0.5)
    def custom_faithfulness_via_ragas(response: str, contexts: list) -> float:
        if not contexts:
            return 0.0
        context_text = " ".join(contexts).lower()
        response_lower = response.lower()
        context_words = set(context_text.split())
        response_words = set(response_lower.split())
        overlap = len(context_words & response_words)
        return min(overlap / max(len(context_words), 1), 1.0)

    @custom_metric(execute_via="deepeval", threshold=0.4)
    def custom_relevancy_via_deepeval(response: str, question: str) -> float:
        if not response or not question:
            return 0.0
        question_words = set(question.lower().split())
        response_words = set(response.lower().split())
        overlap = len(question_words & response_words)
        return min(overlap / max(len(question_words), 1), 1.0)

    criteria(
        name="clarity",
        description="Rate how clear and easy to understand the response is (0-1).",
        threshold=0.5,
    )
    criteria(
        name="empathy_via_ragas",
        description="Rate how empathetic the response is on a scale of 0-1.",
        threshold=0.5,
        execute_via="ragas",
    )


@pytest.fixture(scope="module")
def _registered_custom_metrics():
    _define_custom_metrics()
    yield


def _llm_config(config_data_dir: Path) -> OpenAIProviderConfig:
    path = config_data_dir / "common_config" / "config.new.part-dataset.yaml"
    with open(path, encoding="utf-8") as f:
        data = resolve_env_placeholders(yaml.safe_load(f))
    return OpenAIProviderConfig(**data["llm_config"])


def test_custom_full_dataset_response_length_and_clarity(
    _registered_custom_metrics,
    config_data_dir: Path,
    requires_llm_credentials,
) -> None:
    llm_config = _llm_config(config_data_dir)
    full_ds = DatasetLoader.from_file(
        config_data_dir / "datasets" / "general_qa_full_dataset.jsonl",
        partial_dataset=False,
    )
    assert isinstance(full_ds, Dataset)
    result = Evaluation(
        dataset=full_ds,
        metrics=["response_length", "clarity"],
        llm_config=llm_config,
    ).run()
    assert result.summary.get("total_samples", 0) >= 1


@pytest.mark.ragas
def test_custom_full_dataset_ragas_answer_relevancy_with_custom_length(
    _registered_custom_metrics,
    config_data_dir: Path,
    requires_llm_credentials,
) -> None:
    llm_config = _llm_config(config_data_dir)
    full_ds = DatasetLoader.from_file(
        config_data_dir / "datasets" / "general_qa_full_dataset.jsonl",
        partial_dataset=False,
    )
    result = Evaluation(
        dataset=full_ds,
        metrics=["ragas:answer_relevancy", "custom:response_length"],
        llm_config=llm_config,
    ).run()
    assert result.summary.get("total_samples", 0) >= 1


@pytest.mark.ragas
def test_custom_partial_dataset_empathy_via_ragas(
    _registered_custom_metrics,
    config_data_dir: Path,
    requires_llm_credentials,
) -> None:
    llm_config = _llm_config(config_data_dir)
    partial_ds = DatasetLoader.from_file(
        config_data_dir / "datasets" / "general_qa_partial_dataset_20_samples.jsonl",
        partial_dataset=True,
    )
    assert isinstance(partial_ds, PartialDataset)
    result = Evaluation(
        dataset=partial_ds,
        metrics=["response_length", "empathy_via_ragas"],
        llm_config=llm_config,
        dataset_generator_model="flotorch/evaluation-testing-chat-model",
    ).run()
    assert result.summary.get("total_samples", 0) >= 1
