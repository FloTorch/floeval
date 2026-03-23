"""Integration: evaluation from shared gateway config — one backend per test."""

from pathlib import Path
from typing import Any

import pytest

from floeval.api import DatasetLoader, Evaluation
from floeval.config.schemas.io.llm import OpenAIProviderConfig
from tests._support.assertions import assert_samples_have_metric

pytestmark = pytest.mark.integration


@pytest.fixture
def provider_samples_dataset(config_data_dir: Path, load_yaml):
    data = load_yaml(config_data_dir / "datasets" / "provider_metrics_samples.yaml")
    return DatasetLoader.from_samples(data["samples"], partial_dataset=False)


@pytest.mark.ragas
def test_ragas_answer_relevancy_from_gateway_config(
    config_data_dir: Path,
    load_yaml,
    resolve_env,
    requires_llm_credentials,
    provider_samples_dataset,
) -> None:
    config_data: dict[str, Any] = resolve_env(
        load_yaml(config_data_dir / "common_config" / "config_gateway_ragas.yaml")
    )
    llm_config = OpenAIProviderConfig(**config_data["llm_config"])
    evaluation = Evaluation(
        dataset=provider_samples_dataset,
        metrics=config_data["evaluation_config"]["metrics"],
        llm_config=llm_config,
        prompts_file=config_data["evaluation_config"].get("prompts_file"),
    )
    results = evaluation.run()
    assert_samples_have_metric(results, "ragas:answer_relevancy", require_score=True)


@pytest.mark.deepeval
def test_deepeval_answer_relevancy_from_gateway_config(
    config_data_dir: Path,
    load_yaml,
    resolve_env,
    requires_llm_credentials,
    provider_samples_dataset,
) -> None:
    config_data: dict[str, Any] = resolve_env(
        load_yaml(config_data_dir / "common_config" / "config_gateway_deepeval.yaml")
    )
    llm_config = OpenAIProviderConfig(**config_data["llm_config"])
    evaluation = Evaluation(
        dataset=provider_samples_dataset,
        metrics=config_data["evaluation_config"]["metrics"],
        llm_config=llm_config,
        prompts_file=config_data["evaluation_config"].get("prompts_file"),
    )
    results = evaluation.run()
    assert_samples_have_metric(results, "deepeval:answer_relevancy", require_score=True)
