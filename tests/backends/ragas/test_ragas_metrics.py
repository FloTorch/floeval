"""RAGAS backend tests."""

from pathlib import Path

import pytest

from floeval.api import DatasetLoader, Evaluation
from floeval.config.schemas.io.llm import OpenAIProviderConfig
from tests._support.assertions import assert_samples_have_metric

pytestmark = [pytest.mark.ragas, pytest.mark.integration]


@pytest.mark.parametrize(
    "config_rel,dataset_rel,metric_key",
    [
        pytest.param(
            "ragas_config/config.ragas_context_recall.yaml",
            "datasets/general_qa_full_dataset.jsonl",
            "ragas:context_recall",
            id="context_recall",
        ),
        pytest.param(
            "ragas_config/config.ragas_context_precision.yaml",
            "datasets/general_qa_full_dataset.jsonl",
            "ragas:context_precision",
            id="context_precision",
        ),
        pytest.param(
            "ragas_config/config.ragas_context_entity_recall.yaml",
            "datasets/general_qa_full_dataset.jsonl",
            "ragas:context_entity_recall",
            id="context_entity_recall",
        ),
        pytest.param(
            "ragas_config/config.ragas_noise_sensitivity.yaml",
            "datasets/general_qa_full_dataset.jsonl",
            "ragas:noise_sensitivity",
            id="noise_sensitivity",
        ),
    ],
)
def test_ragas_metrics(
    config_data_dir: Path,
    load_yaml,
    resolve_env,
    requires_llm_credentials,
    config_rel: str,
    dataset_rel: str,
    metric_key: str,
) -> None:
    """Run one RAGAS metric and assert metric output exists."""
    config_data = resolve_env(load_yaml(config_data_dir / config_rel))
    dataset = DatasetLoader.from_file(config_data_dir / dataset_rel, partial_dataset=False)
    llm_config = OpenAIProviderConfig(**config_data["llm_config"])
    results = Evaluation(
        dataset=dataset,
        metrics=config_data["evaluation_config"]["metrics"],
        llm_config=llm_config,
    ).run()
    assert_samples_have_metric(results, metric_key, require_score=True)
