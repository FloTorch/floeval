"""DeepEval backend tests."""

from pathlib import Path

import pytest
from pydantic import BaseModel

from floeval.api import DatasetLoader, Evaluation
from floeval.config.schemas.io.llm import OpenAIProviderConfig

from tests._support.assertions import assert_samples_have_metric

pytestmark = pytest.mark.deepeval


class JsonCorrectnessSchema(BaseModel):
    answer: str


@pytest.mark.parametrize(
    "config_rel,dataset_rel,metric_key,needs_llm,use_json_schema",
    [
        pytest.param(
            "deepeval_config/config.exact_match.yaml",
            "datasets/general_qa_full_dataset.jsonl",
            "deepeval:exact_match",
            False,
            False,
            id="exact_match",
        ),
        pytest.param(
            "deepeval_config/config.pattern_match.yaml",
            "datasets/pattern_match_evaluation_dataset.jsonl",
            "deepeval:pattern_match",
            False,
            False,
            id="pattern_match",
        ),
        pytest.param(
            "deepeval_config/config.json_correctness.yaml",
            "datasets/json_correctness_evaluation_dataset.jsonl",
            "deepeval:json_correctness",
            True,
            True,
            marks=pytest.mark.integration,
            id="json_correctness",
        ),
        pytest.param(
            "deepeval_config/config.toxicity.yaml",
            "datasets/toxicity_evaluation_dataset.jsonl",
            "deepeval:toxicity",
            True,
            False,
            marks=pytest.mark.integration,
            id="toxicity",
        ),
        pytest.param(
            "deepeval_config/config.hallucination.yaml",
            "datasets/general_qa_full_dataset.jsonl",
            "deepeval:hallucination",
            True,
            False,
            marks=pytest.mark.integration,
            id="hallucination",
        ),
        pytest.param(
            "deepeval_config/config.contextual_precision.yaml",
            "datasets/general_qa_full_dataset.jsonl",
            "deepeval:contextual_precision",
            True,
            False,
            marks=pytest.mark.integration,
            id="contextual_precision",
        ),
        pytest.param(
            "deepeval_config/config.contextual_recall.yaml",
            "datasets/general_qa_full_dataset.jsonl",
            "deepeval:contextual_recall",
            True,
            False,
            marks=pytest.mark.integration,
            id="contextual_recall",
        ),
        pytest.param(
            "deepeval_config/config.contextual_relevancy.yaml",
            "datasets/general_qa_full_dataset.jsonl",
            "deepeval:contextual_relevancy",
            True,
            False,
            marks=pytest.mark.integration,
            id="contextual_relevancy",
        ),
    ],
)
def test_deepeval_metrics(
    request: pytest.FixtureRequest,
    config_data_dir: Path,
    load_yaml,
    resolve_env,
    config_rel: str,
    dataset_rel: str,
    metric_key: str,
    needs_llm: bool,
    use_json_schema: bool,
) -> None:
    """Run one DeepEval metric and assert metric output exists."""
    if needs_llm:
        request.getfixturevalue("requires_llm_credentials")

    config_data = resolve_env(load_yaml(config_data_dir / config_rel))
    dataset = DatasetLoader.from_file(config_data_dir / dataset_rel, partial_dataset=False)
    metrics = config_data["evaluation_config"]["metrics"]
    if use_json_schema:
        metrics[0]["params"]["expected_schema"] = JsonCorrectnessSchema

    kwargs = {"dataset": dataset, "metrics": metrics}
    if "llm_config" in config_data:
        kwargs["llm_config"] = OpenAIProviderConfig(**config_data["llm_config"])

    results = Evaluation(**kwargs).run()
    assert_samples_have_metric(results, metric_key, require_score=True)
