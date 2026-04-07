"""Integration tests for DeepEval conversational G-Eval."""

from pathlib import Path

import pytest

from floeval.api import DatasetLoader, Evaluation
from floeval.config.schemas.io.llm import OpenAIProviderConfig
from tests._support.assertions import assert_samples_have_metric

pytestmark = pytest.mark.deepeval


@pytest.mark.integration
def test_deepeval_conversational_g_eval_from_config(
    config_data_dir: Path,
    load_yaml,
    resolve_env,
    request: pytest.FixtureRequest,
) -> None:
    """Run conversational G-Eval on a multi-turn dataset."""
    request.getfixturevalue("requires_llm_credentials")

    config_data = resolve_env(
        load_yaml(config_data_dir / "deepeval_config" / "config.conversational_g_eval.yaml")
    )
    dataset = DatasetLoader.conversational_from_file(
        config_data_dir / "datasets" / "multi_turn_conversational_dataset.json",
        partial_dataset=False,
    )
    metrics = config_data["evaluation_config"]["metrics"]
    kwargs: dict = {"dataset": dataset, "metrics": metrics}
    if "llm_config" in config_data:
        kwargs["llm_config"] = OpenAIProviderConfig(**config_data["llm_config"])

    results = Evaluation(**kwargs).run()
    assert_samples_have_metric(results, "deepeval:conversational_g_eval", require_score=True)
