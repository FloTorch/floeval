"""Criteria-based prompt evaluation (integration)."""

import json
from pathlib import Path

import pytest
import yaml

from floeval.api.dataset import DatasetLoader
from floeval.api.evaluation import Evaluation
from floeval.api.metrics.custom import criteria
from floeval.config.schemas.io.llm import OpenAIProviderConfig

from tests.conftest import resolve_env_placeholders

pytestmark = pytest.mark.integration


def test_prompt_evaluation_with_criteria(tmp_path: Path, config_data_dir: Path, requires_llm_credentials) -> None:
    config_path = config_data_dir / "common_config" / "config.new.yaml"
    criteria_path = config_data_dir / "common_config" / "prompt_eval_criteria.yaml"
    dataset_path = config_data_dir / "datasets" / "prompt_evaluation_multi_prompt_dataset.jsonl"

    with open(config_path, encoding="utf-8") as f:
        config = resolve_env_placeholders(yaml.safe_load(f))
    with open(criteria_path, encoding="utf-8") as f:
        criteria_data = yaml.safe_load(f)["criteria"]

    llm_config = OpenAIProviderConfig(**config["llm_config"])
    partial_dataset = DatasetLoader.from_file(dataset_path, partial_dataset=True)

    response_clarity = criteria(
        name=criteria_data["name"],
        description=criteria_data["description"],
        threshold=criteria_data["threshold"],
        evaluation_steps=criteria_data["evaluation_steps"],
        llm_config=llm_config,
    )

    result = Evaluation(
        dataset=partial_dataset,
        metrics=[response_clarity],
        llm_config=llm_config,
        dataset_generator_model=config["dataset_generation_config"]["generator_model"],
        prompts_file=config["evaluation_config"].get("prompts_file"),
    ).run()

    out = tmp_path / "prompt-eval-result.json"
    out.write_text(json.dumps(result.model_dump(), indent=2), encoding="utf-8")
    assert result.summary.get("total_samples", 0) >= 1
    assert result.aggregate_scores
