"""Populate partial dataset via OpenAI provider (integration)."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from floeval.api.dataset import DatasetLoader
from floeval.cli.export import export_dataset_to_json
from floeval.config.schemas.io.dataset import PartialDataset
from floeval.config.schemas.io.llm import OpenAIProviderConfig
from floeval.core.execution.llm_executor import OpenAIProvider
from floeval.core.execution.response_synthesizer import populate_llm_responses
from tests.conftest import resolve_env_placeholders

pytestmark = pytest.mark.integration


def test_generate_dataset_from_partial(tmp_path: Path, config_data_dir: Path, requires_llm_credentials) -> None:
    cfg_path = config_data_dir / "common_config" / "generate_dataset.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        config_data: dict[str, Any] = resolve_env_placeholders(yaml.safe_load(f))

    provider_config = OpenAIProviderConfig(**config_data["provider_config"])
    openai_provider = OpenAIProvider(
        config_name="generation_config",
        **provider_config.model_dump(),
    )
    partial_dataset = DatasetLoader.from_dict(config_data["partial_data"], partial_dataset=True)
    assert isinstance(partial_dataset, PartialDataset)

    dataset = populate_llm_responses(
        partial_dataset=partial_dataset,
        llm_provider=openai_provider,
    )
    out_path = tmp_path / "generated_dataset.json"
    export_dataset_to_json(dataset, out_path, indent=2)
    assert out_path.is_file()
    assert len(dataset.samples) >= 1
