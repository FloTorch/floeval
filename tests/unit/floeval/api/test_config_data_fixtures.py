"""Ensure committed config_data dataset fixtures load without schema errors."""

from pathlib import Path

import pytest
import yaml

from floeval.api import DatasetLoader
from floeval.config.schemas.io.dataset import Dataset, PartialDataset
from floeval.config.schemas.prompts import PromptFile

pytestmark = pytest.mark.unit

# (path under config_data/, partial_dataset flag, expected sample count)
_JSONL_FULL_FIXTURES: list[tuple[str, bool, int]] = [
    ("datasets/general_qa_full_dataset.jsonl", False, 5),
    ("datasets/general_qa_full_dataset_v2.jsonl", False, 5),
    ("datasets/general_qa_example_dataset.jsonl", False, 3),
    ("datasets/pattern_match_evaluation_dataset.jsonl", False, 8),
    ("datasets/json_correctness_evaluation_dataset.jsonl", False, 5),
    ("datasets/toxicity_evaluation_dataset.jsonl", False, 8),
]

_JSONL_PARTIAL_FIXTURES: list[tuple[str, bool, int]] = [
    ("datasets/general_qa_partial_dataset_5_samples.jsonl", True, 5),
    ("datasets/general_qa_partial_dataset_20_samples.jsonl", True, 20),
    ("datasets/prompt_evaluation_multi_prompt_dataset.jsonl", True, 5),
]

_JSON_FILE_FIXTURES: list[tuple[str, bool, int]] = [
    ("datasets/tiny_full_dataset.json", False, 1),
    ("datasets/tiny_partial_dataset.json", True, 1),
]


@pytest.mark.parametrize(
    "relative_path,partial,expected_len",
    _JSONL_FULL_FIXTURES + _JSONL_PARTIAL_FIXTURES + _JSON_FILE_FIXTURES,
)
def test_config_data_dataset_loads(
    config_data_dir: Path,
    relative_path: str,
    partial: bool,
    expected_len: int,
) -> None:
    path = config_data_dir / relative_path
    ds = DatasetLoader.from_file(path, partial_dataset=partial)
    if partial:
        assert isinstance(ds, PartialDataset)
    else:
        assert isinstance(ds, Dataset)
    assert len(ds) == expected_len


def test_provider_metrics_samples_yaml_loads(config_data_dir: Path, load_yaml) -> None:
    data = load_yaml(config_data_dir / "datasets" / "provider_metrics_samples.yaml")
    samples = data["samples"]
    assert isinstance(samples, list) and len(samples) == 3
    ds = DatasetLoader.from_samples(samples, partial_dataset=False)
    assert isinstance(ds, Dataset)
    assert len(ds) == 3


def test_prompts_yaml_loads(config_data_dir: Path) -> None:
    with open(
        config_data_dir / "datasets" / "prompts.yaml",
        encoding="utf-8",
    ) as f:
        raw = yaml.safe_load(f)
    pf = PromptFile.model_validate(raw)
    assert "v1_concise" in pf.prompts
    assert "v2_detailed" in pf.prompts
    assert "template" in pf.prompts["v1_concise"].model_dump()
