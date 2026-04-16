"""DatasetLoader / file loading."""

import json

import pytest

from floeval.api import DatasetLoader
from floeval.config.schemas.io.conversational_dataset import ConversationalDataset
from floeval.config.schemas.io.dataset import Dataset, PartialDataset

pytestmark = pytest.mark.unit


def test_from_dict_full_dataset() -> None:
    data = {
        "samples": [
            {"user_input": "Q?", "llm_response": "A."},
        ]
    }
    ds = DatasetLoader.from_dict(data, partial_dataset=False)
    assert isinstance(ds, Dataset)
    assert len(ds.samples) == 1
    assert ds.samples[0].llm_response == "A."


def test_from_dict_partial_dataset() -> None:
    data = {"samples": [{"user_input": "Q?"}]}
    ds = DatasetLoader.from_dict(data, partial_dataset=True)
    assert isinstance(ds, PartialDataset)
    assert ds.samples[0].llm_response is None


def test_from_file_jsonl(tmp_path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps({"user_input": "one", "llm_response": "uno"}) + "\n"
        + json.dumps({"user_input": "two", "llm_response": "dos"}) + "\n",
        encoding="utf-8",
    )
    ds = DatasetLoader.from_file(path, partial_dataset=False)
    assert len(ds.samples) == 2


def test_from_file_missing_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        DatasetLoader.from_file(tmp_path / "nope.jsonl", partial_dataset=False)


def test_from_samples_full_dataset_accepts_dicts() -> None:
    ds = DatasetLoader.from_samples(
        [{"user_input": "Q?", "llm_response": "A."}],
        partial_dataset=False,
    )
    assert isinstance(ds, Dataset)
    assert ds.samples[0].llm_response == "A."


def test_from_samples_partial_dataset_allows_missing_response() -> None:
    ds = DatasetLoader.from_samples(
        [{"user_input": "Q?"}],
        partial_dataset=True,
    )
    assert isinstance(ds, PartialDataset)
    assert ds.samples[0].llm_response is None


def test_conversational_from_file_accepts_samples_wrapper(tmp_path) -> None:
    path = tmp_path / "conv.json"
    path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "messages": [
                            {"role": "human", "content": "Q?"},
                            {"role": "ai", "content": "A."},
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    ds = DatasetLoader.conversational_from_file(path, partial_dataset=False)
    assert isinstance(ds, ConversationalDataset)
    assert len(ds.samples) == 1
    assert ds.samples[0].turns[0].role == "user"
