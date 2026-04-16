"""Dataset and Sample classes.

The RAGAS adapter supports Pydantic models via `model_dump()`.
"""

import json
from pathlib import Path
from typing import Any, Sequence

from floeval.api.dataset_loaders.base import BaseDatasetLoader
from floeval.api.dataset_loaders.local_file_loader import (
    get_conversational_loader_for_file,
    get_loader_for_file,
)
from floeval.config.schemas.io.conversational_dataset import (
    ConversationalDataset,
    ConversationalSample,
    PartialConversationalDataset,
    PartialConversationalSample,
)
from floeval.config.schemas.io.dataset import (
    Dataset,
    PartialDataset,
    PartialSample,
    Sample,
)


def conversational_dataset_from_dict(
    data: dict[str, Any],
    partial_dataset: bool,
) -> ConversationalDataset | PartialConversationalDataset:
    """Build conversational dataset from dict with top-level ``samples``."""
    raw = data.get("samples")
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            "Conversational dataset dict must contain a non-empty 'samples' array."
        )
    if partial_dataset:
        partial_conv_samples = [PartialConversationalSample(**s) for s in raw]
        return PartialConversationalDataset(samples=partial_conv_samples)
    conv_samples = [ConversationalSample(**s) for s in raw]
    return ConversationalDataset(samples=conv_samples)


def conversational_dataset_from_json(
    path: str | Path,
    partial_dataset: bool,
) -> ConversationalDataset | PartialConversationalDataset:
    """Load conversational dataset from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return conversational_dataset_from_dict(data, partial_dataset=partial_dataset)


def conversational_dataset_from_file(
    ds_path: str | Path,
    partial_dataset: bool,
) -> ConversationalDataset | PartialConversationalDataset:
    """Load conversational dataset from file (JSON uses ``samples`` key)."""
    ds_path = Path(ds_path)
    if not ds_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {ds_path}")
    loader_cls: type[BaseDatasetLoader] = get_conversational_loader_for_file(str(ds_path))
    if partial_dataset:
        samples = loader_cls.to_partial_samples(str(ds_path))
        return PartialConversationalDataset(samples=samples)
    samples = loader_cls.to_samples(str(ds_path))
    return ConversationalDataset(samples=samples)


def dataset_from_dict(data: dict[str, Any], partial_dataset: bool) -> Dataset | PartialDataset:
    """Create single-turn dataset from a dict object.

    Dataset should be of the form:
    {
        "samples": [
            {
                "user_input": "What is the capital of France?",
                "llm_response": "The capital of France is Paris."
            },
            ...
        ]
    }
    Note: In case of partial dataset, the llm_response field can be omitted or set to empty string.
    The loader will handle it accordingly.

    Args:
        data: A dict containing the dataset information, typically loaded from a JSON file.
        partial_dataset: flag to return PartialDataset instead of Dataset.

    Returns:
        A Dataset or PartialDataset instance.
    """
    raw_samples = data.get("samples", [])
    if partial_dataset:
        samples = [PartialSample(**s) for s in raw_samples]
        return PartialDataset(samples=samples)
    samples = [Sample(**s) for s in raw_samples]
    return Dataset(samples=samples)


def dataset_from_json(path: str | Path, partial_dataset: bool) -> Dataset | PartialDataset:
    """Load dataset from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return dataset_from_dict(data, partial_dataset=partial_dataset)


def dataset_from_file(ds_path: str | Path, partial_dataset: bool) -> Dataset | PartialDataset:
    """Convenience method to load dataset from file with type detection."""
    ds_path = Path(ds_path)
    if not ds_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {ds_path}")

    loader_cls: type[BaseDatasetLoader] = get_loader_for_file(str(ds_path))
    if partial_dataset:
        samples = loader_cls.to_partial_samples(str(ds_path))
        return PartialDataset(samples=samples)
    samples = loader_cls.to_samples(str(ds_path))
    return Dataset(samples=samples)


def dataset_from_samples(
    samples: Sequence[Sample | dict[str, Any]], partial_dataset: bool
) -> Dataset | PartialDataset:
    """Convenience helper: build dataset from Sample objects or flat dicts.

    Args:
        samples: Sequence of Sample/PartialSample objects or dicts convertible to Sample.
        partial_dataset: If True, convert to PartialSample and return PartialDataset.

    Returns:
        Dataset or PartialDataset containing the provided samples.
    """
    if partial_dataset:
        normalized_samples = []
        for s in samples:
            if isinstance(s, Sample):
                normalized_samples.append(PartialSample(**s.model_dump()))
            elif isinstance(s, PartialSample):
                normalized_samples.append(s)
            elif isinstance(s, dict):
                normalized_samples.append(PartialSample(**s))
            else:
                raise ValueError(f"Invalid sample type: {type(s)}; expected PartialSample, or dict")
        return PartialDataset(samples=normalized_samples)

    normalized_samples = []
    for s in samples:
        if isinstance(s, Sample):
            normalized_samples.append(s)
        elif isinstance(s, dict):
            normalized_samples.append(Sample(**s))
        else:
            raise ValueError(f"Invalid sample type: {type(s)}; expected Sample or dict")
    return Dataset(samples=normalized_samples)


class DatasetLoader:
    """Unified interface for loading datasets from various sources."""

    @staticmethod
    def from_dict(data: dict[str, Any], partial_dataset: bool = False) -> Dataset | PartialDataset:
        """Create dataset from a dict shaped like PRD examples."""
        return dataset_from_dict(data, partial_dataset=partial_dataset)

    @staticmethod
    def from_json(path: str | Path, partial_dataset: bool = False) -> Dataset | PartialDataset:
        """Load dataset from JSON file."""
        return dataset_from_json(path, partial_dataset=partial_dataset)

    @staticmethod
    def from_file(ds_path: str | Path, partial_dataset: bool = False) -> Dataset | PartialDataset:
        """Convenience method to load dataset from file with type detection."""
        return dataset_from_file(ds_path, partial_dataset=partial_dataset)

    @staticmethod
    def from_samples(
        samples: Sequence[Sample | dict[str, Any]], partial_dataset: bool = False
    ) -> Dataset | PartialDataset:
        """Build dataset from Sample objects or flat dicts."""
        return dataset_from_samples(samples, partial_dataset=partial_dataset)

    @staticmethod
    def conversational_from_dict(
        data: dict[str, Any], partial_dataset: bool = False
    ) -> ConversationalDataset | PartialConversationalDataset:
        """Load ``ConversationalDataset`` / ``PartialConversationalDataset`` from a dict."""
        return conversational_dataset_from_dict(data, partial_dataset=partial_dataset)

    @staticmethod
    def conversational_from_file(
        ds_path: str | Path, partial_dataset: bool = False
    ) -> ConversationalDataset | PartialConversationalDataset:
        """Load conversational dataset JSON / JSONL via ``samples`` rows."""
        return conversational_dataset_from_file(ds_path, partial_dataset=partial_dataset)
