"""Dataset and Sample classes.
The RAGAS adapter supports Pydantic models via `model_dump()`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence, Union

from floeval.api.dataset_loaders.base import BaseDatasetLoader
from floeval.api.dataset_loaders.local_file_loader import get_loader_for_file
from floeval.config.schemas.io.dataset import (
    Dataset,
    PartialDataset,
    PartialSample,
    Sample,
)


def dataset_from_dict(
    data: Dict[str, Any], partial_dataset: bool
) -> Dataset | PartialDataset:
    """Create dataset from a dict object.

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


def dataset_from_json(
    path: Union[str, Path], partial_dataset: bool
) -> Dataset | PartialDataset:
    """Load dataset from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return dataset_from_dict(data, partial_dataset=partial_dataset)


def dataset_from_file(
    ds_path: Union[str, Path], partial_dataset: bool
) -> Dataset | PartialDataset:
    """Convenience method to load dataset from file with type detection."""
    ds_path = Path(ds_path)
    if not ds_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {ds_path}")

    loader_cls: type[BaseDatasetLoader] = get_loader_for_file(str(ds_path))
    if partial_dataset:
        samples = loader_cls.to_partial_samples(ds_path)
        return PartialDataset(samples=samples)
    else:
        samples = loader_cls.to_samples(ds_path)
        return Dataset(samples=samples)


def dataset_from_samples(
    samples: Sequence[Union[Sample, Dict[str, Any]]], partial_dataset: bool
) -> Dataset | PartialDataset:
    """Convenience helper: build dataset from Sample objects or flat dicts.

    Args:
        samples: A sequence of Sample/PartialSample objects or dicts that can be converted to Sample/PartialSample.
        partial_dataset: If True, will convert to PartialSample and return PartialDataset; otherwise returns Dataset.

    Returns:
        Dataset | PartialDataset: A Dataset or PartialDataset object containing the provided samples.
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
                raise ValueError(
                    f"Invalid sample type: {type(s)}; expected PartialSample, or dict"
                )
        return PartialDataset(samples=normalized_samples)

    # --- complete dataset case (containing llm_response field) --
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
    def from_dict(
        data: Dict[str, Any], partial_dataset=False
    ) -> Dataset | PartialDataset:
        """Create dataset from a dict shaped like PRD examples."""
        return dataset_from_dict(data, partial_dataset=partial_dataset)

    @staticmethod
    def from_json(
        path: Union[str, Path], partial_dataset=False
    ) -> Dataset | PartialDataset:
        """Load dataset from JSON file."""
        return dataset_from_json(path, partial_dataset=partial_dataset)

    @staticmethod
    def from_file(
        ds_path: Union[str, Path], partial_dataset=False
    ) -> Dataset | PartialDataset:
        """Convenience method to load dataset from file with type detection."""
        return dataset_from_file(ds_path, partial_dataset=partial_dataset)

    @staticmethod
    def from_samples(
        samples: Sequence[Union[Sample, Dict[str, Any]]], partial_dataset=False
    ) -> Dataset | PartialDataset:
        """Build dataset from Sample objects or flat dicts."""
        return dataset_from_samples(samples, partial_dataset=partial_dataset)
