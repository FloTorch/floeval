"""Dataset and Sample classes.
The RAGAS adapter supports Pydantic models via `model_dump()`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

from floeval.api.dataset_loaders.base import BaseDatasetLoader
from floeval.api.dataset_loaders.local_file_loader import get_loader_for_file
from floeval.config.schemas.io.dataset import Dataset, Sample


def dataset_from_dict(data: Dict[str, Any]) -> Dataset:
    """Create dataset from a dict shaped like PRD examples."""
    raw_samples = data.get("samples", [])
    samples = [Sample(**s) for s in raw_samples]
    return Dataset(samples=samples)


def dataset_from_json(path: Union[str, Path]) -> Dataset:
    """Load dataset from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return dataset_from_dict(data)


def dataset_from_file(ds_path: Union[str, Path]) -> Dataset:
    """Convenience method to load dataset from file with type detection."""
    ds_path = Path(ds_path)
    if not ds_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {ds_path}")

    loader_cls: type[BaseDatasetLoader] = get_loader_for_file(str(ds_path))
    samples = loader_cls.to_samples(ds_path)
    return Dataset(samples=samples)


def dataset_from_samples(samples: Sequence[Union[Sample, Dict[str, Any]]]) -> Dataset:
    """
    Convenience helper: build dataset from Sample objects or flat dicts.

    Accepts flat format: {"user_input": "...", "llm_response": "...", "contexts": [...], "ground_truth": "..."}
    Normalizes to Sample structure internally.
    """
    parsed: List[Sample] = []
    for s in samples:
        if isinstance(s, Sample):
            parsed.append(s)
        elif isinstance(s, dict):
            # Flat format: normalize to nested structure
            flat = s
            inputs = {
                "user_input": flat.get("user_input", ""),
                "llm_response": flat.get("llm_response", ""),
                "contexts": flat.get("contexts", []),
                "ground_truth": flat.get("ground_truth", None),
            }
            _sample = Sample.model_validate(inputs)
            parsed.append(_sample)
        else:
            raise TypeError(f"Sample must be Sample instance or dict, got {type(s)}")
    return Dataset(samples=parsed)


class DatasetLoader:
    """Unified interface for loading datasets from various sources."""

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Dataset:
        """Create dataset from a dict shaped like PRD examples."""
        return dataset_from_dict(data)

    @staticmethod
    def from_json(path: Union[str, Path]) -> Dataset:
        """Load dataset from JSON file."""
        return dataset_from_json(path)

    @staticmethod
    def from_file(ds_path: Union[str, Path]) -> Dataset:
        """Convenience method to load dataset from file with type detection."""
        return dataset_from_file(ds_path)

    @staticmethod
    def from_samples(samples: Sequence[Union[Sample, Dict[str, Any]]]) -> Dataset:
        """Build dataset from Sample objects or flat dicts."""
        return dataset_from_samples(samples)
