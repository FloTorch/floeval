"""Dataset and Sample classes.
The RAGAS adapter supports Pydantic models via `model_dump()`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union
import json

from pydantic import BaseModel, Field, field_validator


class Sample(BaseModel):
    """Single evaluation sample."""

    inputs: Dict[str, Any] = Field(..., description="Model inputs (question/contexts/answer etc.)")
    ground_truth: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional ground truth/reference information"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional sample metadata")


class Dataset(BaseModel):
    """Collection of samples."""

    samples: List[Sample] = Field(..., min_length=1, description="Non-empty list of samples")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional dataset metadata")

    @field_validator("samples")
    @classmethod
    def _validate_samples(cls, v: List[Sample]) -> List[Sample]:
        if not v:
            raise ValueError("Dataset must contain at least one sample.")
        return v

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Dataset":
        """Create dataset from a dict shaped like PRD examples."""
        raw_samples = data.get("samples", [])
        samples = [Sample(**s) for s in raw_samples]
        return cls(samples=samples, metadata=data.get("metadata", {}))

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "Dataset":
        """Load dataset from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_samples(cls, samples: Sequence[Union[Sample, Dict[str, Any]]]) -> "Dataset":
        """Convenience helper: build dataset from Sample objects or dicts."""
        parsed: List[Sample] = []
        for s in samples:
            parsed.append(s if isinstance(s, Sample) else Sample(**s))
        return cls(samples=parsed)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

