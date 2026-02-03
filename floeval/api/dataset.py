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
        """
        Convenience helper: build dataset from Sample objects or flat dicts.
        
        Accepts flat format: {"question": "...", "answer": "...", "contexts": [...], "expected_answer": "..."}
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
                    "question": flat.get("question", ""),
                    "answer": flat.get("answer", ""),
                }
                if "contexts" in flat:
                    contexts = flat["contexts"]
                    inputs["contexts"] = contexts if isinstance(contexts, list) else [contexts]
                
                ground_truth = None
                if "expected_answer" in flat and flat["expected_answer"] is not None:
                    ground_truth = {"expected_answer": flat["expected_answer"]}
                
                parsed.append(Sample(inputs=inputs, ground_truth=ground_truth, metadata=flat.get("metadata", {})))
            else:
                raise TypeError(f"Sample must be Sample instance or dict, got {type(s)}")
        return cls(samples=parsed)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

