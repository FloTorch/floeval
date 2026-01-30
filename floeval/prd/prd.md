# EvalKit - Complete Production PRD v2.0
## AI Code Editor Ready - Full Implementation Guide

**Version:** 2.0 Final  
**Status:** Production-Ready Architecture  
**Timeline:** 2-3 Days Implementation  
**Last Updated:** January 2026

---

## 📋 Table of Contents

1. [Executive Summary](#executive-summary)
2. [Folder Structure](#folder-structure)
3. [Dependencies](#dependencies)
4. [Core Implementation - Complete Files](#core-implementation)
5. [Examples - Complete Files](#examples)
6. [Tests - Complete Files](#tests)
7. [Implementation Checklist](#implementation-checklist)
8. [Demo Script](#demo-script)

---

## Executive Summary

### What We're Building

A **config-driven, multi-backend evaluation framework** that allows users to:

✅ **Swap metric backends** - Same metric via RAGAS, DeepEval, or custom  
✅ **Flexible specification** - Metrics as instances, strings, dicts, or YAML  
✅ **Smart resolution** - Automatic provider selection with clear disambiguation  
✅ **Future-proof** - Ready for two-phase execution without refactor  
✅ **CI/CD ready** - YAML/JSON configs work out of the box  

### Key Architectural Decisions

1. **`metric_providers/`** instead of `adapters/` - Clearer semantics
2. **Two-level MetricRegistry** - `(provider, metric_id) → MetricClass`
3. **Three metric specs** - Instance, string, dict (all supported)
4. **Smart `resolve_best()`** - Deterministic provider selection
5. **Simplified backend** - Execution only, no metrics

---

## Folder Structure

```
evalkit/
│
├── api/                          # PUBLIC API
│   ├── __init__.py
│   ├── evaluation.py
│   ├── dataset.py
│   └── metrics/
│       ├── __init__.py
│       ├── base.py
│       └── registry.py
│
├── backend/                      # INTERNAL (Future: execution)
│   ├── __init__.py
│   └── execution/
│       ├── __init__.py
│       └── trace.py
│
├── metric_providers/             # ALL METRICS
│   ├── __init__.py
│   ├── builtin/
│   │   ├── __init__.py
│   │   └── metrics.py
│   ├── ragas/
│   │   ├── __init__.py
│   │   ├── adapter.py
│   │   └── metrics.py
│   └── deepeval/
│       ├── __init__.py
│       ├── adapter.py
│       └── metrics.py
│
├── config/
│   ├── __init__.py
│   └── schemas.py
│
├── utils/
│   ├── __init__.py
│   └── loaders.py
│
├── examples/
│   ├── 01_basic.py
│   ├── 02_config_driven.py
│   ├── 03_comparison.py
│   ├── 04_yaml.py
│   └── data/
│       ├── dataset.json
│       └── config.yaml
│
├── tests/
│   ├── conftest.py
│   ├── test_evaluation.py
│   ├── test_registry.py
│   └── test_providers/
│       ├── test_ragas.py
│       └── test_deepeval.py
│
├── pyproject.toml
├── requirements.txt
├── README.md
└── .env.example
```

---

## Dependencies

### `requirements.txt`

```txt
# Core
python>=3.10
pydantic>=2.0.0
openai>=1.0.0

# Metric providers
ragas>=0.1.0
deepeval>=0.20.0

# Utilities
pandas>=2.0.0
pyyaml>=6.0.0
python-dotenv>=1.0.0

# Development
pytest>=7.0.0
black>=23.0.0
mypy>=1.0.0
ruff>=0.1.0
```

### `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "evalkit"
version = "0.2.0"
description = "Config-driven multi-backend evaluation framework"
requires-python = ">=3.10"
dependencies = [
    "pydantic>=2.0.0",
    "openai>=1.0.0",
    "ragas>=0.1.0",
    "deepeval>=0.20.0",
    "pandas>=2.0.0",
    "pyyaml>=6.0.0",
    "python-dotenv>=1.0.0"
]

[project.optional-dependencies]
dev = ["pytest>=7.0.0", "black>=23.0.0", "mypy>=1.0.0", "ruff>=0.1.0"]

[tool.black]
line-length = 88

[tool.mypy]
python_version = "3.10"
strict = true

[tool.ruff]
line-length = 88
```

### `.env.example`

```bash
# OpenAI API Key (required for RAGAS and DeepEval)
OPENAI_API_KEY=your_openai_key_here
```

---

## Core Implementation

### File: `config/schemas.py`

```python
"""Pydantic configuration schemas."""

from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field


class MetricSpecSchema(BaseModel):
    """Metric specification in config files."""
    
    id: str = Field(..., description="Metric ID")
    provider: Optional[str] = Field(None, description="Provider name")
    params: Dict[str, Any] = Field(default_factory=dict)


class EvaluationConfigSchema(BaseModel):
    """Complete evaluation configuration for YAML/JSON."""
    
    dataset: Union[str, Dict[str, Any]]
    metrics: List[Union[str, MetricSpecSchema]]
    default_provider: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

---

### File: `api/dataset.py`

```python
"""Dataset and Sample classes."""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from pydantic import BaseModel, Field, validator
import json
import pandas as pd


class Sample(BaseModel):
    """Single test case."""
    
    inputs: Dict[str, Any] = Field(..., description="Inputs")
    ground_truth: Optional[Dict[str, Any]] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Dataset(BaseModel):
    """Collection of samples."""
    
    samples: List[Sample] = Field(..., min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('samples')
    def validate_samples(cls, v):
        if not v:
            raise ValueError("Dataset must have samples")
        return v
    
    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "Dataset":
        """Load from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_csv(cls, path: Union[str, Path]) -> "Dataset":
        """Load from CSV."""
        df = pd.read_csv(path)
        samples = []
        for _, row in df.iterrows():
            contexts = json.loads(row['contexts']) if 'contexts' in row else []
            sample = Sample(
                inputs={
                    "question": row['question'],
                    "contexts": contexts,
                    "answer": row['answer']
                },
                ground_truth={"expected_answer": row.get('expected_answer')} if 'expected_answer' in row else None
            )
            samples.append(sample)
        return cls(samples=samples)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Dataset":
        """Create from dict."""
        samples = [Sample(**s) for s in data.get('samples', [])]
        return cls(samples=samples, metadata=data.get('metadata', {}))
    
    def __len__(self):
        return len(self.samples)
    
    def __iter__(self):
        return iter(self.samples)
```

---

### File: `api/metrics/base.py`

```python
"""Base metric interface."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class MetricResult(BaseModel):
    """Result of metric evaluation."""
    
    metric_name: str
    score: float = Field(..., ge=0.0, le=1.0)
    passed: bool
    reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    provider: str


class BaseMetric(ABC):
    """Abstract base for all metrics."""
    
    name: str = "base"
    provider: str = "unknown"
    threshold: float = 0.7
    
    @abstractmethod
    def evaluate(self, sample: Any) -> MetricResult:
        """Evaluate sample."""
        pass
    
    def __call__(self, sample: Any) -> MetricResult:
        return self.evaluate(sample)
```

---

### File: `api/metrics/registry.py`

```python
"""Two-level metric registry with smart resolution."""

from typing import Dict, Type, List, Optional
from .base import BaseMetric


class MetricRegistry:
    """
    Two-level registry: (provider, metric_id) -> MetricClass
    
    Examples:
        @MetricRegistry.register(provider="ragas", metric_id="answer_relevancy")
        class RAGASAnswerRelevancy(BaseMetric):
            pass
    """
    
    _metrics: Dict[str, Dict[str, Type[BaseMetric]]] = {}
    
    @classmethod
    def register(cls, provider: str, metric_id: str):
        """Register metric."""
        def decorator(metric_class: Type[BaseMetric]):
            if not issubclass(metric_class, BaseMetric):
                raise TypeError(f"{metric_class} must inherit BaseMetric")
            
            if provider not in cls._metrics:
                cls._metrics[provider] = {}
            
            if metric_id in cls._metrics[provider]:
                raise ValueError(f"Metric '{metric_id}' already registered for '{provider}'")
            
            cls._metrics[provider][metric_id] = metric_class
            return metric_class
        return decorator
    
    @classmethod
    def get(cls, provider: str, metric_id: str, **params) -> BaseMetric:
        """Get metric instance."""
        if provider not in cls._metrics:
            raise KeyError(f"Provider '{provider}' not found. Available: {list(cls._metrics.keys())}")
        
        if metric_id not in cls._metrics[provider]:
            raise KeyError(f"Metric '{metric_id}' not in '{provider}'. Available: {list(cls._metrics[provider].keys())}")
        
        return cls._metrics[provider][metric_id](**params)
    
    @classmethod
    def resolve_best(cls, metric_id: str, default_provider: Optional[str] = None) -> str:
        """
        Resolve best provider for metric_id.
        
        Logic:
        1. If one provider has it -> use that
        2. If multiple -> use default_provider if it has it
        3. Else -> raise ambiguity error
        """
        providers = [p for p in cls._metrics if metric_id in cls._metrics[p]]
        
        if len(providers) == 0:
            raise KeyError(f"No provider has '{metric_id}'. Available: {cls.list_all_metrics()}")
        
        if len(providers) == 1:
            return providers[0]
        
        # Multiple providers
        if default_provider and default_provider in providers:
            return default_provider
        
        raise ValueError(
            f"'{metric_id}' ambiguous. Found in: {providers}. "
            f"Use 'provider:metric_id' or set default_provider."
        )
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """List all providers."""
        return list(cls._metrics.keys())
    
    @classmethod
    def list_metrics(cls, provider: Optional[str] = None) -> Dict[str, List[str]]:
        """List metrics by provider."""
        if provider:
            return {provider: list(cls._metrics.get(provider, {}).keys())}
        return {p: list(m.keys()) for p, m in cls._metrics.items()}
    
    @classmethod
    def list_all_metrics(cls) -> List[str]:
        """List all metric IDs."""
        all_metrics = set()
        for provider_metrics in cls._metrics.values():
            all_metrics.update(provider_metrics.keys())
        return sorted(all_metrics)
    
    @classmethod
    def clear(cls):
        """Clear registry (for testing)."""
        cls._metrics.clear()
```

---

### File: `metric_providers/ragas/adapter.py`

```python
"""RAGAS adapter."""

from typing import Dict, Any
from ragas import SingleTurnSample


def sample_to_ragas(sample_data: Dict[str, Any]) -> SingleTurnSample:
    """Convert EvalKit sample to RAGAS format."""
    inputs = sample_data.get("inputs", {})
    gt = sample_data.get("ground_truth", {})
    
    return SingleTurnSample(
        user_input=inputs.get("question", ""),
        retrieved_contexts=inputs.get("contexts", []),
        response=inputs.get("answer", ""),
        reference=gt.get("expected_answer") if gt else None
    )
```

---

### File: `metric_providers/ragas/metrics.py`

```python
"""RAGAS metric implementations."""

import asyncio
from ragas.metrics import answer_relevancy, faithfulness

from evalkit.api.metrics.base import BaseMetric, MetricResult
from evalkit.api.metrics.registry import MetricRegistry
from evalkit.api.dataset import Sample
from .adapter import sample_to_ragas


@MetricRegistry.register(provider="ragas", metric_id="answer_relevancy")
class RAGASAnswerRelevancy(BaseMetric):
    """RAGAS Answer Relevancy."""
    
    def __init__(self, threshold: float = 0.7, **kwargs):
        self.name = "answer_relevancy"
        self.provider = "ragas"
        self.threshold = threshold
        self.ragas_metric = answer_relevancy
    
    def evaluate(self, sample: Sample) -> MetricResult:
        ragas_sample = sample_to_ragas(sample.dict())
        score = asyncio.run(self.ragas_metric.single_turn_ascore(ragas_sample))
        
        return MetricResult(
            metric_name=self.name,
            score=float(score),
            passed=score >= self.threshold,
            reason=f"RAGAS {self.name}: {score:.3f}",
            provider=self.provider,
            metadata={"threshold": self.threshold}
        )


@MetricRegistry.register(provider="ragas", metric_id="faithfulness")
class RAGASFaithfulness(BaseMetric):
    """RAGAS Faithfulness."""
    
    def __init__(self, threshold: float = 0.7, **kwargs):
        self.name = "faithfulness"
        self.provider = "ragas"
        self.threshold = threshold
        self.ragas_metric = faithfulness
    
    def evaluate(self, sample: Sample) -> MetricResult:
        ragas_sample = sample_to_ragas(sample.dict())
        score = asyncio.run(self.ragas_metric.single_turn_ascore(ragas_sample))
        
        return MetricResult(
            metric_name=self.name,
            score=float(score),
            passed=score >= self.threshold,
            reason=f"RAGAS {self.name}: {score:.3f}",
            provider=self.provider,
            metadata={"threshold": self.threshold}
        )
```

---

### File: `metric_providers/deepeval/adapter.py`

```python
"""DeepEval adapter."""

from typing import Dict, Any
from deepeval.test_case import LLMTestCase


def sample_to_deepeval(sample_data: Dict[str, Any]) -> LLMTestCase:
    """Convert EvalKit sample to DeepEval format."""
    inputs = sample_data.get("inputs", {})
    gt = sample_data.get("ground_truth", {})
    
    return LLMTestCase(
        input=inputs.get("question", ""),
        actual_output=inputs.get("answer", ""),
        expected_output=gt.get("expected_answer") if gt else None,
        context=inputs.get("contexts", [])
    )
```

---

### File: `metric_providers/deepeval/metrics.py`

```python
"""DeepEval metric implementations."""

import asyncio
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

from evalkit.api.metrics.base import BaseMetric, MetricResult
from evalkit.api.metrics.registry import MetricRegistry
from evalkit.api.dataset import Sample
from .adapter import sample_to_deepeval


@MetricRegistry.register(provider="deepeval", metric_id="answer_relevancy")
class DeepEvalAnswerRelevancy(BaseMetric):
    """DeepEval Answer Relevancy."""
    
    def __init__(self, threshold: float = 0.7, **kwargs):
        self.name = "answer_relevancy"
        self.provider = "deepeval"
        self.threshold = threshold
        self.deepeval_metric = AnswerRelevancyMetric(threshold=threshold)
    
    def evaluate(self, sample: Sample) -> MetricResult:
        test_case = sample_to_deepeval(sample.dict())
        asyncio.run(self.deepeval_metric.a_measure(test_case))
        
        score = self.deepeval_metric.score
        passed = self.deepeval_metric.is_successful()
        
        return MetricResult(
            metric_name=self.name,
            score=float(score),
            passed=passed,
            reason=f"DeepEval {self.name}: {score:.3f}",
            provider=self.provider,
            metadata={"threshold": self.threshold}
        )


@MetricRegistry.register(provider="deepeval", metric_id="faithfulness")
class DeepEvalFaithfulness(BaseMetric):
    """DeepEval Faithfulness."""
    
    def __init__(self, threshold: float = 0.7, **kwargs):
        self.name = "faithfulness"
        self.provider = "deepeval"
        self.threshold = threshold
        self.deepeval_metric = FaithfulnessMetric(threshold=threshold)
    
    def evaluate(self, sample: Sample) -> MetricResult:
        test_case = sample_to_deepeval(sample.dict())
        asyncio.run(self.deepeval_metric.a_measure(test_case))
        
        score = self.deepeval_metric.score
        passed = self.deepeval_metric.is_successful()
        
        return MetricResult(
            metric_name=self.name,
            score=float(score),
            passed=passed,
            reason=f"DeepEval {self.name}: {score:.3f}",
            provider=self.provider,
            metadata={"threshold": self.threshold}
        )
```

---

### File: `api/evaluation.py`

```python
"""Main Evaluation class with flexible metric specification."""

from typing import List, Union, Dict, Any, Optional
from pydantic import BaseModel, Field
import yaml
from pathlib import Path

from .dataset import Dataset, Sample
from .metrics.base import BaseMetric, MetricResult
from .metrics.registry import MetricRegistry


MetricSpec = Union[BaseMetric, str, Dict[str, Any]]


class EvaluationResult(BaseModel):
    """Evaluation results."""
    
    sample_results: List[Dict[str, Any]]
    aggregate_scores: Dict[str, float]
    summary: Dict[str, Any] = Field(default_factory=dict)


class Evaluation:
    """
    Main evaluation orchestrator.
    
    Supports three metric formats:
    1. Instance: MetricRegistry.get("ragas", "answer_relevancy")
    2. String: "answer_relevancy" or "ragas:answer_relevancy"
    3. Dict: {"id": "answer_relevancy", "provider": "ragas", "params": {...}}
    
    Examples:
        # Format 1: Instances
        evaluation = Evaluation(
            dataset=dataset,
            metrics=[MetricRegistry.get("ragas", "answer_relevancy")]
        )
        
        # Format 2: Strings with default provider
        evaluation = Evaluation(
            dataset=dataset,
            metrics=["answer_relevancy", "faithfulness"],
            default_provider="ragas"
        )
        
        # Format 3: Mixed
        evaluation = Evaluation(
            dataset=dataset,
            metrics=[
                "ragas:answer_relevancy",
                "deepeval:faithfulness"
            ]
        )
    """
    
    def __init__(
        self,
        dataset: Dataset,
        metrics: List[MetricSpec],
        default_provider: Optional[str] = None
    ):
        self.dataset = dataset
        self.default_provider = default_provider
        self.metrics = self._resolve_metrics(metrics)
    
    def _resolve_metrics(self, specs: List[MetricSpec]) -> List[BaseMetric]:
        """Resolve metric specs to instances."""
        resolved = []
        
        for spec in specs:
            # Case 1: Already an instance
            if isinstance(spec, BaseMetric):
                resolved.append(spec)
                continue
            
            # Case 2: Dict
            if isinstance(spec, dict):
                metric_id = spec.get("id")
                provider = spec.get("provider")
                params = spec.get("params", {})
                
                if not provider:
                    provider = self.default_provider or MetricRegistry.resolve_best(metric_id, self.default_provider)
                
                metric = MetricRegistry.get(provider, metric_id, **params)
                resolved.append(metric)
                continue
            
            # Case 3: String
            if isinstance(spec, str):
                if ":" in spec:
                    provider, metric_id = spec.split(":", 1)
                else:
                    metric_id = spec
                    provider = self.default_provider or MetricRegistry.resolve_best(metric_id, self.default_provider)
                
                metric = MetricRegistry.get(provider, metric_id)
                resolved.append(metric)
                continue
            
            raise TypeError(f"Invalid metric spec: {spec}")
        
        return resolved
    
    def run(self) -> EvaluationResult:
        """Run evaluation."""
        sample_results = []
        
        for sample in self.dataset:
            metric_results = {}
            
            for metric in self.metrics:
                result = metric.evaluate(sample)
                key = f"{result.provider}:{result.metric_name}"
                metric_results[key] = {
                    "score": result.score,
                    "passed": result.passed,
                    "reason": result.reason,
                    "provider": result.provider
                }
            
            sample_results.append({
                "inputs": sample.inputs,
                "ground_truth": sample.ground_truth,
                "metrics": metric_results
            })
        
        aggregate_scores = self._aggregate(sample_results)
        summary = self._summarize(sample_results, aggregate_scores)
        
        return EvaluationResult(
            sample_results=sample_results,
            aggregate_scores=aggregate_scores,
            summary=summary
        )
    
    def _aggregate(self, results: List[Dict]) -> Dict[str, float]:
        """Aggregate scores across samples."""
        scores = {}
        for result in results:
            for key, data in result["metrics"].items():
                if key not in scores:
                    scores[key] = []
                scores[key].append(data["score"])
        
        return {key: sum(vals) / len(vals) for key, vals in scores.items()}
    
    def _summarize(self, results: List[Dict], agg: Dict[str, float]) -> Dict[str, Any]:
        """Compute summary."""
        total = len(results)
        passes = {}
        providers = set()
        
        for result in results:
            for key, data in result["metrics"].items():
                if key not in passes:
                    passes[key] = 0
                if data["passed"]:
                    passes[key] += 1
                providers.add(data["provider"])
        
        return {
            "total_samples": total,
            "providers_used": list(providers),
            "pass_rates": {k: v / total for k, v in passes.items()},
            "aggregate_scores": agg
        }
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Evaluation":
        """Load from YAML config."""
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load dataset
        dataset_config = config.get("dataset")
        if isinstance(dataset_config, str):
            dataset = Dataset.from_json(dataset_config)
        else:
            dataset = Dataset.from_dict(dataset_config)
        
        # Parse metrics
        metrics = config.get("metrics", [])
        default_provider = config.get("default_provider")
        
        return cls(
            dataset=dataset,
            metrics=metrics,
            default_provider=default_provider
        )
```

---

### File: `api/__init__.py`

```python
"""Public API exports."""

from .evaluation import Evaluation, EvaluationResult
from .dataset import Dataset, Sample
from .metrics.base import BaseMetric, MetricResult
from .metrics.registry import MetricRegistry

__all__ = [
    "Evaluation",
    "EvaluationResult",
    "Dataset",
    "Sample",
    "BaseMetric",
    "MetricResult",
    "MetricRegistry"
]
```

---

### File: `metric_providers/__init__.py`

```python
"""Metric providers package."""

# Auto-import to register metrics
from . import ragas
from . import deepeval

__all__ = ["ragas", "deepeval"]
```

---

### File: `metric_providers/ragas/__init__.py`

```python
"""RAGAS provider."""

from .metrics import RAGASAnswerRelevancy, RAGASFaithfulness

__all__ = ["RAGASAnswerRelevancy", "RAGASFaithfulness"]
```

---

### File: `metric_providers/deepeval/__init__.py`

```python
"""DeepEval provider."""

from .metrics import DeepEvalAnswerRelevancy, DeepEvalFaithfulness

__all__ = ["DeepEvalAnswerRelevancy", "DeepEvalFaithfulness"]
```

---

## Examples

### File: `examples/data/dataset.json`

```json
{
  "samples": [
    {
      "inputs": {
        "question": "What is RAG?",
        "contexts": [
          "RAG combines retrieval with generation.",
          "It retrieves documents then generates responses."
        ],
        "answer": "RAG is a technique combining retrieval and generation for better AI responses."
      },
      "ground_truth": {
        "expected_answer": "RAG combines retrieval and generation."
      }
    },
    {
      "inputs": {
        "question": "How does RAG reduce hallucinations?",
        "contexts": [
          "RAG grounds responses in retrieved documents.",
          "This prevents making up information."
        ],
        "answer": "RAG reduces hallucinations by grounding responses in real documents."
      },
      "ground_truth": {
        "expected_answer": "RAG grounds responses in factual documents."
      }
    }
  ]
}
```

---

### File: `examples/data/config.yaml`

```yaml
dataset: examples/data/dataset.json

metrics:
  - answer_relevancy
  - faithfulness

default_provider: ragas

metadata:
  name: "RAG Evaluation"
  version: "1.0"
```

---

### File: `examples/01_basic.py`

```python
"""Basic evaluation example."""

from evalkit.api import Evaluation, Dataset, MetricRegistry

# Load dataset
dataset = Dataset.from_json("examples/data/dataset.json")

# Create evaluation
evaluation = Evaluation(
    dataset=dataset,
    metrics=[
        MetricRegistry.get("ragas", "answer_relevancy"),
        MetricRegistry.get("ragas", "faithfulness")
    ]
)

# Run
results = evaluation.run()

# Print
print("=== Results ===")
print(f"Samples: {results.summary['total_samples']}")
print(f"Providers: {results.summary['providers_used']}")
print("\nScores:")
for metric, score in results.aggregate_scores.items():
    print(f"  {metric}: {score:.3f}")
```

---

### File: `examples/02_config_driven.py`

```python
"""Config-driven evaluation."""

from evalkit.api import Evaluation, Dataset

# Load dataset
dataset = Dataset.from_json("examples/data/dataset.json")

# Config-driven: strings + default provider
evaluation = Evaluation(
    dataset=dataset,
    metrics=["answer_relevancy", "faithfulness"],
    default_provider="ragas"
)

results = evaluation.run()

print("=== Config-Driven Results ===")
for metric, score in results.aggregate_scores.items():
    print(f"{metric}: {score:.3f}")
```

---

### File: `examples/03_comparison.py`

```python
"""Backend comparison."""

from evalkit.api import Evaluation, Dataset

dataset = Dataset.from_json("examples/data/dataset.json")

# RAGAS backend
eval_ragas = Evaluation(
    dataset=dataset,
    metrics=["ragas:answer_relevancy", "ragas:faithfulness"]
)
results_ragas = eval_ragas.run()

# DeepEval backend
eval_deepeval = Evaluation(
    dataset=dataset,
    metrics=["deepeval:answer_relevancy", "deepeval:faithfulness"]
)
results_deepeval = eval_deepeval.run()

# Compare
print("="*60)
print("BACKEND COMPARISON")
print("="*60)
print("\nRAGAS:")
for k, v in results_ragas.aggregate_scores.items():
    print(f"  {k}: {v:.3f}")

print("\nDeepEval:")
for k, v in results_deepeval.aggregate_scores.items():
    print(f"  {k}: {v:.3f}")
```

---

### File: `examples/04_yaml.py`

```python
"""YAML config evaluation."""

from evalkit.api import Evaluation

# Load from YAML
evaluation = Evaluation.from_yaml("examples/data/config.yaml")

# Run
results = evaluation.run()

print("=== YAML Config Results ===")
for metric, score in results.aggregate_scores.items():
    print(f"{metric}: {score:.3f}")
```

---

## Tests

### File: `tests/conftest.py`

```python
"""Pytest fixtures."""

import pytest
from evalkit.api import Dataset, Sample


@pytest.fixture
def sample():
    return Sample(
        inputs={
            "question": "Test?",
            "contexts": ["Context 1"],
            "answer": "Answer"
        },
        ground_truth={"expected_answer": "Expected"}
    )


@pytest.fixture
def dataset(sample):
    return Dataset(samples=[sample])


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear registry before each test."""
    from evalkit.api.metrics.registry import MetricRegistry
    yield
    MetricRegistry.clear()
```

---

### File: `tests/test_evaluation.py`

```python
"""Test Evaluation class."""

import pytest
from evalkit.api import Evaluation, MetricRegistry


def test_evaluation_init(dataset):
    """Test initialization."""
    evaluation = Evaluation(
        dataset=dataset,
        metrics=["ragas:answer_relevancy"],
    )
    assert evaluation.dataset == dataset
    assert len(evaluation.metrics) == 1


def test_evaluation_run(dataset):
    """Test run()."""
    evaluation = Evaluation(
        dataset=dataset,
        metrics=["ragas:answer_relevancy"]
    )
    results = evaluation.run()
    
    assert len(results.sample_results) == len(dataset)
    assert "ragas:answer_relevancy" in results.aggregate_scores


def test_default_provider(dataset):
    """Test default provider."""
    evaluation = Evaluation(
        dataset=dataset,
        metrics=["answer_relevancy"],
        default_provider="ragas"
    )
    results = evaluation.run()
    assert "ragas:answer_relevancy" in results.aggregate_scores
```

---

### File: `tests/test_registry.py`

```python
"""Test MetricRegistry."""

import pytest
from evalkit.api.metrics.registry import MetricRegistry
from evalkit.api.metrics.base import BaseMetric, MetricResult


def test_register():
    """Test registration."""
    @MetricRegistry.register(provider="test", metric_id="test_metric")
    class TestMetric(BaseMetric):
        def evaluate(self, sample):
            return MetricResult(
                metric_name="test",
                score=0.5,
                passed=True,
                provider="test"
            )
    
    metric = MetricRegistry.get("test", "test_metric")
    assert isinstance(metric, TestMetric)


def test_resolve_best_single():
    """Test resolve with one provider."""
    @MetricRegistry.register(provider="only", metric_id="unique")
    class UniqueMetric(BaseMetric):
        def evaluate(self, sample):
            pass
    
    provider = MetricRegistry.resolve_best("unique")
    assert provider == "only"


def test_resolve_best_ambiguous():
    """Test resolve with multiple providers."""
    @MetricRegistry.register(provider="p1", metric_id="common")
    class M1(BaseMetric):
        def evaluate(self, sample):
            pass
    
    @MetricRegistry.register(provider="p2", metric_id="common")
    class M2(BaseMetric):
        def evaluate(self, sample):
            pass
    
    with pytest.raises(ValueError):
        MetricRegistry.resolve_best("common")
    
    # With default provider
    provider = MetricRegistry.resolve_best("common", default_provider="p1")
    assert provider == "p1"


def test_list_metrics():
    """Test listing."""
    @MetricRegistry.register(provider="test", metric_id="m1")
    class M1(BaseMetric):
        def evaluate(self, sample):
            pass
    
    metrics = MetricRegistry.list_metrics(provider="test")
    assert "test" in metrics
    assert "m1" in metrics["test"]
```

---

### File: `tests/test_providers/test_ragas.py`

```python
"""Test RAGAS provider."""

import pytest
from evalkit.api import Sample, MetricRegistry


def test_ragas_answer_relevancy(sample):
    """Test RAGAS answer relevancy."""
    metric = MetricRegistry.get("ragas", "answer_relevancy")
    result = metric.evaluate(sample)
    
    assert result.metric_name == "answer_relevancy"
    assert result.provider == "ragas"
    assert 0.0 <= result.score <= 1.0


def test_ragas_faithfulness(sample):
    """Test RAGAS faithfulness."""
    metric = MetricRegistry.get("ragas", "faithfulness")
    result = metric.evaluate(sample)
    
    assert result.metric_name == "faithfulness"
    assert result.provider == "ragas"
    assert 0.0 <= result.score <= 1.0
```

---

### File: `tests/test_providers/test_deepeval.py`

```python
"""Test DeepEval provider."""

import pytest
from evalkit.api import Sample, MetricRegistry


def test_deepeval_answer_relevancy(sample):
    """Test DeepEval answer relevancy."""
    metric = MetricRegistry.get("deepeval", "answer_relevancy")
    result = metric.evaluate(sample)
    
    assert result.metric_name == "answer_relevancy"
    assert result.provider == "deepeval"
    assert 0.0 <= result.score <= 1.0


def test_deepeval_faithfulness(sample):
    """Test DeepEval faithfulness."""
    metric = MetricRegistry.get("deepeval", "faithfulness")
    result = metric.evaluate(sample)
    
    assert result.metric_name == "faithfulness"
    assert result.provider == "deepeval"
    assert 0.0 <= result.score <= 1.0
```

---

## Implementation Checklist

### Day 1 (4-6 hours)

- [ ] **Setup** (30 min)
  - [ ] Create folder structure
  - [ ] `requirements.txt`, `pyproject.toml`
  - [ ] `.env.example`

- [ ] **Core** (2 hours)
  - [ ] `config/schemas.py`
  - [ ] `api/dataset.py`
  - [ ] `api/metrics/base.py`
  - [ ] `api/metrics/registry.py`

- [ ] **RAGAS** (1.5 hours)
  - [ ] `metric_providers/ragas/adapter.py`
  - [ ] `metric_providers/ragas/metrics.py`
  - [ ] `metric_providers/ragas/__init__.py`

- [ ] **DeepEval** (1.5 hours)
  - [ ] `metric_providers/deepeval/adapter.py`
  - [ ] `metric_providers/deepeval/metrics.py`
  - [ ] `metric_providers/deepeval/__init__.py`

### Day 2 (4-6 hours)

- [ ] **Evaluation** (2 hours)
  - [ ] `api/evaluation.py`
  - [ ] `api/__init__.py`
  - [ ] `metric_providers/__init__.py`

- [ ] **Examples** (1.5 hours)
  - [ ] `examples/data/dataset.json`
  - [ ] `examples/data/config.yaml`
  - [ ] `examples/01_basic.py`
  - [ ] `examples/02_config_driven.py`
  - [ ] `examples/03_comparison.py`
  - [ ] `examples/04_yaml.py`

- [ ] **Tests** (1.5 hours)
  - [ ] `tests/conftest.py`
  - [ ] `tests/test_evaluation.py`
  - [ ] `tests/test_registry.py`
  - [ ] `tests/test_providers/test_ragas.py`
  - [ ] `tests/test_providers/test_deepeval.py`

- [ ] **Verify** (1 hour)
  - [ ] Run all examples
  - [ ] Run all tests
  - [ ] Create README.md

---

## Demo Script

### Setup

```bash
# Install
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY=your_key_here
```

### Demo 1: Basic Usage

```bash
python examples/01_basic.py
```

**Show:** Explicit metric instantiation via registry

### Demo 2: Config-Driven

```bash
python examples/02_config_driven.py
```

**Show:** String-based metrics with default provider

### Demo 3: Backend Comparison (KEY DEMO)

```bash
python examples/03_comparison.py
```

**Show:** Same metrics, different backends, compare scores

### Demo 4: YAML Config

```bash
python examples/04_yaml.py
```

**Show:** Complete config-driven workflow

### Demo 5: Registry

```bash
python -c "
from evalkit.api import MetricRegistry
print('Providers:', MetricRegistry.list_providers())
print('Metrics:', MetricRegistry.list_metrics())
"
```

**Show:** Metric discovery

### Demo 6: Tests

```bash
pytest tests/ -v
```

**Show:** All tests pass

---

## README.md

```markdown
# EvalKit - Multi-Backend Evaluation Framework

Config-driven evaluation supporting RAGAS, DeepEval, and custom metrics.

## Features

✅ Multiple backends (RAGAS, DeepEval)  
✅ Flexible metric specification (instances, strings, dicts, YAML)  
✅ Smart provider resolution  
✅ CI/CD ready  

## Quick Start

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_key
python examples/01_basic.py
```

## Usage

### Basic

```python
from evalkit.api import Evaluation, Dataset, MetricRegistry

dataset = Dataset.from_json("data.json")
evaluation = Evaluation(
    dataset=dataset,
    metrics=[MetricRegistry.get("ragas", "answer_relevancy")]
)
results = evaluation.run()
```

### Config-Driven

```python
evaluation = Evaluation(
    dataset=dataset,
    metrics=["answer_relevancy", "faithfulness"],
    default_provider="ragas"
)
```

### Backend Comparison

```python
# RAGAS
eval_ragas = Evaluation(
    dataset=dataset,
    metrics=["ragas:answer_relevancy"]
)

# DeepEval
eval_deepeval = Evaluation(
    dataset=dataset,
    metrics=["deepeval:answer_relevancy"]
)
```

## Testing

```bash
pytest tests/ -v
```

## License

MIT
```

---

## Key Points for Architect

### 1. Flexible Metric Specification ✅
- Instances: `MetricRegistry.get("ragas", "answer_relevancy")`
- Strings: `"answer_relevancy"` or `"ragas:answer_relevancy"`
- Dicts: `{"id": "answer_relevancy", "provider": "ragas"}`

### 2. Smart Resolution ✅
- Unambiguous: Auto-select if one provider has metric
- Ambiguous: Use `default_provider` or explicit `provider:metric_id`
- Clear errors with helpful messages

### 3. Backend Swapping ✅
- Same metric ID works with multiple backends
- Results tagged with provider
- Easy comparison

### 4. Config-Driven ✅
- YAML configs work
- CI/CD ready
- No code changes needed

### 5. Future-Proof ✅
- Two-phase execution: Just add storage layer
- Plugin system: Entry points ready
- No refactor needed

---

**END OF PRD - Ready for Implementation**

This PRD is complete and can be fed directly to any AI code editor.