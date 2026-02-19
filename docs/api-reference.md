# API Reference

Complete reference for config, CLI, and API.

---

## LLM config

### Python: OpenAIProviderConfig

Use `OpenAIProviderConfig` from a dict. Pass to `Evaluation` as `llm_config`:

```python
from floeval.config.schemas.io.llm import OpenAIProviderConfig

llm_config = OpenAIProviderConfig(
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
    chat_model="gpt-4o-mini",
    embedding_model="text-embedding-3-small",
    system_prompt="You are a helpful assistant."  # optional, for partial dataset generation
)
```

Or from a dict:

```python
LLM_CONFIG = {
    "base_url": "https://api.openai.com/v1",
    "api_key": "your-api-key",
    "chat_model": "gpt-4o-mini",
    "embedding_model": "text-embedding-3-small",
    "system_prompt": "You are a helpful assistant.",  # optional
}
llm_config = OpenAIProviderConfig(**LLM_CONFIG)
```

### Config file keys (CLI)

Under `llm_config`:

| Field | Required | Description |
|-------|----------|-------------|
| base_url | Yes | API endpoint (e.g. https://api.openai.com/v1) |
| api_key | Yes | Your API key |
| chat_model | Yes | LLM model name (e.g. gpt-4o-mini) |
| embedding_model | Yes | Embedding model (e.g. text-embedding-3-small) |
| system_prompt | No | Optional. Used when generating responses for partial datasets |

---

## Evaluation constructor

| Parameter | Description |
|-----------|-------------|
| dataset | Dataset or PartialDataset (required) |
| llm_config | Config for RAGAS/DeepEval/criteria metrics and partial dataset generation |
| metrics | List of metric specs (required) |
| default_provider | "ragas" or "deepeval" when using string metric names |
| metric_params | Dict of metric_name → params (e.g. `{"answer_relevancy": {"threshold": 0.8}}`) |
| dataset_generator_model | Required when using PartialDataset (samples without llm_response). Label for the provider. |

```python
from floeval import Evaluation, DatasetLoader
from floeval.config.schemas.io.llm import OpenAIProviderConfig

llm_config = OpenAIProviderConfig(**LLM_CONFIG)

evaluation = Evaluation(
    dataset=dataset,
    llm_config=llm_config,
    metrics=["answer_relevancy", "faithfulness"],
    default_provider="ragas",
    metric_params={"answer_relevancy": {"threshold": 0.8}},
    dataset_generator_model="gpt-4o-mini",  # only for partial datasets
)
```

---

## Results structure

```python
results = evaluation.run()

# Aggregate scores per metric
results.aggregate_scores   # {"ragas:answer_relevancy": 0.91, ...}

# Per-sample results
results.sample_results     # List of {user_input, llm_response, metrics: {...}}

# Summary
results.summary            # {total_samples, providers_used, pass_rates, aggregate_scores}
```

---

## MetricRegistry

Discover available metrics and their providers programmatically.

```python
from floeval.api.metrics.registry import MetricRegistry

# List all available providers
providers = MetricRegistry.list_providers()
# Returns: ['ragas', 'deepeval', 'custom']

# List metrics for a specific provider
ragas_metrics = MetricRegistry.list_metrics("ragas")
# Returns: ['answer_relevancy', 'faithfulness']

# Get all available metrics
all_metrics = MetricRegistry.list_all_metrics()
# Returns: ['answer_relevancy', 'faithfulness', 'custom_metric_name', ...]
```

### MetricRegistry methods

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `list_providers()` | None | list[str] | Get all registered metric providers |
| `list_metrics(provider: str)` | provider name | list[str] | Get metrics available for a provider |
| `list_all_metrics()` | None | list[str] | Get all registered metric names |

### Available providers

| Provider | Metrics | Description |
|----------|---------|-------------|
| `ragas` | answer_relevancy, faithfulness | RAGAS framework metrics for RAG evaluation |
| `deepeval` | answer_relevancy, faithfulness | DeepEval metrics (alternative provider) |
| `custom` | user-defined | Custom metrics defined with `@custom_metric` or `criteria()` |

---

## Usage flows summary

| Flow | Description |
|------|-------------|
| **Flow A: CLI (full dataset)** | Config YAML/JSON + dataset JSON/JSONL → `floeval evaluate -c config.yaml -d dataset.json` |
| **Flow B: CLI (partial dataset)** | Same, but dataset has no llm_response. CLI auto-detects and generates. Requires `dataset_generation_config` in config. |
| **Flow C: Python (from samples)** | DatasetLoader.from_samples([dicts]) + OpenAIProviderConfig + Evaluation |
| **Flow D: Python (from file)** | DatasetLoader.from_file() / from_json() + Evaluation |
| **Flow E: Custom metrics** | @custom_metric or criteria() + mix with built-in metrics |

---

## CLI commands

### evaluate

```bash
floeval evaluate -c CONFIG -d DATASET [-o OUTPUT]
```

| Option | Required | Description |
|--------|----------|-------------|
| `-c, --config` | Yes | Path to config file (YAML or JSON) |
| `-d, --dataset` | Yes | Path to dataset file (JSON or JSONL). Full or partial (auto-detected). |
| `-o, --output` | No | Save results to JSON file |

### generate

Generate `llm_response` for partial datasets and save to file. Useful when you want to generate responses separately from evaluation.

```bash
floeval generate -c CONFIG -d PARTIAL_DATASET -o OUTPUT
```

#### Parameters

| Option | Required | Description |
|--------|----------|-------------|
| `-c, --config` | Yes | Path to config file (YAML or JSON). Must include `llm_config` and `dataset_generation_config` |
| `-d, --dataset` | Yes | Path to partial dataset file (JSON or JSONL).  Samples should only have `user_input` and optional `contexts` |
| `-o, --output` | Yes | Path to save complete dataset with generated `llm_response` for each sample |

#### Use cases

| Use case | Flow |
|----------|------|
| **Generate once, evaluate multiple times** | `floeval generate ...` → `floeval evaluate ... (multiple times)` |
| **Audit generated responses** | `floeval generate ...` → inspect output → `floeval evaluate ...` |
| **Parallelize generation and evaluation** | Generate responses in batch, then evaluate separately |
| **Build reusable evaluation dataset** | Generate and save, then version control or share the dataset |

#### Example: Two-step workflow

**Step 1: Generate responses**

```bash
floeval generate -c config.yaml -d partial_questions.json -o generated_dataset.json
```

Input (`partial_questions.json`):

```json
{"samples": [{"user_input": "What is Python?"}, {"user_input": "What is RAG?"}]}
```

Output (`generated_dataset.json`):

```json
{"samples": [{"user_input": "What is Python?", "llm_response": "Python is..."}, {"user_input": "What is RAG?", "llm_response": "RAG is..."}]}
```

**Step 2: Evaluate the complete dataset**

```bash
floeval evaluate -c config.yaml -d generated_dataset.json -o results.json
```

#### Config requirements

Both `llm_config` and `dataset_generation_config` are required:

```json
{
  "llm_config": {
    "base_url": "https://api.openai.com/v1",
    "api_key": "your-api-key",
    "chat_model": "gpt-4o-mini",
    "embedding_model": "text-embedding-3-small"
  },
  "dataset_generation_config": {
    "generator_model": "gpt-4o-mini"
  }
}
```

### --version

```bash
floeval --version
```

---

## DatasetLoader

Load datasets from various sources for evaluation.

```python
from floeval import DatasetLoader

# From file (auto-detect JSON/JSONL)
dataset = DatasetLoader.from_file("dataset.json", partial_dataset=False)
partial_ds = DatasetLoader.from_file("partial.json", partial_dataset=True)

# From JSON file explicitly
dataset = DatasetLoader.from_json("dataset.json", partial_dataset=False)

# From Python list of dicts
dataset = DatasetLoader.from_samples([
    {"user_input": "Q?", "llm_response": "A."}
], partial_dataset=False)

# From dict with "samples" key
dataset = DatasetLoader.from_dict({"samples": [...]}, partial_dataset=False)
```

### DatasetLoader methods

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `from_file()` | `path: str`, `partial_dataset: bool` | Dataset \| PartialDataset | Auto-detect JSON/JSONL and load |
| `from_json()` | `path: str`, `partial_dataset: bool` | Dataset \| PartialDataset | Load from JSON file |
| `from_samples()` | `samples: list[dict]`, `partial_dataset: bool` | Dataset \| PartialDataset | Create from Python list |
| `from_dict()` | `data: dict`, `partial_dataset: bool` | Dataset \| PartialDataset | Create from dict with "samples" key |

### When to use full vs. partial datasets

| Scenario | Use | Dataset type |
|----------|-----|--------------|
| You have pre-generated LLM responses | Full dataset | `partial_dataset=False` |
| You only have questions; want Floeval to generate responses | Partial dataset | `partial_dataset=True` |
| Evaluating existing outputs | Full dataset | `partial_dataset=False` |
| Building evaluation dataset; generating responses on-the-fly | Partial dataset | `partial_dataset=True` |

**Note**: Partial datasets require `dataset_generation_config` in your config or `llm_config` in Python.

---

## Data models

### Sample (full dataset)

| Field | Type | Required |
|-------|------|----------|
| user_input | str | Yes |
| llm_response | str | Yes |
| contexts | list[str] | For faithfulness |
| ground_truth | str | Optional |
| metadata | dict | Optional |

### PartialSample (partial dataset)

Same as Sample, but `llm_response` can be omitted or empty. Used when generating responses.

---

## Config file schema (evaluate)

```json
{
  "llm_config": {
    "base_url": "...",
    "api_key": "...",
    "chat_model": "...",
    "embedding_model": "...",
    "system_prompt": "..."
  },
  "evaluation_config": {
    "metrics": ["ragas:answer_relevancy", "ragas:faithfulness"],
    "default_provider": "ragas",
    "metric_params": { "answer_relevancy": { "threshold": 0.6 } }
  },
  "dataset_generation_config": {
    "generator_model": "gpt-4o-mini"
  }
}
```

`dataset_generation_config` is required only when evaluating a partial dataset.

---

## Error handling

```python
try:
    results = evaluation.run()
except FileNotFoundError:
    print("Config or dataset file not found")
except ValueError:
    print("Invalid configuration or dataset")
except Exception as e:
    print(f"Evaluation failed: {e}")
```

---

## Next steps

- **[Copy & Run](copy-run.md)** — Copy-paste examples
- **[Examples](examples.md)** — Usage patterns
- **[Troubleshooting](troubleshooting.md)** — Common issues
