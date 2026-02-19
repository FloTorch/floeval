# All Usage Examples

This page shows all the ways you can use Floeval. Pick what works for you!

---

## Flow 1: Using CLI

Run evaluations from the command line. No Python code needed.

### Three CLI workflows

| Workflow | Command | Best for |
|----------|---------|----------|
| **Evaluate full dataset** | `floeval evaluate -c config -d full_dataset.json -o results.json` | You already have LLM responses ready |
| **Evaluate + generate (one step)** | `floeval evaluate -c config -d partial_dataset.json -o results.json` | Quick one-off evaluation; don't need saved responses |
| **Generate then evaluate (two steps)** | `floeval generate -c config -d partial_dataset.json -o complete.json`<br/>`floeval evaluate -c config -d complete.json -o results.json` | Audit responses, reuse dataset, parallel workflows |

### CLI commands overview

#### evaluate - Run metrics on dataset

```bash
floeval evaluate -c CONFIG_FILE -d DATASET_FILE [-o OUTPUT_FILE]
```

**Use cases**:

- Full dataset: You have pre-generated LLM responses
- Partial dataset: Floeval generates responses and evaluates in one step

**Auto-detection**: CLI detects if dataset is full (has `llm_response`) or partial (doesn't have it) and behaves accordingly.

| Option | Required | Description |
|--------|----------|-------------|
| `-c, --config` | Yes | Path to config file (YAML or JSON) |
| `-d, --dataset` | Yes | Path to dataset file (JSON or JSONL). Can be full or partial (auto-detected). |
| `-o, --output` | No | Save results to JSON file. If omitted, prints to terminal. |

#### generate - Create responses for partial dataset

```bash
floeval generate -c CONFIG_FILE -d PARTIAL_DATASET [-o OUTPUT_FILE]
```

**Use cases**:

- Generate responses separately from evaluation
- Audit/review generated responses before evaluation
- Build reusable evaluation datasets
- Parallelize generation and evaluation

| Option | Required | Description |
|--------|----------|-------------|
| `-c, --config` | Yes | Path to config file (YAML or JSON) - must include `dataset_generation_config` |
| `-d, --dataset` | Yes | Path to partial dataset file (JSON or JSONL) - should NOT have `llm_response` |
| `-o, --output` | Yes | Path to save complete dataset with generated responses |

---

### Full vs partial dataset

| Type | Required fields | Optional fields | Use case |
|------|-----------------|-----------------|----------|
| **Full** | `user_input`, `llm_response` | `contexts`, `ground_truth` | You already have LLM outputs to evaluate |
| **Partial** | `user_input` only | `contexts`, `ground_truth`, `system_prompt` | You have questions (and optionally context); Floeval generates responses at runtime |

**How partial works at runtime**: For each sample, Floeval calls your LLM with `user_input` (and optional `system_prompt`), obtains `llm_response`, then runs metrics on it. You never need to pre-generate responses. Requires `dataset_generation_config.generator_model` in config; CLI auto-detects partial datasets when samples lack `llm_response`.

### Config file (JSON)

```json
{
  "llm_config": {
    "base_url": "https://api.openai.com/v1",
    "api_key": "your-api-key",
    "chat_model": "gpt-4o-mini",
    "embedding_model": "text-embedding-3-small",
    "system_prompt": "You are a helpful assistant. Answer concisely."
  },
  "evaluation_config": {
    "metrics": ["ragas:answer_relevancy", "ragas:faithfulness"],
    "default_provider": "ragas",
    "metric_params": {
      "answer_relevancy": { "threshold": 0.6 }
    }
  },
  "dataset_generation_config": {
    "generator_model": "gpt-4o-mini"
  }
}
```

`dataset_generation_config` is **required only for partial datasets** (samples without `llm_response`). `system_prompt` is optional; used when generating responses for partial datasets.

### Config file (YAML)

```yaml
llm_config:
  base_url: "https://api.openai.com/v1"
  api_key: "your-api-key"
  chat_model: gpt-4o-mini
  embedding_model: text-embedding-3-small
  system_prompt: "You are a helpful assistant. Answer concisely."

evaluation_config:
  metrics:
    - ragas:answer_relevancy
    - ragas:faithfulness
  default_provider: ragas
  metric_params:
    answer_relevancy:
      threshold: 0.6

dataset_generation_config:
  generator_model: gpt-4o-mini
```

### Dataset file (JSON) — Full dataset

Samples already have `llm_response`. Use when you have pre-generated outputs:

```json
{
  "samples": [
    {
      "user_input": "What is Python?",
      "llm_response": "Python is a programming language.",
      "contexts": ["Python is widely used."],
      "ground_truth": "A programming language"
    },
    {
      "user_input": "What is RAG?",
      "llm_response": "RAG stands for Retrieval-Augmented Generation.",
      "contexts": ["RAG combines retrieval with generation."],
      "ground_truth": "Retrieval-Augmented Generation"
    }
  ]
}
```

### Dataset file (JSON) — Partial dataset

Only `user_input` required; `contexts` and `ground_truth` are optional. Omit `llm_response`; CLI generates it at runtime:

```json
{
  "samples": [
    {
      "user_input": "What is Python?",
      "contexts": ["Python is widely used."]
    },
    {
      "user_input": "What is RAG?",
      "contexts": ["RAG combines retrieval with generation."],
      "ground_truth": "Retrieval-Augmented Generation"
    }
  ]
}
```

### Dataset file (JSONL)

One sample per line. Full: include `llm_response`. Partial: omit it (CLI auto-detects):

```jsonl
{"user_input": "Q1?", "llm_response": "A1.", "contexts": ["..."]}
{"user_input": "Q2?", "llm_response": "A2.", "contexts": ["..."]}
```

Partial JSONL example (no llm_response):

```jsonl
{"user_input": "What is Python?", "contexts": ["Python is widely used."]}
{"user_input": "What is RAG?", "contexts": ["RAG combines retrieval with generation."]}
```

!!! note "contexts required for faithfulness"
    The `faithfulness` metric requires `contexts` in each sample.

### CLI command — Generate responses separately

When working with partial datasets, you can generate responses once and reuse them:

```bash
floeval generate -c CONFIG -d PARTIAL_DATASET -o OUTPUT
```

| Option | Required | Description |
|--------|----------|-------------|
| `-c, --config` | Yes | Path to config file (YAML or JSON) |
| `-d, --dataset` | Yes | Path to partial dataset file (JSON or JSONL) - should not contain `llm_response` |
| `-o, --output` | Yes | Path to save complete dataset with generated responses |

#### When to use generate separately

**Use `floeval generate` when**:

- You want to generate responses once and evaluate them multiple times
- You need to audit generated responses before evaluation
- You want to parallelize generation and evaluation
- You're building a reusable evaluation dataset

**Use `floeval evaluate` on partial dataset when**:

- Quick one-off evaluation is your goal
- You don't need to save generated responses
- Simplicity is more important than reusability

#### Example workflow: Generate then evaluate

```bash
# Step 1: Generate responses from your partial dataset
floeval generate -c config.yaml -d questions.json -o complete_dataset.json

# Step 2: Inspect the generated dataset
cat complete_dataset.json | head

# Step 3: Evaluate the complete dataset
floeval evaluate -c config.yaml -d complete_dataset.json -o results.json
```

Output from step 1: `complete_dataset.json` now has full samples with `llm_response` populated.

#### Config requirements for generate

Your config file must have both `llm_config` and `dataset_generation_config`:

```yaml
llm_config:
  base_url: "https://api.openai.com/v1"
  api_key: "your-api-key"
  chat_model: "gpt-4o-mini"
  embedding_model: "text-embedding-3-small"
  system_prompt: "You are a helpful assistant."  # Optional

dataset_generation_config:
  generator_model: "gpt-4o-mini"  # Required for generate command
```

---

## Flow 2: Using Python

Integrate Floeval into your application. Full control over evaluation.

### LLM config (Python)

Build config from a dict and pass to `Evaluation`:

```python
from floeval.config.schemas.io.llm import OpenAIProviderConfig

LLM_CONFIG = {
    "base_url": "https://api.openai.com/v1",
    "api_key": "your-api-key",
    "chat_model": "gpt-4o-mini",
    "embedding_model": "text-embedding-3-small",
    "system_prompt": "You are a helpful assistant."  # optional
}

llm_config = OpenAIProviderConfig(**LLM_CONFIG)
```

Load the dict from env, a file, or secrets—your choice.

### Imports

```python
from floeval import Evaluation, DatasetLoader
from floeval.config.schemas.io.llm import OpenAIProviderConfig
```

### Ways to load dataset

| Method | Use when |
|--------|----------|
| `DatasetLoader.from_file("file.json")` | Load from file (auto-detect JSON/JSONL) |
| `DatasetLoader.from_json("file.json")` | Load from JSON file |
| `DatasetLoader.from_samples([{...}, {...}], partial_dataset=False)` | Build from Python list |
| `DatasetLoader.from_dict({"samples": [...]}, partial_dataset=False)` | Build from dict |

### Full vs partial dataset (Python)

Same as CLI: **Full** = samples have `llm_response`. **Partial** = only `user_input`; Floeval generates `llm_response` at runtime by calling your LLM. Use `partial_dataset=True` and pass `dataset_generator_model`.

### Basic Python example — Full dataset

```python
from floeval import Evaluation, DatasetLoader
from floeval.config.schemas.io.llm import OpenAIProviderConfig

LLM_CONFIG = {
    "base_url": "https://api.openai.com/v1",
    "api_key": "your-api-key",
    "chat_model": "gpt-4o-mini",
    "embedding_model": "text-embedding-3-small",
}
llm_config = OpenAIProviderConfig(**LLM_CONFIG)

dataset = DatasetLoader.from_samples([
    {"user_input": "What is Python?", "llm_response": "Python is a programming language."},
    {"user_input": "What is RAG?", "llm_response": "RAG stands for Retrieval-Augmented Generation."}
], partial_dataset=False)

evaluation = Evaluation(
    dataset=dataset,
    llm_config=llm_config,
    metrics=["answer_relevancy"],
    default_provider="ragas"
)

results = evaluation.run()
print(results.aggregate_scores)
print(results.sample_results)
print(results.summary)
```

### Basic Python example — Partial dataset

Only `user_input` required; `contexts` and `ground_truth` optional. Floeval generates `llm_response` internally, then runs metrics:

```python
partial_ds = DatasetLoader.from_samples([
    {"user_input": "What is Python?", "contexts": ["Python is widely used."]},
    {"user_input": "What is RAG?", "contexts": ["RAG combines retrieval with generation."]}
], partial_dataset=True)

evaluation = Evaluation(
    dataset=partial_ds,
    llm_config=llm_config,
    metrics=["answer_relevancy", "faithfulness"],
    default_provider="ragas",
    dataset_generator_model="gpt-4o-mini",  # required for partial
)
results = evaluation.run()
# Internally: for each sample, Floeval sends user_input to LLM,
# gets llm_response, then runs metrics on it
print(results.aggregate_scores)
```

### Ways to specify metrics

| Format | Example |
|--------|---------|
| String (uses default_provider) | `["answer_relevancy", "faithfulness"]` |
| Provider:metric | `["ragas:answer_relevancy", "deepeval:faithfulness"]` |
| Dict with params | `[{"id": "answer_relevancy", "provider": "ragas", "params": {"threshold": 0.8}}]` |
| Metric instance | `[my_custom_metric, "answer_relevancy"]` |

### Async evaluation

```python
results = await evaluation.arun()
```

---

## For RAG systems

Include `contexts` in your dataset to use `faithfulness`:

```python
dataset = DatasetLoader.from_samples([
    {
        "user_input": "What is RAG?",
        "llm_response": "RAG stands for Retrieval-Augmented Generation.",
        "contexts": ["RAG combines retrieval with generation."]
    }
], partial_dataset=False)

evaluation = Evaluation(
    dataset=dataset,
    llm_config=llm_config,
    metrics=["answer_relevancy", "faithfulness"],
    default_provider="ragas"
)
```

---

## Save results

```python
import json

results = evaluation.run()

with open("results.json", "w") as f:
    json.dump(results.model_dump(), f, indent=2)
```

---

## Process results

```python
for i, sample in enumerate(results.sample_results, 1):
    print(f"\nSample {i}: {sample['user_input']}")
    for metric, data in sample['metrics'].items():
        score = data['score']
        passed = data.get('passed', True)
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} {metric}: {score:.2f}")
```

---

## Next steps

- **[Copy & Run](copy-run.md)** — Copy-paste examples
- **[Metrics](metrics.md)** — Built-in metrics reference
- **[Custom Metrics](custom-metrics.md)** — Create your own metrics
- **[API Reference](api-reference.md)** — Full technical details
