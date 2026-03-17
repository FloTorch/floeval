# Prompt Evaluation

Use this guide to evaluate one or more system prompts against the same dataset. Prompt evaluation lets you compare prompt variants at scale before shipping.

---

## What prompt evaluation is

Prompt evaluation is a specialized form of LLM evaluation where you:

1. Write a partial dataset with questions (no pre-generated responses)
2. Define one or more system prompts
3. Have Floeval generate responses for each prompt variant
4. Score all responses using the same metrics

This answers the question: *which prompt works best for my use case?*

---

## Sub-modes

| Sub-mode | Description | When to use |
|----------|-------------|-------------|
| **Prompt (no RAG)** | Single or multi-prompt against questions only | No retrieval context needed |
| **Prompt (with RAG)** | Single or multi-prompt with a vector store or pre-fetched context | Retrieval is part of the pipeline |

The distinction matters for metrics: no-RAG prompts only support answer relevancy metrics, while RAG prompts support the full set of retrieval metrics.

---

## Inline single prompt (CLI)

The simplest form: a single `system_prompt` and `user_prompt` defined directly in the config. The dataset is a standard partial dataset with `user_input` values (or `question` as an alias).

### Config

```yaml
# config.yaml
llm_config:
  base_url: "https://api.openai.com/v1"
  api_key: "your-api-key"
  chat_model: "gpt-4o-mini"
  embedding_model: "text-embedding-3-small"

evaluation_config:
  default_provider: "ragas"
  metrics:
    - "answer_relevancy"
  prompts_file: null  # not needed for inline prompt

dataset_generation_config:
  generator_model: "gpt-4o-mini"
```

To use an inline system prompt without a prompts file, set `system_prompt` inside `llm_config`:

```yaml
llm_config:
  base_url: "https://api.openai.com/v1"
  api_key: "your-api-key"
  chat_model: "gpt-4o-mini"
  system_prompt: "You are a concise assistant. Keep answers under two sentences."

evaluation_config:
  metrics:
    - "answer_relevancy"

dataset_generation_config:
  generator_model: "gpt-4o-mini"
```

### Dataset

```json
{
  "samples": [
    {"user_input": "What is RAG?"},
    {"user_input": "How does vector search work?"}
  ]
}
```

You can also use the `question` alias instead of `user_input`:

```json
{
  "samples": [
    {"question": "What is RAG?"},
    {"question": "How does vector search work?"}
  ]
}
```

### Run

```bash
floeval evaluate -c config.yaml -d partial_dataset.json -o results.json
```

---

## Multi-prompt evaluation with a prompts file

A prompts file defines named prompt templates. Each sample in the dataset carries a list of `prompt_ids` that references the prompts to use. Floeval generates one response per `prompt_id` and scores each one separately.

### Prompts file (YAML)

```yaml
# prompts.yaml
prompts:
  concise:
    template: "Answer in one short paragraph."
  detailed:
    template: "Answer in detail with at least three points."
  formal:
    template: "Respond in formal business English."
```

JSON format is also accepted:

```json
{
  "prompts": {
    "concise": {"template": "Answer in one short paragraph."},
    "detailed": {"template": "Answer in detail with at least three points."},
    "formal": {"template": "Respond in formal business English."}
  }
}
```

### Dataset with `prompt_ids`

Each sample lists which prompts should generate responses for it:

```json
{
  "samples": [
    {
      "user_input": "Summarize this customer support ticket.",
      "prompt_ids": ["concise", "detailed", "formal"]
    },
    {
      "user_input": "Explain the refund policy.",
      "prompt_ids": ["concise", "detailed"]
    }
  ]
}
```

The above dataset with three prompt IDs generates six total samples (3 prompts × 2 questions = 6 responses), each tagged with a `prompt_id` in the output.

### Config

```yaml
# config.yaml
llm_config:
  base_url: "https://api.openai.com/v1"
  api_key: "your-api-key"
  chat_model: "gpt-4o-mini"
  embedding_model: "text-embedding-3-small"

evaluation_config:
  default_provider: "ragas"
  metrics:
    - "answer_relevancy"
  prompts_file: "prompts.yaml"

dataset_generation_config:
  generator_model: "gpt-4o-mini"
```

### Run

```bash
floeval evaluate -c config.yaml -d partial_dataset.json -o results.json
```

Or generate first and inspect before scoring:

```bash
floeval generate -c config.yaml -d partial_dataset.json -o generated.json
floeval evaluate -c config.yaml -d generated.json -o results.json
```

---

## Prompt with RAG (vectorstore)

When prompts run against a vectorstore, the config includes a `vectorstore_id` and Floeval fetches context for each sample before generating responses.

```yaml
llm_config:
  base_url: "https://gateway.example/openai/v1"
  api_key: "your-gateway-key"
  chat_model: "gpt-4o-mini"
  embedding_model: "text-embedding-3-small"

evaluation_config:
  default_provider: "ragas"
  metrics:
    - "answer_relevancy"
    - "faithfulness"
    - "context_precision"
  prompts_file: "prompts.yaml"

dataset_generation_config:
  generator_model: "gpt-4o-mini"
```

The dataset also specifies `vectorstore_id` at the data level:

```yaml
data:
  vectorstore_id: "my-vectorstore-id"
```

!!! note
    When using a vectorstore, Floeval automatically fetches context for each sample and injects it during generation. Use RAG metrics such as `faithfulness` and `context_precision` in this case.

---

## Prompt with pre-fetched context

If you have context in the dataset already (no vectorstore fetch needed), include it in each sample:

```json
{
  "samples": [
    {
      "user_input": "What is RAG?",
      "contexts": ["RAG combines retrieval with generation."],
      "prompt_ids": ["concise", "detailed"]
    }
  ]
}
```

Floeval skips the vectorstore fetch when `contexts` is present.

---

## Python API

### Single prompt (inline)

```python
from floeval import Evaluation, DatasetLoader
from floeval.config.schemas.io.llm import OpenAIProviderConfig

llm_config = OpenAIProviderConfig(
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
    chat_model="gpt-4o-mini",
    embedding_model="text-embedding-3-small",
    system_prompt="You are a concise assistant. Keep answers under two sentences.",
)

partial_dataset = DatasetLoader.from_samples(
    [
        {"user_input": "What is RAG?"},
        {"user_input": "How does vector search work?"},
    ],
    partial_dataset=True,
)

evaluation = Evaluation(
    dataset=partial_dataset,
    llm_config=llm_config,
    default_provider="ragas",
    metrics=["answer_relevancy"],
    dataset_generator_model="gpt-4o-mini",
)

results = evaluation.run()
print(results.aggregate_scores)
```

### Multi-prompt with prompts file

```python
from floeval import Evaluation, DatasetLoader
from floeval.config.schemas.io.llm import OpenAIProviderConfig

llm_config = OpenAIProviderConfig(
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
    chat_model="gpt-4o-mini",
    embedding_model="text-embedding-3-small",
)

partial_dataset = DatasetLoader.from_samples(
    [
        {
            "user_input": "Summarize this customer support ticket.",
            "prompt_ids": ["concise", "detailed"],
        }
    ],
    partial_dataset=True,
)

evaluation = Evaluation(
    dataset=partial_dataset,
    llm_config=llm_config,
    default_provider="ragas",
    metrics=["answer_relevancy"],
    dataset_generator_model="gpt-4o-mini",
    prompts_file="prompts.yaml",
)

results = evaluation.run()

# One result row per (sample, prompt_id) combination
for row in results.sample_results:
    print(row["prompt_id"], row["metrics"])
```

### Async execution

```python
results = await evaluation.arun()
```

---

## Metrics to use

### Prompt (no RAG)

| Provider | Metric |
|----------|--------|
| ragas | `answer_relevancy` |
| deepeval | `answer_relevancy` |

### Prompt (with RAG)

| Provider | Metric |
|----------|--------|
| ragas | `answer_relevancy`, `faithfulness`, `context_precision`, `context_recall`, `context_entity_recall`, `noise_sensitivity` |
| deepeval | `answer_relevancy`, `faithfulness`, `contextual_precision`, `contextual_recall`, `contextual_relevancy` |

---

## Understanding multi-prompt results

When `prompt_ids` is used, each row in `sample_results` carries a `prompt_id` field. You can group by `prompt_id` to compare which prompt performed best:

```python
from collections import defaultdict

by_prompt = defaultdict(list)
for row in results.sample_results:
    pid = row.get("prompt_id", "default")
    by_prompt[pid].append(row["metrics"].get("answer_relevancy", {}).get("score"))

for pid, scores in by_prompt.items():
    valid = [s for s in scores if s is not None]
    avg = sum(valid) / len(valid) if valid else None
    print(f"{pid}: avg answer_relevancy = {avg}")
```

---

## Common pitfalls

| Problem | Cause | Fix |
|---------|-------|-----|
| No responses generated | `dataset_generator_model` or `generator_model` is missing | Add `dataset_generation_config.generator_model` to config |
| All rows have the same `prompt_id` | Samples missing `prompt_ids` field | Add `prompt_ids` to each sample |
| Prompt template not applied | `prompts_file` path is wrong or prompt IDs do not match | Check file path and that all IDs in `prompt_ids` exist in the prompts file |
| RAG metrics fail | No context in samples and no vectorstore | Either add `contexts` to samples or configure a `vectorstore_id` |

---

## Related references

- [Examples](examples.md#dataset-examples) for prompt expansion dataset shapes
- [Metrics](metrics.md) for the full metric catalog
- [API Reference](api-reference.md) for `Evaluation` constructor and `prompts_file` field
- [Troubleshooting](troubleshooting.md#generate-command-issues) for generation issues
