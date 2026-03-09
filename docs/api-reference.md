# API Reference

Reference for Floeval's user-facing config, CLI, and Python API surface.

---

## Public Python entry points

Top-level imports from `floeval`:

```python
from floeval import Evaluation, Dataset, DatasetLoader, MetricRegistry
```

Additional public imports commonly used:

```python
from floeval.api.agent_evaluation import AgentEvaluation
from floeval.api.metrics.custom import custom_metric, criteria
from floeval.config.schemas.io.llm import OpenAIProviderConfig
from floeval.utils.agent_trace import capture_trace, log_turn, log_tool_result, wrap_langchain_agent
```

---

## `llm_config`

Use `OpenAIProviderConfig` for OpenAI-compatible providers.

```python
from floeval.config.schemas.io.llm import OpenAIProviderConfig

llm_config = OpenAIProviderConfig(
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
    chat_model="gpt-4o-mini",
    chat_endpoint="chat/completions",
    embedding_model="text-embedding-3-small",
    embedding_endpoint="embeddings",
    system_prompt="You are a helpful assistant.",  # optional
    extra_kwargs=None,                             # optional
)
```

### Fields

| Field | Required | Notes |
|-------|----------|-------|
| `provider_type` | No | Defaults to `"openai"` |
| `base_url` | Yes | Base API URL |
| `api_key` | Yes | Provider credential |
| `chat_model` | Yes | Model used for chat/completions |
| `chat_endpoint` | No | Defaults to `chat/completions` |
| `embedding_model` | No | Optional in the schema, but commonly needed for provider-backed evaluation metrics |
| `embedding_endpoint` | No | Optional embedding endpoint |
| `system_prompt` | No | Used during response generation when provided |
| `extra_kwargs` | No | Provider-specific keyword args |

---

## Config file structure

Floeval's CLI reads YAML, YML, or JSON config files.

### `evaluation_config`

| Field | Required | Notes |
|-------|----------|-------|
| `metrics` | Yes | List of metric specs |
| `default_provider` | No | Used when metric IDs are ambiguous |
| `metric_params` | No | Mapping of metric name or `provider:metric` to params |
| `prompts_file` | No | Prompt file used for prompt-aware partial generation |
| `agent_name` | No | Required for CLI partial agent evaluation |
| `dataset_generator_model` | No | Fallback location for partial-dataset generation model |

### `dataset_generation_config`

| Field | Required | Notes |
|-------|----------|-------|
| `generator_model` | Yes for generation flows | Model used to populate missing `llm_response` values |
| `batch_size` | No | Defaults are handled in the generation layer |
| `max_concurrency` | No | Controls async generation fan-out |

### Example config

```yaml
llm_config:
  base_url: "https://api.openai.com/v1"
  api_key: "your-api-key"
  chat_model: "gpt-4o-mini"
  chat_endpoint: "chat/completions"
  embedding_model: "text-embedding-3-small"
  embedding_endpoint: "embeddings"
  system_prompt: "You are a concise assistant."

evaluation_config:
  default_provider: "ragas"
  metrics:
    - "answer_relevancy"
    - "faithfulness"
  metric_params:
    answer_relevancy:
      threshold: 0.7
  prompts_file: "prompts.yaml"

dataset_generation_config:
  generator_model: "gpt-4o-mini"
  batch_size: 20
  max_concurrency: 10
```

---

## CLI reference

### `floeval --version`

```bash
floeval --version
```

### `floeval evaluate`

```bash
floeval evaluate -c CONFIG -d DATASET [-o OUTPUT] [--agent]
```

| Option | Required | Notes |
|--------|----------|-------|
| `-c, --config` | Yes | YAML, YML, or JSON config file |
| `-d, --dataset` | Yes | `.json` or `.jsonl` dataset |
| `-o, --output` | No | Saves results to JSON |
| `--agent` | No | Switches into agent-evaluation mode |

Behavior:

- standard mode auto-detects partial datasets by checking whether samples are missing `llm_response`
- agent mode loads agent datasets and uses `AgentEvaluation`
- partial agent datasets in CLI mode require `evaluation_config.agent_name`

### `floeval generate`

```bash
floeval generate -c CONFIG -d DATASET -o OUTPUT
```

| Option | Required | Notes |
|--------|----------|-------|
| `-c, --config` | Yes | Must include `llm_config` and `dataset_generation_config` |
| `-d, --dataset` | Yes | Partial standard dataset in `.json` or `.jsonl` |
| `-o, --output` | Yes | Output must use `.json` or `.jsonl` |

Behavior:

- fills missing `llm_response` values
- supports prompt-driven expansion when samples contain `prompt_ids`
- chooses export format from the output file extension

---

## `Evaluation`

Use `Evaluation` for standard LLM and RAG evaluation.

### Constructor

| Parameter | Required | Notes |
|-----------|----------|-------|
| `dataset` | Yes | `Dataset` or `PartialDataset` |
| `metrics` | Yes | Metric specs in string, dict, or instance form |
| `default_provider` | No | Helps resolve ambiguous metric names |
| `llm_config` | No | Needed for provider-backed metrics and partial generation |
| `metric_params` | No | Shared params keyed by metric name |
| `dataset_generator_model` | No | Required when `dataset` is partial |
| `prompts_file` | No | Prompt file path used for prompt-aware generation |

### Example

```python
from floeval import Evaluation, DatasetLoader

dataset = DatasetLoader.from_file("dataset.json", partial_dataset=False)

evaluation = Evaluation(
    dataset=dataset,
    llm_config=llm_config,
    default_provider="ragas",
    metrics=["answer_relevancy", "faithfulness"],
    metric_params={"answer_relevancy": {"threshold": 0.8}},
)

results = evaluation.run()
```

### Async

```python
results = await evaluation.arun()
```

---

## `AgentEvaluation`

Use `AgentEvaluation` for agent traces and partial agent datasets.

### Constructor

| Parameter | Required | Notes |
|-----------|----------|-------|
| `dataset` | Yes | `AgentDataset` |
| `metrics` | Yes | Builtin and provider-qualified agent metrics |
| `llm_config` | No | Needed for LLM-judged agent metrics |
| `agent` | No | Python callable for Mode 2 |
| `agent_runner` | No | Runner object for Mode 4 |
| `default_provider` | No | Defaults to `"builtin"` |
| `metric_params` | No | Shared params keyed by metric name |

### Example

```python
from floeval.api.agent_evaluation import AgentEvaluation
from floeval.config.schemas.io.agent_dataset import AgentDataset

dataset = AgentDataset.from_file("agent_dataset.json")

evaluation = AgentEvaluation(
    dataset=dataset,
    llm_config=llm_config,
    metrics=["goal_achievement", "ragas:tool_call_accuracy"],
)

results = evaluation.run()
```

### Async

```python
results = await evaluation.arun()
```

---

## `DatasetLoader`

Load standard evaluation datasets from files, lists, or dicts.

```python
from floeval import DatasetLoader

full_dataset = DatasetLoader.from_file("dataset.json", partial_dataset=False)
partial_dataset = DatasetLoader.from_file("partial.json", partial_dataset=True)

dataset = DatasetLoader.from_samples(
    [{"user_input": "Q?", "llm_response": "A."}],
    partial_dataset=False,
)
```

### Common methods

| Method | Returns | Notes |
|--------|---------|-------|
| `from_file(path, partial_dataset=...)` | `Dataset` or `PartialDataset` | Supports `.json` and `.jsonl` |
| `from_json(path, partial_dataset=...)` | `Dataset` or `PartialDataset` | JSON only |
| `from_samples(samples, partial_dataset=...)` | `Dataset` or `PartialDataset` | Build from Python objects |
| `from_dict(data, partial_dataset=...)` | `Dataset` or `PartialDataset` | Build from a dict with `samples` |

---

## Dataset models

### Standard datasets

#### `Sample`

| Field | Notes |
|-------|-------|
| `user_input` | Required |
| `llm_response` | Required for full datasets |
| `contexts` | Optional, used by grounding and retrieval metrics |
| `ground_truth` | Optional, used by some recall and precision metrics |
| `metadata` | Optional |
| `prompt_id` | Present after prompt-driven generation |

#### `PartialSample`

Same as `Sample`, but `llm_response` can be empty or omitted. Partial samples also support:

| Field | Notes |
|-------|-------|
| `prompt_ids` | Optional list of prompt IDs that expands one sample into multiple generated outputs |

### Agent datasets

| Model | Notes |
|-------|-------|
| `AgentDataset` | Collection of full or partial agent samples |
| `AgentSample` | Full sample with `trace` |
| `PartialAgentSample` | Sample without `trace` yet |
| `AgentTrace` | Trace with `messages`, `final_response`, and optional `metadata` |

Agent trace message roles in saved datasets must be:

- `human`
- `ai`
- `tool`

---

## Results objects

### `EvaluationResult`

```python
results = evaluation.run()

results.sample_results
results.aggregate_scores
results.summary
```

`summary` includes:

- `total_samples`
- `providers_used`
- `pass_rates`
- `aggregate_scores`

### `AgentEvaluationResult`

```python
agent_results = agent_evaluation.run()

agent_results.sample_results
agent_results.summary
```

Agent summaries currently contain per-metric averages for successful metric runs.

---

## `MetricRegistry`

Use `MetricRegistry` to inspect what is currently registered.

```python
from floeval.api.metrics.registry import MetricRegistry

print(MetricRegistry.list_providers())
print(MetricRegistry.list_metrics("ragas"))
print(MetricRegistry.list_metrics("deepeval"))
print(MetricRegistry.list_metrics("builtin"))
print(MetricRegistry.list_all_metrics())
```

### Common providers

| Provider | Notes |
|----------|-------|
| `ragas` | Standard and agent metrics |
| `deepeval` | Standard LLM and retrieval metrics |
| `builtin` | Agent judge metrics |
| `custom` | Appears after custom metrics are defined |

---

## Related references

- [Examples](examples.md)
- [Agent Evaluation](agent-evaluation.md)
- [Metrics](metrics.md)
- [Troubleshooting](troubleshooting.md)
