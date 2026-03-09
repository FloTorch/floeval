# All Usage Examples

This page documents the main usage patterns supported by Floeval.

---

## CLI workflows

### Standard evaluation

Use `floeval evaluate` for full datasets and for partial datasets that Floeval should complete at runtime.

| Workflow | Command | Use when |
|----------|---------|----------|
| Full dataset | `floeval evaluate -c config.yaml -d full_dataset.json -o results.json` | You already have `llm_response` values |
| Partial dataset | `floeval evaluate -c config.yaml -d partial_dataset.json -o results.json` | You want Floeval to generate missing responses before scoring |
| Two-step generation | `floeval generate -c config.yaml -d partial_dataset.json -o complete.json` | You want to inspect or reuse generated outputs |

CLI evaluation auto-detects partial datasets by checking whether samples are missing `llm_response`.

### Agent evaluation

Use `--agent` when the dataset contains agent traces or when you want the CLI to run a FloTorch agent runner:

```bash
floeval evaluate -c agent_config.yaml -d agent_dataset.json --agent -o agent_results.json
```

For partial agent datasets in CLI mode, include `evaluation_config.agent_name` in the config.

---

## Standard config examples

### YAML config for full or partial datasets

```yaml
llm_config:
  base_url: "https://api.openai.com/v1"
  api_key: "your-api-key"
  chat_model: "gpt-4o-mini"
  chat_endpoint: "chat/completions"
  embedding_model: "text-embedding-3-small"
  embedding_endpoint: "embeddings"
  system_prompt: "You are a concise assistant."  # optional

evaluation_config:
  default_provider: "ragas"
  metrics:
    - "answer_relevancy"
    - "faithfulness"
  metric_params:
    answer_relevancy:
      threshold: 0.7

dataset_generation_config:
  generator_model: "gpt-4o-mini"   # required for partial datasets
  batch_size: 20                   # optional
  max_concurrency: 10              # optional
```

### Agent config for CLI Mode 4

```yaml
llm_config:
  base_url: "https://gateway.example/openai/v1"
  api_key: "your-gateway-key"
  chat_model: "gpt-4o-mini"

evaluation_config:
  agent_name: "support-agent"
  metrics:
    - "goal_achievement"
    - "ragas:tool_call_accuracy"
```

---

## Dataset examples

### Full dataset

```json
{
  "samples": [
    {
      "user_input": "What is RAG?",
      "llm_response": "RAG stands for Retrieval-Augmented Generation.",
      "contexts": ["RAG combines retrieval with generation."],
      "ground_truth": "Retrieval-Augmented Generation"
    }
  ]
}
```

### Partial dataset

```json
{
  "samples": [
    {
      "user_input": "What is RAG?",
      "contexts": ["RAG combines retrieval with generation."],
      "ground_truth": "Retrieval-Augmented Generation"
    }
  ]
}
```

### Partial dataset with prompt expansion

Each `prompt_id` causes Floeval to generate a separate output sample.

```json
{
  "samples": [
    {
      "user_input": "Summarize this customer ticket.",
      "prompt_ids": ["brief", "supportive"]
    }
  ]
}
```

Matching prompt file:

```yaml
prompts:
  brief:
    template: "Answer in one short paragraph."
  supportive:
    template: "Answer with a warm, reassuring tone."
```

Add the prompt file to `evaluation_config`:

```yaml
evaluation_config:
  metrics:
    - "answer_relevancy"
  prompts_file: "prompts.yaml"
```

If you run either `floeval evaluate` on a partial dataset or `floeval generate`, the output dataset will contain one generated sample per `prompt_id`, each with a resulting `prompt_id` field.

---

## Python evaluation examples

### Full dataset with `Evaluation`

```python
from floeval import Evaluation, DatasetLoader
from floeval.config.schemas.io.llm import OpenAIProviderConfig

llm_config = OpenAIProviderConfig(
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
    chat_model="gpt-4o-mini",
    embedding_model="text-embedding-3-small",
)

dataset = DatasetLoader.from_samples(
    [
        {
            "user_input": "What is RAG?",
            "llm_response": "RAG stands for Retrieval-Augmented Generation.",
            "contexts": ["RAG combines retrieval with generation."],
        }
    ],
    partial_dataset=False,
)

evaluation = Evaluation(
    dataset=dataset,
    llm_config=llm_config,
    default_provider="ragas",
    metrics=["answer_relevancy", "faithfulness"],
)

results = evaluation.run()
print(results.aggregate_scores)
```

### Partial dataset with `Evaluation`

```python
partial_dataset = DatasetLoader.from_samples(
    [
        {
            "user_input": "What is RAG?",
            "contexts": ["RAG combines retrieval with generation."],
        }
    ],
    partial_dataset=True,
)

evaluation = Evaluation(
    dataset=partial_dataset,
    llm_config=llm_config,
    default_provider="ragas",
    metrics=["answer_relevancy", "faithfulness"],
    dataset_generator_model="gpt-4o-mini",
)

results = evaluation.run()
```

### Mixed-provider metric routing

```python
evaluation = Evaluation(
    dataset=dataset,
    llm_config=llm_config,
    metrics=[
        "ragas:answer_relevancy",
        "deepeval:contextual_relevancy",
        {"id": "faithfulness", "provider": "deepeval", "params": {"threshold": 0.8}},
    ],
)
```

### Async execution

```python
results = await evaluation.arun()
```

---

## Agent evaluation examples

### Pre-captured trace dataset

```python
from floeval.api.agent_evaluation import AgentEvaluation
from floeval.config.schemas.io.agent_dataset import AgentDataset

agent_dataset = AgentDataset.from_file("agent_dataset.json")

evaluation = AgentEvaluation(
    dataset=agent_dataset,
    llm_config=llm_config,
    metrics=["goal_achievement", "response_coherence"],
)

results = evaluation.run()
print(results.summary)
```

### Partial agent dataset with a Python callable

```python
from floeval.api.agent_evaluation import AgentEvaluation
from floeval.config.schemas.io.agent_dataset import AgentDataset, PartialAgentSample
from floeval.utils.agent_trace import capture_trace, log_turn


@capture_trace
def support_agent(user_input: str) -> str:
    response = f"Handled request: {user_input}"
    log_turn(response)
    return response


agent_dataset = AgentDataset(
    samples=[
        PartialAgentSample(
            user_input="Reset my password",
            reference_outcome="Password reset instructions were provided.",
        )
    ]
)

evaluation = AgentEvaluation(
    dataset=agent_dataset,
    agent=support_agent,
    llm_config=llm_config,
    metrics=["goal_achievement"],
)

results = evaluation.run()
```

### Partial agent dataset with FloTorch

```python
from floeval.api.agent_evaluation import AgentEvaluation
from floeval.config.schemas.io.agent_dataset import AgentDataset
from floeval.flotorch import create_flotorch_runner

agent_dataset = AgentDataset.from_file("partial_agent_dataset.json")
runner = create_flotorch_runner("support-agent", llm_config=llm_config)

evaluation = AgentEvaluation(
    dataset=agent_dataset,
    agent_runner=runner,
    llm_config=llm_config,
    metrics=["goal_achievement", "ragas:tool_call_accuracy"],
)

results = evaluation.run()
```

---

## Save and inspect results

### Save Python results

```python
import json

with open("results.json", "w", encoding="utf-8") as fh:
    json.dump(results.model_dump(), fh, indent=2)
```

### Inspect generated datasets portably

```bash
python -m json.tool complete.json
```

---

## Related references

- [Minimal Examples](copy-run.md) for shorter examples
- [Agent Evaluation](agent-evaluation.md) for dataset shapes and CLI details
- [Agent Tracing](agent-tracing.md) for trace capture helpers
- [Metrics](metrics.md) for the current metric catalog
