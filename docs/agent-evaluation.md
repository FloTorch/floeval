# Agent Evaluation

Use this guide for single-agent trace evaluation, tool-call checks, and outcome scoring.

For multi-agent DAG pipelines where multiple agents collaborate, see [Agentic Workflow](agentic-workflow.md).

---

## Scope

`AgentEvaluation` is intended for datasets that capture agent behavior rather than plain `llm_response` outputs.

### Single-agent vs agentic workflow

| Scenario | Use |
|----------|-----|
| One agent answers questions end-to-end | This page |
| Multiple agents work in sequence or parallel (DAG) | [Agentic Workflow](agentic-workflow.md) |

Floeval currently supports three implemented agent-evaluation modes for single-agent eval:

| Mode | Input | How traces are obtained | Typical entry point |
|------|-------|-------------------------|---------------------|
| Mode 1 | Full agent dataset | Already captured in the dataset | CLI or Python |
| Mode 2 | Partial agent dataset | Floeval runs your Python agent callable and collects traces | Python |
| Mode 4 | Partial agent dataset | Floeval runs a FloTorch runner | CLI or Python |

Mode 3 is not currently available.

---

## CLI workflow

Use the `--agent` flag to switch the CLI into agent-evaluation mode:

```bash
floeval evaluate -c config.yaml -d agent_dataset.json --agent -o agent_results.json
```

### Config file

For agent evaluation, `evaluation_config.metrics` is required. When you pass a partial agent dataset, `evaluation_config.agent_name` is also required for CLI Mode 4.

```yaml
llm_config:
  base_url: "https://api.openai.com/v1"
  api_key: "your-api-key"
  chat_model: "gpt-4o-mini"
  chat_endpoint: "chat/completions"
  embedding_model: "text-embedding-3-small"
  embedding_endpoint: "embeddings"

evaluation_config:
  metrics:
    - "goal_achievement"
    - "response_coherence"
  metric_params:
    goal_achievement:
      threshold: 0.7
```

### CLI partial dataset with FloTorch

Add `agent_name` when evaluating a partial agent dataset from the CLI:

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

This path requires the optional FloTorch integration:

```bash
pip install "floeval[flotorch]"
```

`create_flotorch_runner()` can read gateway credentials from:

- explicit function arguments
- `llm_config.base_url` and `llm_config.api_key`
- `FLOTORCH_BASE_URL` and `FLOTORCH_API_KEY`

---

## Dataset formats

Agent datasets are loaded through `AgentDataset.from_file(...)` and support `.json` and `.jsonl`.

### Full dataset: pre-captured traces

Use a full dataset when traces are already available and only scoring is required.

```json
{
  "samples": [
    {
      "user_input": "Summarize today's weather for Paris.",
      "reference_outcome": "A short weather summary for Paris.",
      "trace": {
        "messages": [
          {"role": "human", "content": "Summarize today's weather for Paris."},
          {
            "role": "ai",
            "content": "I'll check the weather first.",
            "tool_calls": [
              {"name": "weather_lookup", "args": {"city": "Paris"}}
            ]
          },
          {
            "role": "tool",
            "tool_name": "weather_lookup",
            "content": "Paris: 18C, light rain"
          },
          {
            "role": "ai",
            "content": "Paris is 18C with light rain today."
          }
        ],
        "final_response": "Paris is 18C with light rain today."
      }
    }
  ]
}
```

### Partial dataset: runtime trace capture

Use a partial dataset when Floeval should run the agent and convert the results into full trace-based samples before scoring.

```json
{
  "samples": [
    {
      "user_input": "Book a table for two at 7pm tomorrow.",
      "reference_outcome": "A confirmed reservation for two at 7pm tomorrow.",
      "reference_tool_calls": [
        {
          "name": "create_reservation",
          "args": {"party_size": 2, "time": "7pm tomorrow"}
        }
      ]
    }
  ]
}
```

### Agent dataset fields

| Field | Used in | Notes |
|-------|---------|-------|
| `user_input` | Full and partial datasets | Also accepted as `question` (alias) |
| `trace.messages` | Full datasets | Message roles must be `human`, `ai`, or `tool` |
| `trace.final_response` | Full datasets | Final user-facing answer from the agent |
| `reference_outcome` | Optional | Helps goal-oriented metrics; also accepted as `answer` (alias) |
| `reference_tool_calls` | Optional but required for `tool_call_accuracy` | Expected tool calls for the sample |
| `metadata` | Optional | Extra information carried through into results |

Dataset field aliases: `question` is treated as `user_input`, and `answer` is treated as `reference_outcome`. You can supply datasets using either the alias format or the full field names.

---

## Python API

Import `AgentEvaluation` directly from `floeval.api.agent_evaluation`:

```python
from floeval.api.agent_evaluation import AgentEvaluation
from floeval.config.schemas.io.agent_dataset import AgentDataset
from floeval.config.schemas.io.llm import OpenAIProviderConfig
```

### Mode 1: score pre-captured traces

```python
from floeval.api.agent_evaluation import AgentEvaluation
from floeval.config.schemas.io.agent_dataset import AgentDataset
from floeval.config.schemas.io.llm import OpenAIProviderConfig

dataset = AgentDataset.from_file("agent_dataset.json")

llm_config = OpenAIProviderConfig(
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
    chat_model="gpt-4o-mini",
)

evaluation = AgentEvaluation(
    dataset=dataset,
    llm_config=llm_config,
    metrics=["goal_achievement", "response_coherence"],
)

results = evaluation.run()
print(results.summary)
```

### Mode 2: Python callable with trace collection

Pass `agent=` when the dataset is partial and the agent is available as a Python callable.

```python
from floeval.api.agent_evaluation import AgentEvaluation
from floeval.config.schemas.io.agent_dataset import AgentDataset, PartialAgentSample
from floeval.config.schemas.io.llm import OpenAIProviderConfig
from floeval.utils.agent_trace import capture_trace, log_turn


@capture_trace
def support_agent(user_input: str) -> str:
    response = f"Handled request: {user_input}"
    log_turn(response)
    return response


dataset = AgentDataset(
    samples=[
        PartialAgentSample(
            user_input="Reset my password",
            reference_outcome="Password reset instructions were provided.",
        )
    ]
)

llm_config = OpenAIProviderConfig(
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
    chat_model="gpt-4o-mini",
)

evaluation = AgentEvaluation(
    dataset=dataset,
    agent=support_agent,
    llm_config=llm_config,
    metrics=["goal_achievement"],
)

results = evaluation.run()
```

### Mode 4: FloTorch runner

```python
from floeval.api.agent_evaluation import AgentEvaluation
from floeval.config.schemas.io.agent_dataset import AgentDataset
from floeval.config.schemas.io.llm import OpenAIProviderConfig
from floeval.flotorch import create_flotorch_runner

dataset = AgentDataset.from_file("partial_agent_dataset.json")

llm_config = OpenAIProviderConfig(
    base_url="https://gateway.example/openai/v1",
    api_key="your-gateway-key",
    chat_model="gpt-4o-mini",
)

runner = create_flotorch_runner("support-agent", llm_config=llm_config)

evaluation = AgentEvaluation(
    dataset=dataset,
    agent_runner=runner,
    llm_config=llm_config,
    metrics=["goal_achievement", "ragas:tool_call_accuracy"],
)

results = evaluation.run()
```

### Async execution

`AgentEvaluation` also supports async execution:

```python
results = await evaluation.arun()
```

---

## Agent metrics

### Builtin provider

| Metric | Best for | Notes |
|--------|----------|-------|
| `goal_achievement` | Did the agent satisfy the user request? | Uses an LLM judge; `reference_outcome` helps when available |
| `response_coherence` | Does the final response match the conversation trace? | Uses an LLM judge over the full trace |

### RAGAS provider

| Metric | Best for | Notes |
|--------|----------|-------|
| `ragas:agent_goal_accuracy` | Compare the trace and final response against an expected outcome | Best when `reference_outcome` is present |
| `ragas:tool_call_accuracy` | Check whether the agent used the expected tools | Requires `reference_tool_calls` |

You can mix providers in one run:

```python
metrics = [
    "goal_achievement",
    "response_coherence",
    "ragas:tool_call_accuracy",
]
```

---

## Results

Agent evaluation returns per-sample metric data and a summary:

```python
results = evaluation.run()

results.sample_results
results.summary
```

Each sample result includes:

- `user_input`
- `final_response`
- `reference_outcome`
- `metrics`

The summary contains the average score for each metric that completed successfully.

---

## Related references

- [Agentic Workflow](agentic-workflow.md) for multi-agent DAG evaluation
- [Agent Tracing](agent-tracing.md) for trace capture helpers
- [Metrics](metrics.md) for the full metric catalog
- [API Reference](api-reference.md) for constructors and config shapes
