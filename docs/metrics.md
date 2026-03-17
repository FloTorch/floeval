# Available Metrics

Floeval registers metrics across `ragas`, `deepeval`, and `builtin`, plus any `custom` metrics you define at runtime.

---

## Metrics by eval type

Use this table as a quick reference to pick metrics for each evaluation type.

| Eval type | Provider | Metrics |
|-----------|----------|---------|
| `llm` | ragas | `answer_relevancy` |
| `llm` | deepeval | `answer_relevancy` |
| `rag` | ragas | `answer_relevancy`, `faithfulness`, `context_precision`, `context_recall`, `context_entity_recall`, `noise_sensitivity` |
| `rag` | deepeval | `answer_relevancy`, `faithfulness`, `contextual_precision`, `contextual_recall`, `contextual_relevancy` |
| `prompt (no RAG)` | ragas | `answer_relevancy` |
| `prompt (no RAG)` | deepeval | `answer_relevancy` |
| `prompt (with RAG)` | ragas | Same as `rag` |
| `prompt (with RAG)` | deepeval | Same as `rag` |
| `agent` | builtin | `goal_achievement`, `response_coherence` |
| `agent` | ragas | `agent_goal_accuracy`, `tool_call_accuracy` |
| `agentic_workflow` | builtin | `goal_achievement`, `response_coherence` |
| `agentic_workflow` | ragas | `agent_goal_accuracy`, `tool_call_accuracy` |

---

## How to specify metrics

You can reference metrics in four ways:

| Format | Example | When to use it |
|--------|---------|----------------|
| Plain string | `"answer_relevancy"` | Use with `default_provider` |
| Provider-qualified string | `"deepeval:faithfulness"` | Use when the metric exists in multiple providers |
| Dict spec | `{"id": "faithfulness", "provider": "deepeval", "params": {"threshold": 0.8}}` | Use when you need per-metric params |
| Metric instance | `my_metric` | Use with custom programmatic metrics |

If the same metric ID exists in more than one provider and you do not set `default_provider`, use the `provider:metric` form.

---

## Standard LLM and RAG metrics

### RAGAS

| Metric | Best for | Required fields |
|--------|----------|-----------------|
| `answer_relevancy` | General LLM answer quality | `user_input`, `llm_response` |
| `faithfulness` | Grounding against retrieved context | `user_input`, `llm_response`, `contexts` |
| `context_precision` | Whether relevant contexts are ranked first | `user_input`, `llm_response`, `contexts`, `ground_truth` |
| `context_recall` | Coverage of reference information by retrieved contexts | `user_input`, `llm_response`, `contexts`, `ground_truth` |
| `context_entity_recall` | Entity-level coverage in retrieved contexts | `user_input`, `llm_response`, `contexts`, `ground_truth` |
| `noise_sensitivity` | How often the model produces incorrect claims from noisy context | `user_input`, `llm_response`, `contexts`, `ground_truth` |

### DeepEval

| Metric | Best for | Required fields |
|--------|----------|-----------------|
| `answer_relevancy` | General LLM answer quality | `user_input`, `llm_response` |
| `faithfulness` | Grounding against context | `user_input`, `llm_response`, `contexts` |
| `contextual_precision` | Retrieval ranking precision | `user_input`, `llm_response`, `contexts`, `ground_truth` |
| `contextual_recall` | Retrieval recall against ground truth | `user_input`, `llm_response`, `contexts`, `ground_truth` |
| `contextual_relevancy` | Whether retrieved context is relevant to the question | `user_input`, `llm_response`, `contexts` |
| `hallucination` | Factual contradictions versus the provided context | `user_input`, `llm_response`, `contexts` |
| `toxicity` | Toxic, harmful, or offensive content | `user_input`, `llm_response` |
| `exact_match` | Exact string match against ground truth (no LLM required) | `llm_response`, `ground_truth` |
| `pattern_match` | Regex pattern match against the response (no LLM required) | `llm_response` |
| `json_correctness` | JSON schema validation of the response | `llm_response` |

---

## Agent metrics

These metrics are for `AgentEvaluation` only. Do not use them with standard `Evaluation`.

### Builtin

| Metric | Best for | Required fields | Notes |
|--------|----------|-----------------|-------|
| `goal_achievement` | Did the agent satisfy the user request? | `trace`, `user_input` | Uses an LLM judge; `reference_outcome` helps when available |
| `response_coherence` | Is the final answer consistent with the conversation trace? | `trace` | No `reference_outcome` needed; scores coherence of the trace itself |

### RAGAS

| Metric | Best for | Required fields | Notes |
|--------|----------|-----------------|-------|
| `agent_goal_accuracy` | Compare the agent's final result to an expected outcome | `trace`, `user_input`, `reference_outcome` | Best when `reference_outcome` is present |
| `tool_call_accuracy` | Check whether actual tool calls match expected tool calls | `trace`, `reference_tool_calls` | Silently skips samples that have no `reference_tool_calls`; score is `null` for those samples |

---

## Field guidance

### Standard evaluation datasets

| Field | Commonly used by |
|-------|------------------|
| `user_input` | All standard metrics |
| `llm_response` | All standard metrics |
| `contexts` | `faithfulness`, retrieval-focused metrics, contextual metrics |
| `ground_truth` | Recall and precision style metrics, especially DeepEval contextual metrics |
| `prompt_id` | Generated datasets that came from prompt expansion |

Dataset field aliases are also accepted: `question` is treated as `user_input`, and `answer` is treated as `ground_truth`.

### Agent evaluation datasets

| Field | Commonly used by |
|-------|------------------|
| `trace.messages` | All agent metrics |
| `trace.final_response` | All agent metrics |
| `reference_outcome` | `goal_achievement`, `agent_goal_accuracy` |
| `reference_tool_calls` | `tool_call_accuracy` |

Agent dataset field aliases: `question` maps to `user_input`, `answer` maps to `reference_outcome`.

If a metric needs fields that are missing from your samples, the failure is recorded in that sample's metric metadata and the overall run continues.

---

## Choosing a provider

- Use `ragas` when you want the default provider for common answer and retrieval metrics.
- Use `deepeval` when you want its contextual metrics or prefer its scoring behavior.
- Use `builtin` for agent-specific judge metrics.
- Use `custom` for domain-specific checks you write yourself.

### Example configurations

```yaml
evaluation_config:
  default_provider: "ragas"
  metrics:
    - "answer_relevancy"
    - "faithfulness"
```

```yaml
evaluation_config:
  metrics:
    - "ragas:answer_relevancy"
    - "deepeval:contextual_relevancy"
```

```python
evaluation = Evaluation(
    dataset=dataset,
    llm_config=llm_config,
    metrics=[
        {"id": "faithfulness", "provider": "deepeval", "params": {"threshold": 0.8}},
        "ragas:answer_relevancy",
    ],
)
```

---

## Thresholds and params

Set thresholds through `metric_params` or through metric dict specs.

```yaml
evaluation_config:
  metrics:
    - "answer_relevancy"
    - "faithfulness"
  metric_params:
    answer_relevancy:
      threshold: 0.7
    faithfulness:
      threshold: 0.8
```

Provider-qualified keys also work:

```yaml
evaluation_config:
  metrics:
    - "ragas:answer_relevancy"
    - "deepeval:faithfulness"
  metric_params:
    ragas:answer_relevancy:
      threshold: 0.7
    deepeval:faithfulness:
      threshold: 0.8
```

---

## Understanding results

All built-in provider scores are normalized to the `0.0` to `1.0` range.

| Score range | Typical interpretation |
|-------------|------------------------|
| `0.9 - 1.0` | Strong |
| `0.7 - 0.9` | Good |
| `0.5 - 0.7` | Mixed |
| `0.0 - 0.5` | Weak |

Use:

- aggregate scores for overall system quality
- per-sample scores to inspect failure patterns
- metric metadata to understand thresholds, provider, and recorded errors

---

## Discover metrics programmatically

```python
from floeval.api.metrics.registry import MetricRegistry

print(MetricRegistry.list_providers())
print(MetricRegistry.list_metrics("ragas"))
print(MetricRegistry.list_metrics("deepeval"))
print(MetricRegistry.list_metrics("builtin"))
```

`custom` metrics appear in the registry after you define them.

---

## Next steps

- [Examples](examples.md) for provider-routing examples
- [Prompt Evaluation](prompt-evaluation.md) for prompt-specific metric choices
- [Agent Evaluation](agent-evaluation.md) for agent-only metrics
- [Agentic Workflow](agentic-workflow.md) for multi-agent DAG evaluation
- [Custom Metrics](custom-metrics.md) for user-defined scoring
