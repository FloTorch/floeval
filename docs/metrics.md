# Available Metrics

Floeval currently registers metrics across `ragas`, `deepeval`, and `builtin`, plus any `custom` metrics you define at runtime.

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

| Metric | Best for | Notes |
|--------|----------|-------|
| `answer_relevancy` | General LLM answer quality | Works on standard evaluation datasets |
| `faithfulness` | Grounding against retrieved context | Use when samples include `contexts` |
| `context_precision` | Retrieval ranking quality | Retrieval-focused metric |
| `context_recall` | Coverage of relevant information | Retrieval-focused metric |
| `context_entity_recall` | Entity-level recall from context | Useful when preserving entities matters |
| `noise_sensitivity` | Robustness to noisy or irrelevant context | Retrieval-focused stress metric |

### DeepEval

| Metric | Best for | Notes |
|--------|----------|-------|
| `answer_relevancy` | General LLM answer quality | Alternate implementation to RAGAS |
| `faithfulness` | Grounding against context | Uses DeepEval's test-case flow |
| `contextual_precision` | Retrieval precision | Typically uses `contexts` and `ground_truth` |
| `contextual_recall` | Retrieval recall | Typically uses `contexts` and `ground_truth` |
| `contextual_relevancy` | Whether retrieved context is relevant | Typically uses `contexts` |

---

## Agent metrics

These metrics are intended for `AgentEvaluation`, not standard `Evaluation`.

### Builtin

| Metric | Best for | Notes |
|--------|----------|-------|
| `goal_achievement` | Did the agent satisfy the request? | Uses an LLM judge; `reference_outcome` helps |
| `response_coherence` | Is the final answer consistent with the trace? | Uses the conversation trace and final response |

### RAGAS

| Metric | Best for | Notes |
|--------|----------|-------|
| `agent_goal_accuracy` | Compare the agent's result to an expected outcome | Best with `reference_outcome` |
| `tool_call_accuracy` | Compare actual tool use to expected tool use | Requires `reference_tool_calls` |

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

### Agent evaluation datasets

| Field | Commonly used by |
|-------|------------------|
| `trace.messages` | All agent metrics |
| `trace.final_response` | All agent metrics |
| `reference_outcome` | `goal_achievement`, `agent_goal_accuracy` |
| `reference_tool_calls` | `tool_call_accuracy` |

If a metric needs fields that are missing from your samples, it will fail at evaluation time and the failure will be recorded in that sample's metric metadata.

---

## Choosing a provider

### Provider selection

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
- [Agent Evaluation](agent-evaluation.md) for agent-only metrics
- [Custom Metrics](custom-metrics.md) for user-defined scoring
