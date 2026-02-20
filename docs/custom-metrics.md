# Custom Metrics

Create your own evaluation logic when built-in metrics don't cover your needs.

---

## Two types of custom metrics

### 1. @custom_metric decorator

Function-based metric. Supports params: `response`, `question`, `contexts`, `context` (MetricContext), `llm` (SimpleLLMHelper), `sample`.

```python
from floeval.api.metrics.custom import custom_metric

@custom_metric(threshold=0.5)
def response_length(response: str) -> float:
    return min(len(response) / 100.0, 1.0)
```

### 2. criteria() — LLM-as-judge

Natural language description. LLM evaluates.

```python
from floeval.api.metrics.custom import criteria

empathy = criteria(
    name="empathy",
    description="Rate empathy 0-1. Consider acknowledgment, understanding, support.",
    threshold=0.7
)
```

---

## @custom_metric options

| Parameter | Description | Example |
|-----------|-------------|---------|
| `threshold` | Pass/fail threshold (0–1). Default: 0.5 | `@custom_metric(threshold=0.7)` |
| `execute_via` | Route to "ragas" or "deepeval" provider. Default: standalone (no provider needed) | `@custom_metric(execute_via="ragas")` |
| `name` | Custom metric name. Default: function name | `@custom_metric(name="my_metric")` |

### execute_via: Running metrics through providers

By default, custom metrics run standalone (no API calls). Use `execute_via` to integrate with RAGAS or DeepEval:

```python
@custom_metric(execute_via="ragas", threshold=0.6)
def semantic_consistency(response: str, llm) -> float:
    """Uses RAGAS provider context and LLM."""
    # This metric can now access provider features
    return 0.8

@custom_metric(execute_via="deepeval", threshold=0.7)
def tone_check(response: str) -> float:
    """Runs through DeepEval provider."""
    return 0.9
```

---

## Available parameters in @custom_metric

Custom metric functions can access these parameters (pass only what you need):

| Parameter | Type | Description |
|-----------|------|-------------|
| `response` | str | The LLM's response (aka `llm_response`) |
| `question` | str | The user's question (aka `user_input`) |
| `contexts` | list[str] | Retrieved documents (if using RAG) |
| `context` | MetricContext | Full evaluation context object |
| `llm` | SimpleLLMHelper | LLM interface for calling the model |
| `sample` | dict | Full sample data |

### Example: Using various parameters

```python
from floeval.api.metrics.custom import custom_metric

# Simple: just response
@custom_metric()
def response_length(response: str) -> float:
    return min(len(response) / 100.0, 1.0)

# Multiple params: response + question
@custom_metric()
def covers_topic(response: str, question: str) -> float:
    return 1.0 if len(response) > 0 and len(question) > 0 else 0.0

# With contexts (RAG)
@custom_metric()
def uses_context(response: str, contexts: list) -> float:
    if not contexts:
        return 0.5  # No contexts provided
    # Check if response uses info from contexts
    ctx_text = " ".join(contexts).lower()
    return 0.8 if any(word in response for word in ctx_text.split()) else 0.2

# With LLM helper (needs llm_config)
@custom_metric()
def tone_assessment(response: str, llm) -> float:
    result = llm.generate(f"Is this professional? {response}")
    return 1.0 if "yes" in result.lower() else 0.0
```

---

Your custom metric can return:

- **float** — Simple score 0–1
- **dict** — With `"score"` and optional `"metadata"`
- **MetricResult** — Full result object

---

## Examples

### Simple function (no API key needed)

```python
from floeval import Evaluation, DatasetLoader
from floeval.api.metrics.custom import custom_metric

@custom_metric(threshold=0.3)
def response_length(response: str) -> float:
    return min(len(response) / 100.0, 1.0)

@custom_metric(threshold=0.5)
def politeness(response: str) -> float:
    polite_words = ["please", "thank you", "thanks"]
    count = sum(1 for w in polite_words if w in response.lower())
    return min(count / 3.0, 1.0)

dataset = DatasetLoader.from_samples([
    {"user_input": "Help?", "llm_response": "Please let me help. Thank you for asking."},
    {"user_input": "What?", "llm_response": "No idea."},
])

evaluation = Evaluation(dataset=dataset, metrics=["politeness", "response_length"])
results = evaluation.run()
```

### Criteria-based (needs llm_config)

```python
from floeval import Evaluation, DatasetLoader
from floeval.config.schemas.io.llm import OpenAIProviderConfig
from floeval.api.metrics.custom import criteria

llm_config = OpenAIProviderConfig(
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
    chat_model="gpt-4o-mini",
    embedding_model="text-embedding-3-small",
)

empathy = criteria(
    name="empathy",
    description="Rate empathy 0-1. Consider acknowledgment, understanding, support.",
    threshold=0.6
)

dataset = DatasetLoader.from_samples([
    {"user_input": "I'm stressed.", "llm_response": "I understand. Would you like to talk?"},
    {"user_input": "I'm stressed.", "llm_response": "Just work harder."},
])

evaluation = Evaluation(dataset=dataset, llm_config=llm_config, metrics=[empathy])
results = evaluation.run()
```

### Custom with llm.generate()

```python
from floeval.api.metrics.custom import custom_metric

@custom_metric(threshold=0.5)
def is_professional(response: str, llm) -> float:
    answer = llm.generate(f"""
        Is this professional for business? Answer only 'yes' or 'no'.
        Response: {response}
    """)
    return 1.0 if "yes" in answer.lower() else 0.0
```

### Custom with multiple params (question, contexts)

```python
@custom_metric(threshold=0.4)
def context_relevance(response: str, question: str, contexts: list) -> float:
    if not response or not question:
        return 0.0
    q_words = set(question.lower().split())
    r_words = set(response.lower().split())
    q_match = len(q_words & r_words) / max(len(q_words), 1)
    ctx_usage = 0.0
    if contexts:
        ctx_text = " ".join(contexts).lower()
        ctx_words = set(ctx_text.split())
        ctx_usage = min(len(ctx_words & r_words) / max(len(r_words), 1) * 2, 1.0)
    return min(q_match * 0.6 + ctx_usage * 0.4, 1.0)
```

---

## Mix custom with built-in

```python
evaluation = Evaluation(
    dataset=dataset,
    llm_config=llm_config,
    metrics=["response_length", "answer_relevancy", "faithfulness"],
    default_provider="ragas"
)
```

---

## Next steps

- **[Copy & Run](copy-run.md)** — Copy-paste full examples
- **[Metrics](metrics.md)** — Built-in metrics reference
- **[Examples](examples.md)** — All usage patterns
