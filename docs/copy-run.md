# Copy & Run Examples

Copy-paste any script below and run it. No reading required — just run and see what happens!

!!! tip "Quick tip"
    Examples 2 and 7 need **no API key** — try them first! Others need `llm_config` with your API key in the script or config.

---

## Example 1: Built-in RAGAS metrics

`answer_relevancy` + `faithfulness`. Needs llm_config with API key.

```python
#!/usr/bin/env python3
"""Built-in RAGAS metrics (answer_relevancy, faithfulness)."""
from floeval import Evaluation, DatasetLoader
from floeval.config.schemas.io.llm import OpenAIProviderConfig

LLM_CONFIG = {
    "base_url": "https://api.openai.com/v1",
    "api_key": "your-api-key-here",
    "chat_model": "gpt-4o-mini",
    "embedding_model": "text-embedding-3-small",
}
llm_config = OpenAIProviderConfig(**LLM_CONFIG)

dataset = DatasetLoader.from_samples([
    {
        "user_input": "What is RAG?",
        "llm_response": "RAG stands for Retrieval-Augmented Generation.",
        "contexts": ["RAG combines retrieval with generation."]
    },
    {
        "user_input": "How does photosynthesis work?",
        "llm_response": "Photosynthesis converts sunlight into energy.",
        "contexts": ["Plants use chlorophyll.", "Converts CO2 and water into glucose."]
    }
], partial_dataset=False)

evaluation = Evaluation(
    dataset=dataset,
    llm_config=llm_config,
    metrics=["answer_relevancy", "faithfulness"],
    default_provider="ragas"
)

results = evaluation.run()
print("\n📊 Aggregate Scores:", results.aggregate_scores)
print("✅ Done!")
```

---

## Example 2: Simple custom metric — NO API KEY!

`@custom_metric`, function-based. No LLM required.

```python
#!/usr/bin/env python3
"""Simple custom metric — NO API KEY NEEDED!"""
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
], partial_dataset=False)

evaluation = Evaluation(dataset=dataset, metrics=["politeness", "response_length"])
results = evaluation.run()
print("\n📊 Scores:", results.aggregate_scores)
print("✅ Done! (No API key used)")
```

---

## Example 3: Criteria-based (LLM-as-judge)

`criteria()` natural language evaluation. Needs llm_config.

```python
#!/usr/bin/env python3
"""Criteria-based metric (LLM-as-judge)."""
from floeval import Evaluation, DatasetLoader
from floeval.config.schemas.io.llm import OpenAIProviderConfig
from floeval.api.metrics.custom import criteria

llm_config = OpenAIProviderConfig(
    base_url="https://api.openai.com/v1",
    api_key="your-api-key-here",
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
], partial_dataset=False)

evaluation = Evaluation(dataset=dataset, llm_config=llm_config, metrics=[empathy])
results = evaluation.run()
print("\n📊 Scores:", results.aggregate_scores)
print("✅ Done!")
```

---

## Example 4: Custom with llm.generate()

Custom metric using `llm` parameter. Needs llm_config.

```python
#!/usr/bin/env python3
"""Custom metric with llm.generate()."""
from floeval import Evaluation, DatasetLoader
from floeval.config.schemas.io.llm import OpenAIProviderConfig
from floeval.api.metrics.custom import custom_metric

llm_config = OpenAIProviderConfig(
    base_url="https://api.openai.com/v1",
    api_key="your-api-key-here",
    chat_model="gpt-4o-mini",
    embedding_model="text-embedding-3-small",
)

@custom_metric(threshold=0.5)
def is_professional(response: str, llm) -> float:
    answer = llm.generate(f"""
        Is this professional for business? Answer only 'yes' or 'no'.
        Response: {response}
    """)
    return 1.0 if "yes" in answer.lower() else 0.0

dataset = DatasetLoader.from_samples([
    {"user_input": "Opinion?", "llm_response": "Based on data, I recommend caution."},
    {"user_input": "Opinion?", "llm_response": "LOL that's dumb!"},
], partial_dataset=False)

evaluation = Evaluation(dataset=dataset, llm_config=llm_config, metrics=["is_professional"])
results = evaluation.run()
print("\n📊 Scores:", results.aggregate_scores)
print("✅ Done!")
```

---

## Example 5: Mixed built-in + custom

Combine built-in RAGAS metrics with custom function. Needs llm_config.

```python
#!/usr/bin/env python3
"""Mixed: custom + built-in metrics together."""
from floeval import Evaluation, DatasetLoader
from floeval.config.schemas.io.llm import OpenAIProviderConfig
from floeval.api.metrics.custom import custom_metric

llm_config = OpenAIProviderConfig(
    base_url="https://api.openai.com/v1",
    api_key="your-api-key-here",
    chat_model="gpt-4o-mini",
    embedding_model="text-embedding-3-small",
)

@custom_metric(threshold=0.3)
def response_length(response: str) -> float:
    return min(len(response) / 80.0, 1.0)

dataset = DatasetLoader.from_samples([
    {
        "user_input": "What is Python?",
        "llm_response": "Python is a programming language for data science and web dev.",
        "contexts": ["Python is widely used."]
    },
    {"user_input": "What is JS?", "llm_response": "JavaScript.", "contexts": ["JS runs in browsers."]}
], partial_dataset=False)

evaluation = Evaluation(
    dataset=dataset,
    llm_config=llm_config,
    metrics=["response_length", "answer_relevancy", "faithfulness"],
    default_provider="ragas"
)

results = evaluation.run()
print("\n📊 Scores:", results.aggregate_scores)
print("✅ Done!")
```

---

## Example 6: Partial dataset — questions only, Floeval generates answers

Only `user_input` required. Floeval calls your LLM at runtime to get `llm_response`, then runs metrics. Needs `dataset_generator_model`.

```python
#!/usr/bin/env python3
"""Partial dataset: Floeval generates llm_response, then evaluates."""
from floeval import Evaluation, DatasetLoader
from floeval.config.schemas.io.llm import OpenAIProviderConfig

LLM_CONFIG = {
    "base_url": "https://api.openai.com/v1",
    "api_key": "your-api-key-here",
    "chat_model": "gpt-4o-mini",
    "embedding_model": "text-embedding-3-small",
}
llm_config = OpenAIProviderConfig(**LLM_CONFIG)

# Partial: no llm_response; contexts optional
partial_ds = DatasetLoader.from_samples([
    {"user_input": "What is Python?", "contexts": ["Python is widely used."]},
    {"user_input": "What is RAG?", "contexts": ["RAG combines retrieval with generation."]}
], partial_dataset=True)

evaluation = Evaluation(
    dataset=partial_ds,
    llm_config=llm_config,
    metrics=["answer_relevancy", "faithfulness"],
    default_provider="ragas",
    dataset_generator_model="gpt-4o-mini",
)
results = evaluation.run()
print("\n📊 Aggregate Scores:", results.aggregate_scores)
print("✅ Done!")
```

---

## Example 7: CLI - Generate responses, then evaluate (2-step workflow)

Generate responses in one step, evaluate in another. Great for auditing generated responses.

### Step 1: Create partial dataset (questions only)

`questions.json`:

```json
{
  "samples": [
    {"user_input": "What is Python?", "contexts": ["Python is a programming language."]},
    {"user_input": "What is RAG?", "contexts": ["RAG stands for Retrieval-Augmented Generation."]},
    {"user_input": "How does machine learning work?", "contexts": ["ML uses algorithms to learn from data."]}
  ]
}
```

### Step 2: Create config with llm_config + dataset_generation_config

`config.yaml`:

```yaml
llm_config:
  base_url: "https://api.openai.com/v1"
  api_key: "your-api-key-here"
  chat_model: "gpt-4o-mini"
  embedding_model: "text-embedding-3-small"
  system_prompt: "You are a helpful assistant. Answer concisely."

evaluation_config:
  metrics:
    - "answer_relevancy"
    - "faithfulness"
  default_provider: "ragas"

dataset_generation_config:
  generator_model: "gpt-4o-mini"
```

### Step 3: Generate responses

```bash
floeval generate -c config.yaml -d questions.json -o complete_dataset.json
```

This creates `complete_dataset.json` with `llm_response` populated for each sample.

### Step 4: (Optional) Inspect generated responses

```bash
cat complete_dataset.json | python -m json.tool | head -30
```

### Step 5: Evaluate the complete dataset

```bash
floeval evaluate -c config.yaml -d complete_dataset.json -o results.json
```

### Why this workflow?

✅ **Generates once** - Reuse the same generated dataset for multiple metric configurations
✅ **Audit responses** - Review generated responses before evaluation
✅ **Parallel workflows** - Generate in one job, evaluate in another
✅ **Reusable dataset** - Save and version control the generated dataset

---

## Example 8: Custom with multiple params — NO API KEY

Custom metric with `response`, `question`, `contexts`. No LLM required.

```python
#!/usr/bin/env python3
"""Custom metric with question + contexts. NO API KEY!"""
from floeval import Evaluation, DatasetLoader
from floeval.api.metrics.custom import custom_metric

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

dataset = DatasetLoader.from_samples([
    {
        "user_input": "What is RAG?",
        "llm_response": "RAG stands for Retrieval-Augmented Generation.",
        "contexts": ["RAG combines retrieval with generation."]
    },
    {
        "user_input": "How does ML work?",
        "llm_response": "Machine learning uses algorithms.",
        "contexts": ["RAG stands for Retrieval-Augmented Generation."]
    }
], partial_dataset=False)

evaluation = Evaluation(dataset=dataset, metrics=["context_relevance"])
results = evaluation.run()
print("\n📊 Scores:", results.aggregate_scores)
print("✅ Done! (No API key)")
```

---

## Next steps

- **[Examples](examples.md)** — Full usage patterns
- **[Custom Metrics](custom-metrics.md)** — Create your own
- **[API Reference](api-reference.md)** — Full reference
