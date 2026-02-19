# Available Metrics

Built-in metrics from RAGAS and DeepEval, plus guidance on choosing metrics for your use case.

---

## Built-in metrics

| Metric | Provider | Required fields | What it measures |
|--------|----------|----------------|------------------|
| answer_relevancy | RAGAS, DeepEval | user_input, llm_response | How relevant is the answer to the question? |
| faithfulness | RAGAS, DeepEval | user_input, llm_response, **contexts** | Does the answer stay grounded in the provided contexts? |

### Metric details

#### answer_relevancy

- **What it does**: Measures how well the LLM's response answers the user's question
- **Score range**: 0–1 (higher is better)
- **Use case**: All evaluations (both retrieval-augmented and standalone LLM)
- **Example**: Question: "What is Python?" Answer: "Python is a programming language" → High relevancy score

#### faithfulness

- **What it does**: Measures whether the LLM's response is supported by the provided context
- **Score range**: 0–1 (higher is better)  
- **Use case**: RAG systems, where you want to verify the LLM doesn't hallucinate
- **Requires**: `contexts` field in your dataset (list of reference documents)
- **Example**: Context: "The capital of France is Paris." Answer: "The capital of France is Paris." → High faithfulness
- **Counter-example**: Context: "Python is a programming language." Answer: "Python was created in 1992." → Low faithfulness (info not in context)

---

## Full vs. Partial datasets

Understanding when to use each dataset type helps optimize your evaluation workflow.

### Full dataset

**Use when**: You have pre-generated LLM responses ready to evaluate.

```json
{
  "samples": [
    {
      "user_input": "What is Python?",
      "llm_response": "Python is a programming language.",
      "contexts": ["Python is widely used in data science."]
    }
  ]
}
```

**Workflow**:

1. Generate responses from your LLM offline
2. Save to JSON/JSONL
3. Run evaluation: `floeval evaluate -c config.yaml -d dataset.json`

**Pros**:

- Faster evaluation (no LLM calls during evaluation)
- Good for batch processing
- Can generate responses in parallel

**Cons**:

- Requires pre-generated responses
- Takes up more disk space

### Partial dataset

**Use when**: You only have questions and want Floeval to generate responses on-the-fly.

```json
{
  "samples": [
    {
      "user_input": "What is Python?"
    },
    {
      "user_input": "How does RAG work?",
      "contexts": ["RAG combines retrieval with generation."]
    }
  ]
}
```

**Requires** in config:

```yaml
dataset_generation_config:
  generator_model: "gpt-4o-mini"
```

**Workflow**:

1. Save questions to JSON/JSONL (no responses needed)
2. Run evaluation: `floeval evaluate -c config.yaml -d dataset.json`
3. Floeval automatically generates responses using the specified model

**Pros**:

- Simpler setup; no need to pre-generate
- Single command; generates and evaluates in one pass
- Good for quick iterations

**Cons**:

- Slower (LLM calls happen during evaluation)
- Higher API costs

### Generating then saving (advanced)

Generate responses separately, then reuse them:

```bash
floeval generate -c config.yaml -d partial.json -o complete.json
floeval evaluate -c config.yaml -d complete.json
```

This way, you generate responses once and can evaluate them multiple times.

---

## Sample fields reference

| Field | Required | Description |
|-------|----------|-------------|
| user_input | Yes | Question or prompt |
| llm_response | Yes | LLM's answer |
| contexts | For faithfulness | List of retrieved documents |
| ground_truth | Optional | Expected answer |
| metadata | Optional | Extra metadata |

---

## Quick decision guide

**Step 1: Do you have a RAG system?**

```
Do you have retrieved contexts to include?
│
├─ NO  → Use: answer_relevancy only
│
└─ YES → Use: answer_relevancy + faithfulness
```

**Step 2: Do you have LLM responses ready?**

| Scenario | Dataset type | Command |
|----------|--------------|---------|
| You have pre-generated responses | Full dataset | `floeval evaluate -c config.yaml -d dataset.json` |
| You only have questions; want Floeval to generate responses | Partial dataset | `floeval evaluate -c config.yaml -d dataset.json` (auto-detected) |
| You want to generate & save responses separately | Partial dataset | `floeval generate -c config.yaml -d dataset.json -o complete.json` |

---

## Using both metrics

### Configuration (YAML)

```yaml
evaluation_config:
  default_provider: "ragas"
  metrics:
    - "answer_relevancy"
    - "faithfulness"
```

### Configuration (Python)

```python
evaluation = Evaluation(
    dataset=dataset,
    llm_config=llm_config,
    metrics=["answer_relevancy", "faithfulness"],
    default_provider="ragas"
)
```

---

## Providers: RAGAS vs DeepEval

Both providers implement the same metrics but use different underlying implementations.

| Aspect | RAGAS | DeepEval |
|--------|-------|----------|
| **Default** | Yes | Available as alternative |
| **Speed** | Generally faster | Slightly slower |
| **Accuracy** | Good; research-backed | Good; industry-standard |
| **Use case** | RAG evaluation | General-purpose LLM evaluation |
| **Setup** | `default_provider: "ragas"` | `"deepeval:answer_relevancy"` |

### Selecting a provider

```yaml
# Use RAGAS (default)
evaluation_config:
  default_provider: "ragas"
  metrics:
    - "answer_relevancy"
    - "faithfulness"
```

```yaml
# Mix providers
evaluation_config:
  metrics:
    - "ragas:answer_relevancy"      # From RAGAS
    - "deepeval:faithfulness"        # From DeepEval
```

```python
# Python: specify in metric dict
evaluation = Evaluation(
    dataset=dataset,
    metrics=[
        "ragas:answer_relevancy",
        "deepeval:faithfulness"
    ]
)
```

**Recommendation**: Start with RAGAS (default). Switch to DeepEval if you need specific features or different evaluation behavior.

---

## Setting thresholds

Thresholds define the score boundary for pass/fail decisions.

### YAML configuration

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

### Python configuration

```python
evaluation = Evaluation(
    dataset=dataset,
    llm_config=llm_config,
    metrics=["answer_relevancy", "faithfulness"],
    metric_params={
        "answer_relevancy": {"threshold": 0.7},
        "faithfulness": {"threshold": 0.8}
    }
)
```

### Threshold recommendations

| Environment | Threshold | Notes |
|-------------|-----------|-------|
| Development | 0.5 | Lenient; use for iteration |
| Staging | 0.7 | Moderate; ensures basic quality |
| Production | 0.8–0.9 | Strict; high-quality outputs |

- **Sample pass rate**: If 80% of samples exceed threshold, evaluation passes
- **Adjust based on your domain**: Customer-facing tasks may need higher thresholds

---

## Understanding scores

All built-in metrics return scores between 0 and 1.

| Score range | Interpretation | Action |
|---|---|---|
| 0.9–1.0 | Excellent ✅ | Production-ready |
| 0.7–0.9 | Good ✅ | Acceptable for most use cases |
| 0.5–0.7 | Needs improvement ⚠️ | Iterate; investigate failure patterns |
| 0.0–0.5 | Poor ❌ | Critical; requires redesign |

### Interpreting sample results

```python
results = evaluation.run()

# Check individual sample scores
for sample in results.sample_results:
    print(f"Question: {sample['user_input']}")
    print(f"Answer: {sample['llm_response']}")
    print(f"Metrics: {sample['metrics']}")
    # Output: {"answer_relevancy": 0.92, "faithfulness": 0.87}
```

### Aggregate vs. sample scores

- **Aggregate scores**: Average across all samples (e.g., 0.89 = 89% average relevancy)
- **Sample scores**: Score for each individual evaluation sample
- **Use aggregate** for overall system performance
- **Use sample scores** to identify problematic responses and improve your system

---

## Common issues

### "Field required: contexts"

**Problem:** Using `faithfulness` without contexts in dataset.

**Solution:** Either add contexts to your dataset, or use only `answer_relevancy`:

```yaml
evaluation_config:
  metrics:
    - "answer_relevancy"  # Remove faithfulness
```

---

## Next steps

- **[Custom Metrics](custom-metrics.md)** — Create your own evaluation logic
- **[Examples](examples.md)** — See all usage examples
- **[Copy & Run](copy-run.md)** — Copy-paste examples
