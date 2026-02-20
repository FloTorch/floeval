# Floeval User Guide

Set up and start evaluating your LLM responses using Floeval. All functionalities, flows, and usage options in one place.

---

## What is Floeval?

Floeval evaluates LLM and RAG system responses using metrics like `answer_relevancy` and `faithfulness`.

### What it solves

- Measures how relevant and accurate your LLM answers are
- Helps you verify RAG systems stay grounded in context

### Two ways to use

| Way | Description |
|-----|-------------|
| **Flow 1: CLI** | Run evaluations from command line |
| **Flow 2: Python** | Integrate into your code |

### Built-in metrics

- **RAGAS & DeepEval**: `answer_relevancy`, `faithfulness`
- Plus create your own custom metrics

---

## Quick decision

| I want to... | Use |
|--------------|-----|
| Run evaluations quickly | CLI (Flow 1) |
| Integrate into my app | Python (Flow 2) |
| Check answer relevancy | `answer_relevancy` metric |
| Check RAG accuracy | `faithfulness` metric |
| Create my own check | Custom metrics |

---

## Quick start

1. `pip install floeval`
2. Create config file (YAML/JSON) with `llm_config` (base_url, api_key, chat_model, embedding_model)
3. Create dataset (JSON/JSONL) with samples
4. `floeval evaluate -c config.yaml -d dataset.json`

!!! note "API credentials"
    Pass your API key in the config file (`llm_config.api_key`) or in your Python script. See [Setup](setup.md) and [Examples](examples.md) for details.

---

## What's next?

| Section | Content |
|---------|---------|
| [Prerequisites & Setup](setup.md) | Python, API key, install, verify |
| [Quick Copy & Run](copy-run.md) | Copy-paste ready examples (no reading required!) |
| [Flow 1: CLI](examples.md#flow-1-using-cli) | Run from command line |
| [Flow 2: Python](examples.md#flow-2-using-python) | Integrate into your app |
| [Metrics & Custom](metrics.md) | Built-in metrics and custom metrics |
| [API Reference](api-reference.md) | Complete config and API reference |
| [Troubleshooting](troubleshooting.md) | Common issues and fixes |
