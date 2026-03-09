# Floeval

Multi-backend evaluation framework for LLM and RAG systems.

## Overview

Floeval supports the following workflows:

- evaluating full datasets that already contain `llm_response`
- generating responses from partial datasets and scoring them in the same run
- routing metrics across `ragas`, `deepeval`, `builtin`, and `custom`
- evaluating agent traces with agent-specific metrics
- capturing traces from Python callables, LangChain-style agents, or optional FloTorch runners

## Features

- **CLI and Python API**: run evaluations from config files or integrate directly into code
- **Multi-provider metrics**: mix `ragas`, `deepeval`, builtin, and custom metrics in one evaluation
- **Prompt-aware generation**: expand partial samples with `prompt_ids` and `prompts_file`
- **Agent evaluation**: score pre-captured traces or collect traces at runtime
- **Custom metrics**: define function-based metrics or LLM-as-judge criteria

## Installation

Version `0.1.0b1` is a pre-release. Installing from PyPI may require `--pre`:

```bash
pip install --pre floeval
```

Optional FloTorch support for Mode 4 agent evaluation:

```bash
pip install "floeval[flotorch]"
```

Development install:

```bash
pip install -e .
pip install -e .[dev]
```

## Project Structure

- [api/](https://github.com/FloTorch/floeval/tree/dev/floeval/api) - public evaluation APIs and dataset loaders
- [core/execution/](https://github.com/FloTorch/floeval/tree/dev/floeval/core/execution) - response generation and execution internals
- [metric_providers/](https://github.com/FloTorch/floeval/tree/dev/floeval/metric_providers) - provider-specific metric implementations
- [config/schemas/](https://github.com/FloTorch/floeval/tree/dev/floeval/config/schemas) - config, dataset, and prompt schemas
- [cli/](https://github.com/FloTorch/floeval/tree/dev/floeval/cli) - command-line entry points
- [utils/](https://github.com/FloTorch/floeval/tree/dev/floeval/utils) - trace capture, loaders, and helper utilities

## Quick Start

### Python API

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
    metrics=["answer_relevancy", "faithfulness"],
    default_provider="ragas",
)

results = evaluation.run()
print(results.aggregate_scores)
```

### CLI

```bash
# Evaluate a full dataset
floeval evaluate -c config.yaml -d dataset.json -o results.json

# Evaluate a partial dataset (generate + score in one run)
floeval evaluate -c config.yaml -d partial_dataset.json -o results.json

# Generate first, then evaluate later
floeval generate -c config.yaml -d partial_dataset.json -o complete.json
floeval evaluate -c config.yaml -d complete.json -o results.json

# Evaluate agent traces
floeval evaluate -c agent_config.yaml -d agent_dataset.json --agent -o agent_results.json
```

## Documentation

Detailed docs live in `docs/`:

- [Setup & Prerequisites](docs/setup.md)
- [Examples](docs/examples.md)
- [Agent Evaluation](docs/agent-evaluation.md)
- [Agent Tracing](docs/agent-tracing.md)
- [Metrics](docs/metrics.md)
- [Custom Metrics](docs/custom-metrics.md)
- [API Reference](docs/api-reference.md)
- [Troubleshooting](docs/troubleshooting.md)

## License

MIT
