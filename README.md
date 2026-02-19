# Floeval

Evaluation framework for LLM and RAG systems.

## Overview

Floeval is a flexible evaluation framework designed to support multiple metric providers and execution backends.

## Structure

- [api/](https://github.com/FloTorch/Floeval/tree/dev/floeval/api) - Public API (Evaluation, Dataset, Sample, Metrics)
- [core/execution/](https://github.com/FloTorch/Floeval/tree/dev/floeval/core/execution) - Execution engine (LLM calls, response synthesis)
- [metric_providers/](https://github.com/FloTorch/Floeval/tree/dev/floeval/metric_providers) - Metrics organized by provider (builtin, ragas, deepeval)
- [config/schemas/](https://github.com/FloTorch/Floeval/tree/dev/floeval/config/schemas) - Configuration schemas and data models
- [cli/](https://github.com/FloTorch/Floeval/tree/dev/floeval/cli) - Command-line interface
- [utils/](https://github.com/FloTorch/Floeval/tree/dev/floeval/utils) - Utility functions (loaders, gateways, etc.)

## Installation

```bash
pip install floeval
```

Or from source:

```bash
pip install -e .
```

## Quick Start

### Python API

```python
from floeval import Evaluation, DatasetLoader
from floeval.config.schemas.io.llm import OpenAIProviderConfig

llm_config = OpenAIProviderConfig(
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
    chat_model="gpt-4o-mini",
    embedding_model="text-embedding-3-small"
)

dataset = DatasetLoader.from_samples([
    {"user_input": "What is RAG?", "llm_response": "RAG is Retrieval-Augmented Generation."}
])

evaluation = Evaluation(
    dataset=dataset,
    llm_config=llm_config,
    metrics=["answer_relevancy", "faithfulness"]
)

results = evaluation.run()
print(results.aggregate_scores)
```

### CLI

```bash
# Evaluate with full dataset
floeval evaluate -c config.yaml -d dataset.json -o results.json

# Or use partial dataset (generate + evaluate in one step)
floeval evaluate -c config.yaml -d partial_dataset.json -o results.json

# Or generate responses separately, then evaluate
floeval generate -c config.yaml -d partial_dataset.json -o complete.json
floeval evaluate -c config.yaml -d complete.json -o results.json
```

## Documentation

Full documentation available in `docs/`:

- [Setup & Prerequisites](docs/setup.md)
- [All Examples](docs/examples.md)
- [Available Metrics](docs/metrics.md)
- [Custom Metrics](docs/custom-metrics.md)
- [API Reference](docs/api-reference.md)

## License

MIT
