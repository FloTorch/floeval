# Floeval

Multi-backend evaluation framework for LLM, RAG, prompt, and agent systems.

## Overview

Floeval supports five evaluation types:

| Eval type | What you are scoring | Key dataset fields |
|-----------|----------------------|--------------------|
| **LLM** | Direct question-answer quality without retrieval | `user_input`, `llm_response` |
| **RAG** | Answer quality and retrieval performance with context | `user_input`, `llm_response`, `contexts` |
| **Prompt** | One or more system prompts against the same dataset | Partial dataset + `prompts_file` (with or without RAG) |
| **Agent** | Single-agent trace quality, tool use, and goal achievement | `AgentDataset` (full or partial) |
| **Agentic Workflow** | Multi-agent DAG pipelines scored end-to-end | `AgentDataset` + DAG config |

Floeval supports the following workflows:

- evaluating full datasets that already contain `llm_response`
- generating responses from partial datasets and scoring them in the same run
- expanding partial datasets across prompt variants with `prompt_ids` and `prompts_file`
- routing metrics across `ragas`, `deepeval`, `builtin`, and `custom`
- evaluating single-agent traces (pre-captured, Python callable, or FloTorch runner)
- evaluating multi-agent DAG workflows with `WorkflowRunner`
- capturing traces from Python callables, LangChain-style agents, or optional FloTorch runners

## Features

- **CLI and Python API**: run evaluations from config files or integrate directly into code
- **Five eval types**: LLM, RAG, Prompt (with and without RAG), Agent, and Agentic Workflow
- **Multi-provider metrics**: mix `ragas`, `deepeval`, `builtin`, and `custom` metrics in one evaluation
- **Prompt-aware generation**: compare system-prompt variants at scale with `prompt_ids` and `prompts_file`
- **Agent evaluation**: score pre-captured traces or collect traces at runtime
- **Agentic workflow evaluation**: evaluate multi-agent DAG pipelines with `WorkflowRunner`
- **Custom metrics**: define function-based metrics or LLM-as-judge criteria
- **Dataset format flexibility**: accepts `{"samples": [...]}`, JSON array, or JSONL; field aliases `question`/`answer` supported

## Installation

Version `0.2.0b1` is a pre-release. Installing from PyPI may require `--pre`:

```bash
pip install --pre floeval
```

Optional FloTorch support for agent Mode 4 and agentic workflow evaluation:

```bash
pip install "floeval[flotorch]"
```

Development install:

```bash
pip install -e .
pip install -e .[dev]
```

## Quick Start

### Python API — LLM / RAG evaluation

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

### Python API — Prompt evaluation (multi-prompt)

```python
from floeval import Evaluation, DatasetLoader
from floeval.config.schemas.io.llm import OpenAIProviderConfig

llm_config = OpenAIProviderConfig(
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
    chat_model="gpt-4o-mini",
    embedding_model="text-embedding-3-small",
)

partial_dataset = DatasetLoader.from_samples(
    [
        {
            "user_input": "Summarize this customer ticket.",
            "prompt_ids": ["concise", "detailed"]
        }
    ],
    partial_dataset=True,
)

evaluation = Evaluation(
    dataset=partial_dataset,
    llm_config=llm_config,
    metrics=["answer_relevancy"],
    default_provider="ragas",
    dataset_generator_model="gpt-4o-mini",
    prompts_file="prompts.yaml",
)

results = evaluation.run()
for row in results.sample_results:
    print(row["prompt_id"], row["metrics"]["answer_relevancy"]["score"])
```

### Python API — Agent evaluation

```python
from floeval.api.agent_evaluation import AgentEvaluation
from floeval.config.schemas.io.agent_dataset import AgentDataset, PartialAgentSample
from floeval.config.schemas.io.llm import OpenAIProviderConfig
from floeval.utils.agent_trace import capture_trace, log_turn

llm_config = OpenAIProviderConfig(
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
    chat_model="gpt-4o-mini",
)

@capture_trace
def my_agent(user_input: str) -> str:
    response = f"Handled: {user_input}"
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

evaluation = AgentEvaluation(
    dataset=dataset,
    agent=my_agent,
    llm_config=llm_config,
    metrics=["goal_achievement"],
)

results = evaluation.run()
print(results.summary)
```

### Python API — Agentic Workflow evaluation

```python
import json
from floeval.api.agent_evaluation import AgentEvaluation
from floeval.config.schemas.io.agent_dataset import AgentDataset, PartialAgentSample
from floeval.config.schemas.io.llm import OpenAIProviderConfig
from floeval.flotorch import WorkflowRunner  # requires floeval[flotorch]

llm_config = OpenAIProviderConfig(
    base_url="https://gateway.example/openai/v1",
    api_key="your-gateway-key",
    chat_model="gpt-4o-mini",
)

dag_config = json.loads(open("workflow_config.json").read())
runner = WorkflowRunner(dag_config=dag_config, llm_config=llm_config)

dataset = AgentDataset(
    samples=[
        PartialAgentSample(
            user_input="What is the status of order #12345?",
            reference_outcome="The order is shipped and arriving tomorrow.",
        )
    ]
)

evaluation = AgentEvaluation(
    dataset=dataset,
    agent_runner=runner,
    llm_config=llm_config,
    metrics=["goal_achievement", "ragas:agent_goal_accuracy"],
)

results = evaluation.run()
print(results.summary)
```

### CLI

```bash
# Evaluate a full LLM/RAG dataset
floeval evaluate -c config.yaml -d dataset.json -o results.json

# Evaluate a partial dataset (generate + score in one run)
floeval evaluate -c config.yaml -d partial_dataset.json -o results.json

# Generate first, then evaluate later
floeval generate -c config.yaml -d partial_dataset.json -o complete.json
floeval evaluate -c config.yaml -d complete.json -o results.json

# Prompt evaluation with a prompts file
floeval evaluate -c prompt_config.yaml -d partial_dataset.json -o prompt_results.json

# Single-agent evaluation
floeval evaluate -c agent_config.yaml -d agent_dataset.json --agent -o agent_results.json

# Agentic workflow evaluation
floeval evaluate -c workflow_config.yaml -d agent_dataset.json --agent -o workflow_results.json
```

## Project Structure

- [api/](https://github.com/FloTorch/floeval/tree/dev/floeval/api) - public evaluation APIs and dataset loaders
- [core/execution/](https://github.com/FloTorch/floeval/tree/dev/floeval/core/execution) - response generation and execution internals
- [metric_providers/](https://github.com/FloTorch/floeval/tree/dev/floeval/metric_providers) - provider-specific metric implementations
- [config/schemas/](https://github.com/FloTorch/floeval/tree/dev/floeval/config/schemas) - config, dataset, and prompt schemas
- [cli/](https://github.com/FloTorch/floeval/tree/dev/floeval/cli) - command-line entry points
- [utils/](https://github.com/FloTorch/floeval/tree/dev/floeval/utils) - trace capture, loaders, and helper utilities
- [flotorch/](https://github.com/FloTorch/floeval/tree/dev/floeval/flotorch) - optional FloTorch integration (WorkflowRunner, FloTorchRunner)

## Documentation

Detailed docs live in `docs/`:

- [Setup & Prerequisites](docs/setup.md)
- [Examples](docs/examples.md)
- [Prompt Evaluation](docs/prompt-evaluation.md)
- [Agent Evaluation](docs/agent-evaluation.md)
- [Agentic Workflow](docs/agentic-workflow.md)
- [Agent Tracing](docs/agent-tracing.md)
- [Metrics](docs/metrics.md)
- [Custom Metrics](docs/custom-metrics.md)
- [API Reference](docs/api-reference.md)
- [Troubleshooting](docs/troubleshooting.md)

## License

MIT
