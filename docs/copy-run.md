# Minimal Examples

Use these minimal examples as starting points for common workflows.

!!! note "Prerequisites"
    Examples 2 and 7 do not need an API key. The others use `llm_config`, so replace the placeholder key before running them.

---

## Example 1: RAGAS evaluation

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
    default_provider="ragas",
    metrics=["answer_relevancy", "faithfulness"],
)

print(evaluation.run().aggregate_scores)
```

---

## Example 2: Simple custom metric

```python
from floeval import Evaluation, DatasetLoader
from floeval.api.metrics.custom import custom_metric


@custom_metric(threshold=0.3)
def response_length(response: str) -> float:
    return min(len(response) / 100.0, 1.0)


dataset = DatasetLoader.from_samples(
    [{"user_input": "Hi", "llm_response": "Hello there!"}],
    partial_dataset=False,
)

evaluation = Evaluation(dataset=dataset, metrics=["response_length"])
print(evaluation.run().aggregate_scores)
```

---

## Example 3: Mixed providers

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
            "user_input": "What is Python?",
            "llm_response": "Python is a programming language.",
            "contexts": ["Python is widely used."],
            "ground_truth": "A programming language",
        }
    ],
    partial_dataset=False,
)

evaluation = Evaluation(
    dataset=dataset,
    llm_config=llm_config,
    metrics=[
        "ragas:answer_relevancy",
        "deepeval:contextual_relevancy",
    ],
)

print(evaluation.run().aggregate_scores)
```

---

## Example 4: Partial dataset generation during evaluation

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
            "user_input": "What is RAG?",
            "contexts": ["RAG combines retrieval with generation."],
        }
    ],
    partial_dataset=True,
)

evaluation = Evaluation(
    dataset=partial_dataset,
    llm_config=llm_config,
    default_provider="ragas",
    metrics=["answer_relevancy", "faithfulness"],
    dataset_generator_model="gpt-4o-mini",
)

print(evaluation.run().aggregate_scores)
```

---

## Example 5: Prompt-driven partial generation

```yaml
# prompts.yaml
prompts:
  brief:
    template: "Answer in one sentence."
  detailed:
    template: "Answer in detail with two bullet points."
```

```json
{
  "samples": [
    {
      "user_input": "Summarize this ticket.",
      "prompt_ids": ["brief", "detailed"]
    }
  ]
}
```

```yaml
# config.yaml
llm_config:
  base_url: "https://api.openai.com/v1"
  api_key: "your-api-key"
  chat_model: "gpt-4o-mini"
  embedding_model: "text-embedding-3-small"

evaluation_config:
  metrics:
    - "answer_relevancy"
  prompts_file: "prompts.yaml"

dataset_generation_config:
  generator_model: "gpt-4o-mini"
```

```bash
floeval generate -c config.yaml -d partial_dataset.json -o generated.json
python -m json.tool generated.json
```

---

## Example 6: Agent evaluation with a Python callable

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

print(evaluation.run().summary)
```

---

---

## Example 7: Prompt evaluation — compare two system prompts

No RAG, no vectorstore. Uses two prompt variants against the same question. Does not need an API key if you swap out the LLM metrics for a custom one.

```yaml
# prompts.yaml
prompts:
  direct:
    template: "Answer the question directly and concisely."
  explanatory:
    template: "Answer the question with a brief explanation of your reasoning."
```

```json
{
  "samples": [
    {
      "user_input": "What is the capital of France?",
      "prompt_ids": ["direct", "explanatory"]
    }
  ]
}
```

```yaml
# config.yaml
llm_config:
  base_url: "https://api.openai.com/v1"
  api_key: "your-api-key"
  chat_model: "gpt-4o-mini"
  embedding_model: "text-embedding-3-small"

evaluation_config:
  metrics:
    - "ragas:answer_relevancy"
  prompts_file: "prompts.yaml"

dataset_generation_config:
  generator_model: "gpt-4o-mini"
```

```bash
floeval evaluate -c config.yaml -d partial_dataset.json -o results.json
python -m json.tool results.json
```

---

## Example 8: Agentic workflow (skeleton — needs FloTorch gateway)

Requires `pip install "floeval[flotorch]"` and access to a FloTorch gateway with two agents deployed.

```json
{
  "uid": "demo-workflow",
  "name": "Demo Workflow",
  "nodes": [
    {"id": "start",   "type": "START", "label": "Start"},
    {"id": "agent_a", "type": "AGENT", "label": "Agent A", "agentName": "my-agent-a:latest"},
    {"id": "end",     "type": "END",   "label": "End"}
  ],
  "edges": [
    {"sourceNodeId": "start",   "targetNodeId": "agent_a"},
    {"sourceNodeId": "agent_a", "targetNodeId": "end"}
  ]
}
```

```python
import json
from floeval.api.agent_evaluation import AgentEvaluation
from floeval.config.schemas.io.agent_dataset import AgentDataset, PartialAgentSample
from floeval.config.schemas.io.llm import OpenAIProviderConfig
from floeval.flotorch import WorkflowRunner

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

---

## Next steps

- [Examples](examples.md) for fuller workflows
- [Prompt Evaluation](prompt-evaluation.md) for prompt variant workflows
- [Agent Evaluation](agent-evaluation.md) for trace dataset formats
- [Agentic Workflow](agentic-workflow.md) for multi-agent DAG evaluation
- [Metrics](metrics.md) for the current metric catalog
