# Agentic Workflow Evaluation

Use this guide to evaluate multi-agent pipelines where multiple agents collaborate in a directed acyclic graph (DAG). Each node in the DAG is an independent agent, and the nodes execute in topological order with dependencies resolved automatically.

---

## When to use agentic workflow vs single-agent evaluation

| Use case | Evaluation type |
|----------|----------------|
| One agent answers questions | [Agent Evaluation](agent-evaluation.md) |
| Multiple agents collaborate in sequence or parallel | **Agentic Workflow** |
| Each agent has a specific role (retrieval, reasoning, formatting) | **Agentic Workflow** |

---

## Prerequisites

Agentic workflow evaluation requires the optional FloTorch integration:

```bash
pip install "floeval[flotorch]"
```

This installs `google-adk` and related packages needed by `WorkflowRunner`.

---

## How it works

```
User input
    │
    ▼
 DAG START node
    │
    ▼
 AGENT node A ──► AGENT node B
                      │
                      ▼
                 DAG END node
                      │
                      ▼
              final_response
```

1. Floeval loads the DAG config defining nodes and edges.
2. For each sample in the dataset, `WorkflowRunner` executes the DAG.
3. Each agent node runs in order; its output feeds into downstream nodes.
4. After all nodes complete, the final output is captured as an `AgentSample` with:
   - `trace` — the combined last-node trace
   - `agent_traces` — one `AgentTrace` per completed agent node
   - `metadata.workflow_execution` — per-agent summaries
5. `AgentEvaluation` scores each completed `AgentSample` using the configured metrics.

---

## DAG config structure

The DAG is defined as a JSON object with `uid`, `name`, `nodes`, and `edges`.

```json
{
  "uid": "eval-workflow-001",
  "name": "Support Triage Workflow",
  "nodes": [
    {"id": "start",     "type": "START",  "label": "Start"},
    {"id": "classify",  "type": "AGENT",  "label": "Classify",  "agentName": "classifier-agent:latest"},
    {"id": "respond",   "type": "AGENT",  "label": "Respond",   "agentName": "responder-agent:latest"},
    {"id": "end",       "type": "END",    "label": "End"}
  ],
  "edges": [
    {"sourceNodeId": "start",    "targetNodeId": "classify"},
    {"sourceNodeId": "classify", "targetNodeId": "respond"},
    {"sourceNodeId": "respond",  "targetNodeId": "end"}
  ]
}
```

### Node types

| Type | Description |
|------|-------------|
| `START` | Entry point; receives the user input |
| `AGENT` | An agent that processes the incoming data; must have `agentName` |
| `END` | Marks the end of the workflow |

### Parallel nodes

To run two agents in parallel, give them both the same upstream node as a parent:

```json
{
  "edges": [
    {"sourceNodeId": "start",     "targetNodeId": "retriever"},
    {"sourceNodeId": "start",     "targetNodeId": "classifier"},
    {"sourceNodeId": "retriever", "targetNodeId": "synthesizer"},
    {"sourceNodeId": "classifier","targetNodeId": "synthesizer"},
    {"sourceNodeId": "synthesizer","targetNodeId": "end"}
  ]
}
```

`retriever` and `classifier` run in parallel; `synthesizer` runs after both complete.

---

## CLI workflow

### Config

```yaml
# workflow_config.yaml
llm_config:
  base_url: "https://gateway.example/openai/v1"
  api_key: "your-gateway-key"
  chat_model: "gpt-4o-mini"

evaluation_config:
  metrics:
    - "goal_achievement"
    - "ragas:agent_goal_accuracy"

agent_workflow_config:
  dataset_url: "https://your-storage/agent_dataset.json"
  config:
    uid: "eval-workflow-001"
    name: "Support Triage Workflow"
    nodes:
      - {id: "start",    type: "START", label: "Start"}
      - {id: "classify", type: "AGENT", label: "Classify", agentName: "classifier-agent:latest"}
      - {id: "respond",  type: "AGENT", label: "Respond",  agentName: "responder-agent:latest"}
      - {id: "end",      type: "END",   label: "End"}
    edges:
      - {sourceNodeId: "start",    targetNodeId: "classify"}
      - {sourceNodeId: "classify", targetNodeId: "respond"}
      - {sourceNodeId: "respond",  targetNodeId: "end"}
```

### Command

```bash
floeval evaluate -c workflow_config.yaml -d agent_dataset.json --agent -o workflow_results.json
```

---

## Dataset

Use a standard agent dataset. Partial samples (no trace) work best because Floeval will execute the workflow and fill in the traces.

### Partial agent dataset (recommended)

```json
{
  "samples": [
    {
      "user_input": "My order has not arrived after two weeks.",
      "reference_outcome": "An apology and a case escalation to the shipping team."
    },
    {
      "user_input": "How do I reset my password?",
      "reference_outcome": "Step-by-step password reset instructions."
    }
  ]
}
```

Dataset field aliases are supported: `question` maps to `user_input`, and `answer` maps to `reference_outcome`.

### JSON array format

The dataset can also be a JSON array at the root level (no `samples` wrapper):

```json
[
  {"question": "My order has not arrived.", "answer": "An apology and escalation."},
  {"question": "How do I reset my password?", "answer": "Step-by-step instructions."}
]
```

### JSONL format

```jsonl
{"user_input": "My order has not arrived.", "reference_outcome": "An apology and escalation."}
{"user_input": "How do I reset my password?", "reference_outcome": "Step-by-step instructions."}
```

---

## Python API

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

runner = WorkflowRunner(
    dag_config=dag_config,
    llm_config=llm_config,
    app_name="floeval-workflow",
)

dataset = AgentDataset(
    samples=[
        PartialAgentSample(
            user_input="My order has not arrived after two weeks.",
            reference_outcome="An apology and a case escalation to the shipping team.",
        ),
        PartialAgentSample(
            user_input="How do I reset my password?",
            reference_outcome="Step-by-step password reset instructions.",
        ),
    ]
)

evaluation = AgentEvaluation(
    dataset=dataset,
    agent_runner=runner,
    llm_config=llm_config,
    metrics=[
        "goal_achievement",
        "response_coherence",
        "ragas:agent_goal_accuracy",
    ],
)

results = evaluation.run()
print(results.summary)
```

### Load dataset from file

```python
dataset = AgentDataset.from_file("agent_dataset.json")
```

### Async execution

```python
results = await evaluation.arun()
```

---

## `WorkflowRunner` reference

```python
from floeval.flotorch import WorkflowRunner

runner = WorkflowRunner(
    dag_config=dag_config,       # dict: the DAG JSON object
    llm_config=llm_config,       # OpenAIProviderConfig
    app_name="floeval-workflow", # optional label shown in logs
)

# Run a single sample
full_samples = await runner.run_on_dataset([partial_sample])
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `dag_config` | Yes | DAG definition dict with `uid`, `name`, `nodes`, `edges` |
| `llm_config` | Yes | LLM credentials for each agent in the DAG |
| `app_name` | No | Log label; defaults to `"floeval-workflow"` |

---

## Metrics

Agentic workflow evaluation supports all four agent metrics:

| Provider | Metric | Notes |
|----------|--------|-------|
| `builtin` | `goal_achievement` | LLM judge; scores whether the workflow's final output satisfies the user request |
| `builtin` | `response_coherence` | Checks whether the final response is coherent with the full multi-agent trace |
| `ragas` | `agent_goal_accuracy` | Compares the workflow output against `reference_outcome` |
| `ragas` | `tool_call_accuracy` | Checks tool calls against `reference_tool_calls`; skips samples that lack `reference_tool_calls` |

---

## Results

```python
results = evaluation.run()

# Per-sample results
for row in results.sample_results:
    print(row["user_input"])
    print(row["final_response"])
    print(row["metrics"])
    # Each metric: {score, passed, reason, metadata}

    # Agent traces per node
    agent_traces = row.get("agent_traces", [])
    for trace in agent_traces:
        print(trace)

    # Per-agent summaries
    workflow_exec = row.get("workflow_execution", {})

# Aggregate summary (mean scores per metric)
print(results.summary)
```

The `agent_traces` list in each sample contains one `AgentTrace` per completed AGENT node, in execution order. The `workflow_execution` dict contains per-agent `input`, `output`, `tool_calls`, `turn_count`, and `tool_call_count`.

---

## Common pitfalls

| Problem | Cause | Fix |
|---------|-------|-----|
| `FloTorch extras required` error | `floeval[flotorch]` not installed | `pip install "floeval[flotorch]"` |
| Agent not found | `agentName` in DAG node does not exist in the gateway | Check the agent name and your gateway connection |
| Workflow hangs | Circular edge in DAG | Check the edges list for cycles |
| Empty `agent_traces` | DAG has no AGENT nodes | Ensure at least one `AGENT` type node is in `nodes` |
| `tool_call_accuracy` is null | Samples have no `reference_tool_calls` | Either add `reference_tool_calls` to samples or omit this metric |

---

## Related references

- [Agent Evaluation](agent-evaluation.md) for single-agent evaluation
- [Agent Tracing](agent-tracing.md) for trace capture when building your own agent runner
- [Metrics](metrics.md) for the full metric catalog
- [API Reference](api-reference.md) for `WorkflowRunner`, `AgentEvaluation`, and config shapes
