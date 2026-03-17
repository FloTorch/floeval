# Troubleshooting

Solutions to common installation, configuration, dataset, and runtime issues.

---

## Installation issues

### Python version is too old

Symptom:

- install errors mentioning `requires-python >=3.11`

Fix:

```bash
python --version
```

Upgrade to Python 3.11 or newer, then reinstall.

### Package install fails

Try upgrading pip first:

```bash
python -m pip install --upgrade pip
python -m pip install --pre floeval
```

If you are installing from the local repo:

```bash
python -m pip install -e .
```

---

## Config issues

### Authentication failed

Check that `llm_config.api_key` and `llm_config.base_url` are correct for your provider.

```yaml
llm_config:
  base_url: "https://api.openai.com/v1"
  api_key: "your-api-key"
  chat_model: "gpt-4o-mini"
  embedding_model: "text-embedding-3-small"
```

If you load credentials from environment variables, print or validate them before constructing `OpenAIProviderConfig`.

### Missing `llm_config`

Both `floeval evaluate` and `floeval generate` expect a top-level `llm_config` section.

### Partial dataset but no generator model

Symptom:

- `dataset_generator_model is required for partial datasets`
- `No generator_model in dataset_generation_config`

Fix:

```yaml
dataset_generation_config:
  generator_model: "gpt-4o-mini"
```

You can also provide `evaluation_config.dataset_generator_model`, but `dataset_generation_config.generator_model` is the clearest option.

### Agent CLI mode missing `agent_name`

Symptom:

- partial agent evaluation fails with a message that `agent_name` is required

Fix:

```yaml
evaluation_config:
  agent_name: "support-agent"
  metrics:
    - "goal_achievement"
```

This is required for CLI partial agent evaluation with `--agent`.

---

## Dataset issues

### `faithfulness` or retrieval metrics fail

Make sure your samples include the fields those metrics need:

- `contexts` for grounding and retrieval checks
- `ground_truth` for recall and precision style metrics where applicable

Minimal example:

```json
{
  "samples": [
    {
      "user_input": "What is RAG?",
      "llm_response": "RAG stands for Retrieval-Augmented Generation.",
      "contexts": ["RAG combines retrieval with generation."],
      "ground_truth": "Retrieval-Augmented Generation"
    }
  ]
}
```

### Standard dataset missing `llm_response`

If a standard dataset sample has no `llm_response`, treat it as a partial dataset and add generation config.

### Agent dataset role errors

In saved agent trace datasets, message roles must be:

- `human`
- `ai`
- `tool`

### Invalid JSON

Validate the file before running Floeval:

```bash
python -m json.tool dataset.json
```

---

## Generate command issues

### Missing output path

`floeval generate` always requires `-o` or `--output`:

```bash
floeval generate -c config.yaml -d partial.json -o complete.json
```

### Wrong output extension

`floeval generate` currently exports only:

- `.json`
- `.jsonl`

### Dataset already has `llm_response`

If the input dataset already contains responses, use `floeval evaluate` instead of `floeval generate`.

### Prompt expansion not happening

Check all of the following:

- samples include `prompt_ids`
- `evaluation_config.prompts_file` points to a real YAML or JSON prompt file
- prompt IDs used in the dataset exist in that prompt file

---

## Agent evaluation issues

### FloTorch import errors

Symptom:

- agent CLI or runner setup fails because FloTorch modules are missing

Fix:

```bash
python -m pip install "floeval[flotorch]"
```

### FloTorch credentials missing

Provide gateway credentials through either:

- `llm_config.base_url` and `llm_config.api_key`
- `FLOTORCH_BASE_URL` and `FLOTORCH_API_KEY`

### `tool_call_accuracy` returns an error

This metric needs `reference_tool_calls` in each agent sample.

Example:

```json
{
  "reference_tool_calls": [
    {
      "name": "search",
      "args": {"query": "reset password"}
    }
  ]
}
```

---

## Runtime issues

### Rate limits or slow runs

Try:

- using fewer samples while iterating
- using a smaller model such as `gpt-4o-mini`
- lowering generation fan-out with `dataset_generation_config.max_concurrency`

### Low scores unexpectedly

Check:

1. whether the model is answering the intended question
2. whether contexts are relevant and clean
3. whether thresholds are too strict for your stage of development
4. whether you selected the right provider and metric for the task

---

## Validation

### Verify the installation

```bash
floeval --version
```

### Verify a standard dataset loads

```python
from floeval import DatasetLoader

dataset = DatasetLoader.from_file("dataset.json", partial_dataset=False)
print(len(dataset.samples))
```

### Verify an agent dataset loads

```python
from floeval.config.schemas.io.agent_dataset import AgentDataset

dataset = AgentDataset.from_file("agent_dataset.json")
print(len(dataset.samples))
print(dataset.is_partial)
```

### Smoke-test your own files

```bash
floeval evaluate -c your_config.yaml -d your_dataset.json
```

Replace `your_config.yaml` and `your_dataset.json` with real files from your project.

---

---

## Prompt evaluation issues

### No responses generated for prompt variants

Symptom:

- evaluation completes but `prompt_id` is missing from results
- only one result row per sample instead of one per prompt

Check all of the following:

- samples in the dataset include `prompt_ids` (list of IDs, not a single string)
- `evaluation_config.prompts_file` is set and points to a real file
- prompt IDs in `prompt_ids` all exist in the prompts file
- `dataset_generation_config.generator_model` is set (required for partial datasets)

Minimal working dataset:

```json
{
  "samples": [
    {
      "user_input": "What is RAG?",
      "prompt_ids": ["concise", "detailed"]
    }
  ]
}
```

Minimal prompts file:

```yaml
prompts:
  concise:
    template: "Answer in one sentence."
  detailed:
    template: "Answer in detail."
```

### RAG metrics fail in prompt evaluation

Prompt (with RAG) needs either:

- `contexts` in each sample, or
- a configured `vectorstore_id` (passed through `data.vectorstore_id` in the worker config)

If neither is present, use no-RAG metrics only (`answer_relevancy`).

---

## Agentic workflow issues

### FloTorch extras required

Symptom:

- `ImportError` or runtime error mentioning `google-adk` or `FloTorch`

Fix:

```bash
pip install "floeval[flotorch]"
```

### Agent node not found

Symptom:

- workflow execution fails with a message about the agent not being found

Check:

- `agentName` in the DAG node matches the name of an agent deployed in your FloTorch gateway
- `llm_config.base_url` points to the correct gateway

### DAG never reaches the end

Symptom:

- workflow hangs or times out

Check:

- there are no cycles in the `edges` list
- every non-START node has at least one incoming edge
- the DAG has exactly one `END` node

### Empty `agent_traces` in results

Symptom:

- `agent_traces` is empty or missing from sample results

Check that the DAG config has at least one `AGENT` type node:

```yaml
nodes:
  - {id: "start",   type: "START", label: "Start"}
  - {id: "agent_a", type: "AGENT", label: "Agent A", agentName: "my-agent:latest"}
  - {id: "end",     type: "END",   label: "End"}
```

A DAG with only `START` and `END` nodes will produce no agent traces.

### `tool_call_accuracy` is null for all workflow samples

This metric silently skips samples that have no `reference_tool_calls`. Add expected tool calls to your dataset:

```json
{
  "user_input": "What is the order status?",
  "reference_outcome": "The order is shipped.",
  "reference_tool_calls": [
    {"name": "get_order_status", "args": {"order_id": "12345"}}
  ]
}
```

---

## Related references

- [Examples](examples.md)
- [Prompt Evaluation](prompt-evaluation.md)
- [Agent Evaluation](agent-evaluation.md)
- [Agentic Workflow](agentic-workflow.md)
- [Metrics](metrics.md)
- [API Reference](api-reference.md)
