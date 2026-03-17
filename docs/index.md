# Floeval User Guide

Floeval is a multi-backend evaluation framework for LLM, RAG, prompt, and agent systems.

---

## Supported evaluation types

Floeval supports five distinct evaluation types. Each maps to a different dataset shape and set of metrics.

| Eval type | What you are scoring | Typical dataset |
|-----------|----------------------|-----------------|
| **LLM** | Direct question-answer quality without retrieval | `user_input` + `llm_response` |
| **RAG** | Answer quality and retrieval performance with context | `user_input` + `llm_response` + `contexts` |
| **Prompt** | One or more system prompts against the same dataset | Partial dataset + `prompts_file` or inline prompts |
| **Agent** | Single-agent trace quality, tool use, and goal achievement | `AgentDataset` with full or partial samples |
| **Agentic Workflow** | Multi-agent DAG: each agent node is a step in the pipeline | `AgentDataset` + DAG config |

---

## Supported workflows

| Workflow | Best for | Main entry point |
|----------|----------|------------------|
| Standard CLI evaluation | Evaluation from config and dataset files | `floeval evaluate` |
| Standard Python evaluation | App integration and programmatic control | `Evaluation(...)` |
| Prompt evaluation | Compare system-prompt variants at scale | `Evaluation(..., prompts_file=...)` |
| Agent evaluation | Scoring traces, tool use, and final outcomes | `AgentEvaluation(...)` or `floeval evaluate --agent` |
| Agentic workflow evaluation | Multi-agent DAG scoring | `AgentEvaluation(..., agent_runner=WorkflowRunner(...))` |
| Trace capture | Turning agent runs into evaluation-ready traces | `floeval.utils.agent_trace` |

---

## Core capabilities

- Run `ragas`, `deepeval`, `builtin`, and `custom` metrics in the same project
- Evaluate full datasets that already contain `llm_response`
- Evaluate partial datasets and let Floeval generate missing responses automatically
- Expand partial datasets across prompt variants with `prompt_ids` and `prompts_file`
- Score agent traces with agent-specific metrics (goal achievement, tool accuracy, coherence)
- Run multi-agent DAG workflows and evaluate the combined output
- Capture traces from Python callables, LangChain-style agents, or FloTorch runners
- Define custom scoring logic using function decorators or LLM-as-judge criteria

---

## Metrics by eval type (quick reference)

| Eval type | Recommended metrics |
|-----------|---------------------|
| `llm` | `ragas:answer_relevancy`, `deepeval:answer_relevancy` |
| `rag` | `ragas:faithfulness`, `ragas:context_precision`, `ragas:context_recall`, `deepeval:contextual_precision`, `deepeval:contextual_relevancy` (plus others) |
| `prompt (no RAG)` | `ragas:answer_relevancy`, `deepeval:answer_relevancy` |
| `prompt (with RAG)` | Same as `rag` |
| `agent` | `builtin:goal_achievement`, `builtin:response_coherence`, `ragas:agent_goal_accuracy`, `ragas:tool_call_accuracy` |
| `agentic_workflow` | Same 4 agent metrics |

See [Metrics](metrics.md) for the complete catalog.

---

## Documentation map

| Task | Reference |
|------|-----------|
| Install and set up Floeval | [Prerequisites & Setup](setup.md) |
| Run a quick example | [Minimal Examples](copy-run.md) |
| Run evaluation from files | [Examples](examples.md#cli-workflows) |
| Evaluate with system prompts | [Prompt Evaluation](prompt-evaluation.md) |
| Evaluate an agent trace dataset | [Agent Evaluation](agent-evaluation.md) |
| Evaluate a multi-agent DAG workflow | [Agentic Workflow](agentic-workflow.md) |
| Add trace capture to an agent | [Agent Tracing](agent-tracing.md) |
| Compare providers or metric IDs | [Metrics](metrics.md) |
| Define custom scoring logic | [Custom Metrics](custom-metrics.md) |
| Check config fields and constructors | [API Reference](api-reference.md) |
| Diagnose errors and failures | [Troubleshooting](troubleshooting.md) |

---

## Basic workflow

1. Install Floeval. See [Prerequisites & Setup](setup.md) for version and environment details.
2. Create a config file with `llm_config` and `evaluation_config`.
3. Create a dataset file in `.json` or `.jsonl`.
4. Run `floeval evaluate -c config.yaml -d dataset.json`.

!!! note "Credential handling"
    Examples use placeholder API keys for clarity. In real usage, prefer loading secrets from environment variables or a secrets manager and injecting them into your config or Python code at runtime.

---

## References

| Section | What it covers |
|---------|----------------|
| [Prerequisites & Setup](setup.md) | Installation, beta versioning, credentials, optional FloTorch setup |
| [Minimal Examples](copy-run.md) | Short copy-paste examples for all eval types |
| [Examples](examples.md) | CLI, Python, prompt expansion, mixed providers, agent, and workflow flows |
| [Prompt Evaluation](prompt-evaluation.md) | Single and multi-prompt eval, `prompts_file`, `prompt_ids`, RAG vs no-RAG |
| [Agent Evaluation](agent-evaluation.md) | Single-agent dataset shapes, CLI `--agent`, Python callable, FloTorch runner |
| [Agentic Workflow](agentic-workflow.md) | Multi-agent DAG config, `WorkflowRunner`, traces per node |
| [Agent Tracing](agent-tracing.md) | `capture_trace`, `log_turn`, `log_tool_result`, `wrap_langchain_agent` |
| [Metrics](metrics.md) | Full metric catalog by provider and eval type |
| [Custom Metrics](custom-metrics.md) | `@custom_metric` decorator, `criteria()` LLM-as-judge |
| [API Reference](api-reference.md) | Config keys, dataset models, CLI signatures, and public APIs |
| [Troubleshooting](troubleshooting.md) | Install, config, dataset, generation, prompt, and agent-eval fixes |
