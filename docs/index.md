# Floeval User Guide

Floeval supports evaluation workflows for LLM, RAG, and agent systems.

---

## Supported workflows

Floeval supports the following workflows:

| Workflow | Best for | Main entry point |
|----------|----------|------------------|
| Standard CLI evaluation | Evaluation from config and dataset files | `floeval evaluate` |
| Standard Python evaluation | App integration and programmatic control | `Evaluation(...)` |
| Agent evaluation | Scoring traces, tool use, and final outcomes | `AgentEvaluation(...)` or `floeval evaluate --agent` |
| Trace capture | Turning agent runs into evaluation-ready traces | `floeval.utils.agent_trace` |

---

## Core capabilities

- Run `ragas`, `deepeval`, builtin, and custom metrics in the same project
- Evaluate full datasets that already contain `llm_response`
- Evaluate partial datasets and let Floeval generate missing responses automatically
- Expand partial datasets across prompt variants with `prompt_ids` and `prompts_file`
- Score agent traces with agent-specific metrics
- Capture traces from Python callables, LangChain-style agents, or optional FloTorch runners

---

## Documentation map

| Task | Reference |
|------|-----------|
| Run evaluation from files | [Examples](examples.md#cli-workflows) |
| Compare providers or metric IDs | [Metrics](metrics.md) |
| Define custom scoring logic | [Custom Metrics](custom-metrics.md) |
| Evaluate an agent trace dataset | [Agent Evaluation](agent-evaluation.md) |
| Add trace capture to an agent | [Agent Tracing](agent-tracing.md) |
| Check config fields and constructors | [API Reference](api-reference.md) |

---

## Basic workflow

1. Install Floeval. See [Prerequisites & Setup](setup.md) for version and environment details.
2. Create a config file with `llm_config` and `evaluation_config`
3. Create a dataset file in `.json` or `.jsonl`
4. Run `floeval evaluate -c config.yaml -d dataset.json`

!!! note "Credential handling"
    Examples use placeholder API keys for clarity. In real usage, prefer loading secrets from environment variables or a secrets manager and injecting them into your config or Python code at runtime.

---

## References

| Section | What it covers |
|---------|----------------|
| [Prerequisites & Setup](setup.md) | installation, beta versioning, credentials, optional FloTorch setup |
| [Minimal Examples](copy-run.md) | short copy-paste examples |
| [Examples](examples.md) | CLI, Python, prompt expansion, mixed providers, and agent flows |
| [Agent Evaluation](agent-evaluation.md) | dataset shapes, CLI `--agent`, Python callable mode, FloTorch mode |
| [Agent Tracing](agent-tracing.md) | `capture_trace`, `log_turn`, `log_tool_result`, `wrap_langchain_agent` |
| [Metrics](metrics.md) | the current metric catalog by provider |
| [API Reference](api-reference.md) | config keys, dataset models, CLI signatures, and public APIs |
| [Troubleshooting](troubleshooting.md) | install, config, dataset, generation, and agent-eval fixes |
