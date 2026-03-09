# Agent Tracing

Capture agent conversations and tool activity so they can be evaluated with `AgentEvaluation`.

---

## Trace contents

Floeval's tracing helpers turn agent execution into an `AgentTrace` with:

- ordered messages
- optional tool calls and tool results
- a final user-facing response
- optional metadata collected during execution

The resulting trace can then be:

- saved as a full agent dataset and scored later
- collected on the fly from a Python callable
- wrapped around LangChain or LangGraph style agents

---

## Public API

Import tracing helpers from `floeval.utils.agent_trace`:

```python
from floeval.utils.agent_trace import (
    capture_trace,
    create_span,
    log_tool_result,
    log_turn,
    wrap_langchain_agent,
)
```

### Core helpers

| Helper | Purpose |
|--------|---------|
| `capture_trace` | Decorate an agent function so Floeval creates a trace context automatically |
| `log_turn` | Record an AI response turn inside the active trace |
| `log_tool_result` | Record tool output inside the active trace |
| `create_span` | Alternate helper that records an output turn in a DeepEval-style shape |
| `wrap_langchain_agent` | Adapt a LangChain or LangGraph agent to Floeval's callable interface |

For advanced control, `TraceCollector` is also available.

---

## Basic Python callable

This pattern is suitable for Mode 2 agent evaluation.

```python
from floeval.utils.agent_trace import capture_trace, log_tool_result, log_turn


@capture_trace
def weather_agent(user_input: str) -> str:
    tool_output = "Paris: 18C, light rain"
    log_tool_result("weather_lookup", tool_output)

    response = f"Current weather for Paris: {tool_output}"
    log_turn(response)
    return response
```

`capture_trace` creates the trace context, `log_tool_result(...)` records a tool message, `log_turn(...)` records the agent response, and the returned string becomes the trace's final response.

You can pass this function directly into `AgentEvaluation(agent=...)`.

---

## Async agent

`capture_trace` also supports async agents:

```python
from floeval.utils.agent_trace import capture_trace, log_turn


@capture_trace
async def async_support_agent(user_input: str) -> str:
    response = f"Handled asynchronously: {user_input}"
    log_turn(response)
    return response
```

---

## LangChain or LangGraph agents

If your agent expects a messages-style interface, wrap it before evaluation:

```python
from floeval.api.agent_evaluation import AgentEvaluation
from floeval.config.schemas.io.agent_dataset import AgentDataset
from floeval.utils.agent_trace import wrap_langchain_agent

wrapped_agent = wrap_langchain_agent(langchain_agent)

evaluation = AgentEvaluation(
    dataset=AgentDataset.from_file("partial_agent_dataset.json"),
    agent=wrapped_agent,
    llm_config=llm_config,
    metrics=["goal_achievement"],
)
```

The wrapper adapts:

- string input into the `{"messages": [...]}` shape many agent frameworks expect
- agent output back into a string result that Floeval can trace

`TraceCollector` also tries to attach LangChain callbacks automatically when the wrapped agent exposes `invoke(...)` or `ainvoke(...)`.

---

## Returning `AgentTrace` directly

If your agent already produces a full trace, you can return `AgentTrace` yourself instead of relying on helper logging.

```python
from floeval.config.schemas.io.agent_dataset import AgentTrace


def simple_agent(user_input: str) -> AgentTrace:
    return AgentTrace.from_simple_response(
        user_input=user_input,
        response="Done.",
        source="custom-runtime",
    )
```

This works with `TraceCollector` and `AgentEvaluation` as long as the trace is valid.

---

## Evaluation workflows

Two common workflows are supported:

### Collect traces during evaluation

Use a partial agent dataset and pass `agent=` to `AgentEvaluation`. Floeval runs the agent, captures traces, and evaluates the resulting full samples.

### Capture first, score later

Run your agent separately, persist the resulting traces into an agent dataset file, then evaluate that file later with Mode 1.

This approach is useful when:

- trace capture is expensive
- you want reproducible evaluations
- you want to inspect traces manually before scoring

---

## Trace shape

Full trace datasets use this structure:

```json
{
  "trace": {
    "messages": [
      {"role": "human", "content": "User request"},
      {
        "role": "ai",
        "content": "I'll look that up.",
        "tool_calls": [{"name": "search", "args": {"q": "User request"}}]
      },
      {
        "role": "tool",
        "tool_name": "search",
        "content": "Tool result"
      },
      {"role": "ai", "content": "Final answer"}
    ],
    "final_response": "Final answer"
  }
}
```

Roles in saved datasets should be:

- `human`
- `ai`
- `tool`

---

## Notes

- Prefer `@capture_trace` for custom Python agents. It is the simplest path.
- Use `log_turn()` only inside an active trace context.
- Use `reference_tool_calls` in your dataset when you want to score `ragas:tool_call_accuracy`.
- Use `reference_outcome` when you want stronger outcome-based scoring such as `goal_achievement` or `ragas:agent_goal_accuracy`.
- If your agent already exposes a stable trace format, returning `AgentTrace` directly can be cleaner than manual logging.

---

## Related references

- [Agent Evaluation](agent-evaluation.md)
- [API Reference](api-reference.md)
- [Examples](examples.md)
