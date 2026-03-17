# Prerequisites & Setup

Complete these steps once before using Floeval.

---

## Prerequisites

### Python

Floeval requires Python 3.11 or newer.

Check your version:

```bash
python --version
```

### API access

Most built-in provider-backed metrics require an OpenAI or OpenAI-compatible endpoint plus an API key.

Examples in this docs site use placeholder keys. In real projects, prefer loading secrets from environment variables or a secrets manager and injecting them into your config or Python code at runtime.

### Optional FloTorch support

Install the `flotorch` extra only if you need Mode 4 agent evaluation or agentic workflow evaluation:

```bash
pip install "floeval[flotorch]"
```

---

## Install Floeval

Package version `0.2.0b1` is a pre-release, so installation from PyPI may require `--pre`:

```bash
pip install --pre floeval
```

From local source:

```bash
pip install -e .
```

Development dependencies:

```bash
pip install -e .[dev]
```

### Verify installation

```bash
floeval --version
```

You should see a beta version such as `floeval 0.2.0b1`.

---

## Recommended setup commands

### Linux or macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --pre floeval
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --pre floeval
```

---

## Provide credentials

Floeval accepts credentials through the `llm_config` object you pass in a config file or Python code.

### CLI config example

```yaml
llm_config:
  base_url: "https://api.openai.com/v1"
  api_key: "your-api-key"
  chat_model: "gpt-4o-mini"
  chat_endpoint: "chat/completions"
  embedding_model: "text-embedding-3-small"
  embedding_endpoint: "embeddings"
```

### Python example

```python
import os

from floeval.config.schemas.io.llm import OpenAIProviderConfig

llm_config = OpenAIProviderConfig(
    base_url="https://api.openai.com/v1",
    api_key=os.environ["OPENAI_API_KEY"],
    chat_model="gpt-4o-mini",
    embedding_model="text-embedding-3-small",
    system_prompt="You are a helpful assistant.",  # optional
)
```

### Agent evaluation with FloTorch

When you use Mode 4 agent evaluation, FloTorch can also read:

- `FLOTORCH_BASE_URL`
- `FLOTORCH_API_KEY`

If you already pass `llm_config` with the gateway base URL and key, you may not need separate FloTorch environment variables.

---

## Next steps

- [Examples](examples.md) for CLI, Python, prompt expansion, and agent workflows
- [Agent Evaluation](agent-evaluation.md) for trace-based scoring
- [API Reference](api-reference.md) for the full config surface
