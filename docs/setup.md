# Prerequisites & Setup

What you need before starting. Complete these steps once.

---

## Checklist

### Python 3.11 or higher

Check: `python --version`

### LLM API key (OpenAI or compatible)

Get from OpenAI Platform or your provider. You pass it in your **config file** or **Python script**—not via environment variables by default.

### Install Floeval

```bash
pip install floeval
```

Or from source:

```bash
pip install -e .
```

### Verify installation

```bash
floeval --version
```

Should show `floeval 0.1.0`.

---

## How to provide credentials

### Option 1: Config file (CLI)

Put your API key and endpoint in the config file:

```json
{
  "llm_config": {
    "base_url": "https://api.openai.com/v1",
    "api_key": "sk-your-api-key-here",
    "chat_model": "gpt-4o-mini",
    "embedding_model": "text-embedding-3-small"
  },
  "evaluation_config": { ... }
}
```

### Option 2: Python script

Build an `OpenAIProviderConfig` from a dict:

```python
from floeval.config.schemas.io.llm import OpenAIProviderConfig

LLM_CONFIG = {
    "base_url": "https://api.openai.com/v1",
    "api_key": "sk-your-api-key-here",
    "chat_model": "gpt-4o-mini",
    "embedding_model": "text-embedding-3-small",
    "system_prompt": "You are a helpful assistant."  # optional
}

llm_config = OpenAIProviderConfig(**LLM_CONFIG)
```

You can load the dict from env vars, a secrets file, or any source—Floeval just needs the dict structure.

---

## Setup commands

### Linux / macOS

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install
pip install floeval
```

### Windows

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# Install
pip install floeval
```
