# Troubleshooting

Quick fixes for common issues.

---

## Installation Issues

### "requires-python = '>=3.11'"

**Problem**: Python version too old

**Fix**:
```bash
# Check version
python --version

# Need Python 3.11 or higher
# Download from python.org or use:
# Ubuntu: sudo apt install python3.11
# macOS: brew install python@3.11
```

### "pip install floeval" fails

**Fix**:
```bash
# Update pip first
pip install --upgrade pip

# Then install
pip install floeval
```

---

## Configuration Issues

### "Authentication failed"

**Problem**: API key not set or wrong

**Fix**: Ensure your config file has the correct `api_key` in `llm_config`:

```json
{
  "llm_config": {
    "base_url": "https://api.openai.com/v1",
    "api_key": "sk-your-key-here",
    "chat_model": "gpt-4o-mini",
    "embedding_model": "text-embedding-3-small"
  },
  ...
}
```

Or in Python, pass the key in your `LLM_CONFIG` dict when creating `OpenAIProviderConfig`.

### "Missing 'llm_config' section"

**Problem**: Config file has wrong structure

**Fix**: Use `llm_config` (not `gateway_config`). Required keys: `base_url`, `api_key`, `chat_model`, `embedding_model`.

### "dataset_generator_model is required for partial datasets"

**Problem**: Using a partial dataset (samples without `llm_response`) but config lacks generator model.

**Fix**: Add `dataset_generation_config` to your config:

```json
{
  "dataset_generation_config": {
    "generator_model": "gpt-4o-mini"
  }
}
```

---

## Dataset Issues

### "Field required: contexts"

**Problem**: Using `faithfulness` metric without contexts

**Fix Option 1** - Add contexts to dataset:
```json
{
  "user_input": "Question?",
  "llm_response": "Answer.",
  "contexts": ["Your context here"]
}
```

**Fix Option 2** - Remove faithfulness metric:
```yaml
evaluation_config:
  metrics:
    - answer_relevancy  # Only this
```

### "Field required: llm_response"

**Problem**: Using full-dataset path with samples that lack `llm_response`

**Fix**: Either ensure every sample has `llm_response` (full dataset), or use a partial dataset and add `dataset_generation_config` so the CLI generates responses.

### "Invalid JSON"

**Problem**: JSON syntax error

**Fix**:
```bash
# Validate JSON using Python
python -c "import json; json.load(open('dataset.json'))"
```

---

## Generate Command Issues

### "floeval generate: error: the following arguments are required: -o, --output"

**Problem**: Forgot to specify output file for `floeval generate`

**Fix**: The `generate` command requires an output file to save generated responses:

```bash
floeval generate -c config.yaml -d partial.json -o complete.json
```

### "No generator_model in dataset_generation_config"

**Problem**: Config missing `dataset_generation_config` for generate command

**Fix**: Add to your config:

```json
{
  "dataset_generation_config": {
    "generator_model": "gpt-4o-mini"
  }
}
```

### "Input dataset has llm_response but shouldn't"

**Problem**: Using `floeval generate` on a full dataset (that already has responses)

**Fix**: Use `floeval evaluate` instead, or create a partial dataset without `llm_response`:

```bash
# Full dataset → evaluate
floeval evaluate -c config.yaml -d full_dataset.json -o results.json

# Partial dataset → generate
floeval generate -c config.yaml -d partial_dataset.json -o complete.json
```

### Generated dataset has incorrect format

**Problem**: Output from `floeval generate` doesn't match expected format

**Fix**: Check that input partial dataset is valid:

```bash
# Validate input
python -c "import json; d = json.load(open('partial.json')); print('Samples:', len(d.get('samples', [])))"

# Generate
floeval generate -c config.yaml -d partial.json -o complete.json

# Validate output
python -c "import json; d = json.load(open('complete.json')); s = d['samples'][0]; print('Has response:', 'llm_response' in s)"
```

---

## Runtime Issues

### "Rate limit exceeded"

**Problem**: Too many API calls too fast

**Fix**:
- Wait 30 seconds and retry
- Use fewer samples for testing
- Check OpenAI account has credits

### Evaluation takes too long

**Fix**:
- Start with 2-3 samples to test
- Use `gpt-4o-mini` instead of `gpt-4`
- Check internet connection

### Low scores unexpectedly

**Check**:
1. Is your LLM answering the right question?
2. Are contexts relevant to the question?
3. Is threshold set too high?

---

## Common Mistakes

### API key in config

You can pass your API key in the config file or in your Python script. If using a config file, ensure `llm_config.api_key` is set. Avoid committing real keys to version control—use environment variables or a secrets manager to populate the config at runtime if preferred.

### Missing virtual environment

❌ **Don't**:
```bash
pip install floeval  # Global install
```

✅ **Do**:
```bash
python -m venv venv
source venv/bin/activate
pip install floeval
```

### Testing with too many samples

❌ **Don't**:
```json
{
  "samples": [/* 1000 samples */]
}
```

✅ **Do**:
```json
{
  "samples": [/* Start with 2-3 samples */]
}
```

---

## Quick Checks

### Verify Installation

```bash
floeval --version
```

### Verify Config

```bash
floeval evaluate -c sample_data/eval_config.json -d sample_data/full_dataset.json
```

### Verify Dataset (Python)

```python
from floeval import DatasetLoader

dataset = DatasetLoader.from_file("dataset.json", partial_dataset=False)
print(f"✅ {len(dataset.samples)} samples loaded")
```

---

## Quick Reference

### Required Config (evaluate)

```json
{
  "llm_config": {
    "base_url": "https://api.openai.com/v1",
    "api_key": "your-key",
    "chat_model": "gpt-4o-mini",
    "embedding_model": "text-embedding-3-small"
  },
  "evaluation_config": {
    "metrics": ["ragas:answer_relevancy", "ragas:faithfulness"]
  }
}
```

### Required Config (generate)

```json
{
  "llm_config": {
    "base_url": "https://api.openai.com/v1",
    "api_key": "your-key",
    "chat_model": "gpt-4o-mini",
    "embedding_model": "text-embedding-3-small"
  },
  "dataset_generation_config": {
    "generator_model": "gpt-4o-mini"
  }
}
```

### Partial Dataset Config

When samples lack `llm_response`, add:

```json
{
  "dataset_generation_config": {
    "generator_model": "gpt-4o-mini"
  }
}
```

---

## Still Stuck?

1. **Check examples**: See working code in [Examples](examples.md)
2. **Review metrics**: Understand requirements in [Metrics](metrics.md)
3. **Check API reference**: Full details in [API Reference](api-reference.md)
