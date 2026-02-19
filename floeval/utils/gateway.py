"""Normalize gateway URLs for OpenAI-compatible clients."""

# TODO: need to remove this - in order to generalize for all providers.
def normalize_openai_api_base(url: str) -> str:
    """Return OpenAI-compatible API base URL (adds /openai/v1 if needed)."""
    raw = url.strip()
    if not raw:
        raise ValueError("base_url cannot be empty.")
    if not raw.startswith(("http://", "https://")):
        raw = f"https://{raw}"
    raw = raw.rstrip("/")
    for suffix in ("/chat/completions", "/embeddings"):
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)]
            raw = raw.rstrip("/")
    if raw.endswith("/openai/v1") or raw.endswith("/v1"):
        return raw
    return f"{raw}/openai/v1"
