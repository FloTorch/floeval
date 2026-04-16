"""Memory and vector store HTTP utilities for FloTorch gateway."""

from typing import Any, Dict, List, Optional, Union

from floeval.flotorch.sdk.utils.http_utils import (
    async_http_post,
    http_delete,
    http_get,
    http_post,
    http_put,
)

MemoryMessage = Dict[str, str]
MemoryMetadata = Dict[str, Any]
JSONType = Union[Dict[str, Any], List[Any]]


def _build_headers(
    api_key: str, extra_headers: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update({str(k): str(v) for k, v in extra_headers.items()})
    return headers


def _build_gateway_memory_url(base_url: str, provider_name: str) -> str:
    """Build memory API URL. Memory API does not use /openai in path; strips it from LLM base_url."""
    base = base_url.rstrip("/")
    if "/openai/v1" in base:
        base = base.replace("/openai/v1", "")
    elif "/openai" in base:
        base = base.replace("/openai", "")
    base = base.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/memory/{provider_name}"
    return f"{base}/v1/memory/{provider_name}"


def _gateway_root_from_llm_base_url(base_url: str) -> str:
    """Strip /openai/v1 (or /openai) from LLM base so paths are not doubled."""
    base = base_url.rstrip("/")
    if "/openai/v1" in base:
        base = base.replace("/openai/v1", "")
    elif "/openai" in base:
        base = base.replace("/openai", "")
    return base.rstrip("/")


def _build_vectorstore_search_url(base_url: str, vectorstore_id: str) -> str:
    """Vector store lives under /openai/v1/vector_stores/ on the gateway root."""
    root = _gateway_root_from_llm_base_url(base_url)
    return f"{root}/openai/v1/vector_stores/{vectorstore_id}/search"


def add_memory(
    base_url: str,
    provider_name: str,
    api_key: str,
    messages: List[MemoryMessage],
    userId: Optional[str] = None,
    agentId: Optional[str] = None,
    appId: Optional[str] = None,
    sessionId: Optional[str] = None,
    metadata: Optional[MemoryMetadata] = None,
    timestamp: Optional[str] = None,
    providerParams: Optional[Dict[str, Any]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Create a new memory entry."""
    url = _build_gateway_memory_url(base_url, provider_name) + "/memories"
    headers = _build_headers(api_key, extra_headers)
    payload = {
        "messages": messages,
        "userId": userId,
        "agentId": agentId,
        "appId": appId,
        "sessionId": sessionId,
        "metadata": metadata,
        "timestamp": timestamp,
        "providerParams": providerParams,
    }
    clean_payload = {k: v for k, v in payload.items() if v is not None}
    return http_post(url, headers=headers, json=clean_payload)


def get_memory(
    base_url: str,
    provider_name: str,
    api_key: str,
    memory_id: str,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Retrieve a memory by ID."""
    url = _build_gateway_memory_url(base_url, provider_name) + f"/memories/{memory_id}"
    headers = _build_headers(api_key, extra_headers)
    return http_get(url, headers=headers)


def update_memory(
    base_url: str,
    provider_name: str,
    api_key: str,
    memory_id: str,
    content: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Update an existing memory."""
    url = _build_gateway_memory_url(base_url, provider_name) + f"/memories/{memory_id}"
    headers = _build_headers(api_key, extra_headers)
    payload = {"content": content, "metadata": metadata}
    clean_payload = {k: v for k, v in payload.items() if v is not None}
    return http_put(url, headers=headers, json=clean_payload)


def delete_memory(
    base_url: str,
    provider_name: str,
    api_key: str,
    memory_id: str,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Delete a memory by ID."""
    url = _build_gateway_memory_url(base_url, provider_name) + f"/memories/{memory_id}"
    headers = _build_headers(api_key, extra_headers)
    return http_delete(url, headers=headers)


def search_memories(
    base_url: str,
    provider_name: str,
    api_key: str,
    userId: Optional[str] = None,
    agentId: Optional[str] = None,
    appId: Optional[str] = None,
    sessionId: Optional[str] = None,
    createFrom: Optional[str] = None,
    createTo: Optional[str] = None,
    updateFrom: Optional[str] = None,
    updateTo: Optional[str] = None,
    categories: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    page: Optional[int] = 1,
    limit: Optional[int] = 20,
    query: Optional[str] = None,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Search memories using filters and criteria."""
    url = _build_gateway_memory_url(base_url, provider_name) + "/memories/search"
    headers = _build_headers(api_key, extra_headers)
    query_value = "*" if not (query and isinstance(query, str) and query.strip()) else query
    payload = {
        "userId": userId,
        "agentId": agentId,
        "appId": appId,
        "sessionId": sessionId,
        "createFrom": createFrom,
        "createTo": createTo,
        "updateFrom": updateFrom,
        "updateTo": updateTo,
        "categories": categories,
        "metadata": metadata,
        "query": query_value,
        "page": page,
        "limit": limit,
    }
    clean_payload = {k: v for k, v in payload.items() if v is not None}
    return http_post(url, headers=headers, json=clean_payload)


def search_vectorstore(
    base_url: str,
    api_key: str,
    query: str,
    vectorstore_id: str,
    max_number_of_result: int = 5,
    ranker: str = "auto",
    score_threshold: float = 0.2,
    rewrite_query: bool = True,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Search vector store."""
    headers = _build_headers(api_key, extra_headers)
    payload = {
        "query": query,
        "max_num_results": max_number_of_result,
        "ranking_options": {"ranker": ranker, "score_threshold": score_threshold},
        "rewrite_query": rewrite_query,
    }
    url = _build_vectorstore_search_url(base_url, vectorstore_id)
    return http_post(url, headers=headers, json=payload)


async def async_add_memory(
    base_url: str,
    provider_name: str,
    api_key: str,
    messages: List[MemoryMessage],
    userId: Optional[str] = None,
    agentId: Optional[str] = None,
    appId: Optional[str] = None,
    sessionId: Optional[str] = None,
    metadata: Optional[MemoryMetadata] = None,
    timestamp: Optional[str] = None,
    providerParams: Optional[Dict[str, Any]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Async create a new memory entry."""
    url = _build_gateway_memory_url(base_url, provider_name) + "/memories"
    headers = _build_headers(api_key, extra_headers)
    payload = {
        "messages": messages,
        "userId": userId,
        "agentId": agentId,
        "appId": appId,
        "sessionId": sessionId,
        "metadata": metadata,
        "timestamp": timestamp,
        "providerParams": providerParams,
    }
    clean_payload = {k: v for k, v in payload.items() if v is not None}
    return await async_http_post(url, headers=headers, json=clean_payload)


async def async_search_memories(
    base_url: str,
    provider_name: str,
    api_key: str,
    userId: Optional[str] = None,
    agentId: Optional[str] = None,
    appId: Optional[str] = None,
    sessionId: Optional[str] = None,
    createFrom: Optional[str] = None,
    createTo: Optional[str] = None,
    updateFrom: Optional[str] = None,
    updateTo: Optional[str] = None,
    categories: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    page: Optional[int] = 1,
    limit: Optional[int] = 20,
    query: Optional[str] = None,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Async search memories."""
    url = _build_gateway_memory_url(base_url, provider_name) + "/memories/search"
    headers = _build_headers(api_key, extra_headers)
    query_value = "*" if not (query and isinstance(query, str) and query.strip()) else query
    payload = {
        "userId": userId,
        "agentId": agentId,
        "appId": appId,
        "sessionId": sessionId,
        "createFrom": createFrom,
        "createTo": createTo,
        "updateFrom": updateFrom,
        "updateTo": updateTo,
        "categories": categories,
        "metadata": metadata,
        "query": query_value,
        "page": page,
        "limit": limit,
    }
    clean_payload = {k: v for k, v in payload.items() if v is not None}
    return await async_http_post(url, headers=headers, json=clean_payload)


async def async_search_vectorstore(
    base_url: str,
    api_key: str,
    query: str,
    vectorstore_id: str,
    max_number_of_result: Optional[int] = 5,
    ranker: Optional[str] = "auto",
    score_threshold: Optional[float] = 0.2,
    rewrite_query: Optional[bool] = True,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Async search vector store."""
    headers = _build_headers(api_key, extra_headers)
    payload = {
        "query": query,
        "max_num_results": max_number_of_result,
        "ranking_options": {"ranker": ranker, "score_threshold": score_threshold},
        "rewrite_query": rewrite_query,
    }
    url = _build_vectorstore_search_url(base_url, vectorstore_id)
    return await async_http_post(url, headers=headers, json=payload)
