"""FloTorch memory and vector store clients for external memory usage."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from floeval.flotorch.sdk.utils import memory_utils

logger = logging.getLogger(__name__)


class FlotorchMemory:
    """Memory client for FloTorch gateway (add, get, search)."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        provider_name: str,
        default_headers: Optional[Dict[str, str]] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.provider_name = provider_name
        self.default_headers = dict(default_headers or {})

    def add(
        self,
        messages: List[Dict[str, str]],
        userId: Optional[str] = None,
        agentId: Optional[str] = None,
        appId: Optional[str] = None,
        sessionId: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
        providerParams: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add memory."""
        return memory_utils.add_memory(
            base_url=self.base_url,
            provider_name=self.provider_name,
            api_key=self.api_key,
            messages=messages,
            userId=userId,
            agentId=agentId,
            appId=appId,
            sessionId=sessionId,
            metadata=metadata,
            timestamp=timestamp,
            providerParams=providerParams,
            extra_headers=self.default_headers,
        )

    def search(
        self,
        userId: Optional[str] = None,
        agentId: Optional[str] = None,
        appId: Optional[str] = None,
        sessionId: Optional[str] = None,
        categories: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        page: Optional[int] = 1,
        limit: Optional[int] = 20,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search memories."""
        return memory_utils.search_memories(
            base_url=self.base_url,
            provider_name=self.provider_name,
            api_key=self.api_key,
            userId=userId,
            agentId=agentId,
            appId=appId,
            sessionId=sessionId,
            categories=categories,
            metadata=metadata,
            page=page,
            limit=limit,
            query=query,
            extra_headers=self.default_headers,
        )


class FlotorchVectorStore:
    """Vector store client for FloTorch gateway."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        vectorstore_id: str,
        default_headers: Optional[Dict[str, str]] = None,
    ):
        if not api_key or not api_key.strip():
            raise ValueError("API key cannot be empty.")
        if not vectorstore_id or not vectorstore_id.strip():
            raise ValueError("Vector store ID cannot be empty.")
        self.base_url = base_url
        self.api_key = api_key
        self.vectorstore_id = vectorstore_id
        self.default_headers = dict(default_headers or {})

    def search(
        self,
        query: str,
        max_number_of_result: Optional[int] = None,
        ranker: Optional[str] = None,
        score_threshold: Optional[float] = None,
        rewrite_query: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Search vector store."""
        kwargs: Dict[str, Any] = {
            "base_url": self.base_url,
            "api_key": self.api_key,
            "query": query,
            "vectorstore_id": self.vectorstore_id,
        }
        if max_number_of_result is not None:
            kwargs["max_number_of_result"] = max_number_of_result
        if ranker is not None:
            kwargs["ranker"] = ranker
        if score_threshold is not None:
            kwargs["score_threshold"] = score_threshold
        if rewrite_query is not None:
            kwargs["rewrite_query"] = rewrite_query
        kwargs["extra_headers"] = self.default_headers
        return memory_utils.search_vectorstore(**kwargs)
