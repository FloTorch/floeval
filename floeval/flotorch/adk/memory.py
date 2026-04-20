"""FloTorch ADK memory services for external memory usage."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from google.adk.memory import BaseMemoryService
from google.adk.memory.base_memory_service import SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry
from google.adk.sessions import Session
from google.genai import types
from typing_extensions import override

from floeval.flotorch.sdk.memory import FlotorchMemory, FlotorchVectorStore

logger = logging.getLogger(__name__)


class FlotorchADKVectorMemoryService(BaseMemoryService):
    """Memory service with vector search using FlotorchVectorStore."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        vectorstore_id: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
    ):
        self.vectorstore_id = vectorstore_id
        self.vector_store: Optional[FlotorchVectorStore] = None
        if vectorstore_id:
            self.vector_store = FlotorchVectorStore(
                base_url=base_url,
                api_key=api_key,
                vectorstore_id=vectorstore_id,
                default_headers=default_headers,
            )

    @override
    async def search_memory(
        self,
        query: str,
        knn: Optional[int] = None,
        **kwargs: Any,
    ) -> SearchMemoryResponse:
        """Search using vector store."""
        try:
            if not self.vector_store:
                logger.warning("Vector store not configured for FlotorchADKVectorMemoryService")
                return SearchMemoryResponse(memories=[])

            search_params: Dict[str, Any] = {"query": query}
            if knn:
                search_params["max_number_of_result"] = knn

            vector_response = self.vector_store.search(**search_params)
            formatted: List[str] = []
            if isinstance(vector_response, dict) and "data" in vector_response:
                for result in vector_response["data"]:
                    content = result.get("content", {})
                    if isinstance(content, list) and content:
                        block = content[0] if isinstance(content[0], dict) else {}
                        text = block.get("text", "")
                        if text:
                            formatted.append(text)

            memory_entries = [
                MemoryEntry(
                    content=types.Content(
                        parts=[types.Part(text=text)],
                        role="user",
                    ),
                    author="user",
                )
                for text in formatted
            ]
            return SearchMemoryResponse(memories=memory_entries)
        except Exception as e:
            logger.error("FlotorchADKVectorMemoryService.search_memory failed: %s", e)
            return SearchMemoryResponse(memories=[])

    @override
    async def add_session_to_memory(self, session: Session) -> None:
        """No-op for vector-only service."""
        pass


class FlotorchMemoryService(BaseMemoryService):
    """Memory service using FloTorch gateway for store and retrieve."""

    def __init__(
        self,
        name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
    ):
        self._name = name
        self._base_url = base_url or os.getenv("FLOTORCH_BASE_URL")
        self._api_key = api_key or os.getenv("FLOTORCH_API_KEY")
        if not self._base_url:
            raise ValueError("FLOTORCH_BASE_URL env or base_url parameter is required")
        if not self._api_key:
            raise ValueError("FLOTORCH_API_KEY env or api_key parameter is required")
        self._memory = FlotorchMemory(
            api_key=self._api_key,
            base_url=self._base_url,
            provider_name=self._name,
            default_headers=default_headers,
        )

    def _map_role_to_flotorch(self, role: str) -> str:
        """Map ADK roles to FloTorch expected roles."""
        mapping = {
            "model": "assistant",
            "assistant": "assistant",
            "user": "user",
            "system": "system",
            "tool": "tool",
            "developer": "developer",
            "agent": "assistant",
            "bot": "assistant",
            "ai": "assistant",
            "human": "user",
            "person": "user",
        }
        return mapping.get(role.lower(), "user")

    def _extract_role(self, event: Any) -> str:
        """Extract role from ADK event."""
        try:
            if hasattr(event, "content") and event.content:
                if hasattr(event.content, "role") and event.content.role:
                    return self._map_role_to_flotorch(event.content.role)
            if hasattr(event, "role") and event.role:
                return self._map_role_to_flotorch(event.role)
            if hasattr(event, "author") and event.author:
                return self._map_role_to_flotorch(event.author)
        except Exception:
            pass
        return "user"

    def _extract_content_text(self, event: Any) -> str:
        """Extract text from ADK event."""
        try:
            if hasattr(event, "content") and event.content:
                content_obj = event.content
                if hasattr(content_obj, "parts") and content_obj.parts:
                    texts = []
                    for part in content_obj.parts:
                        if hasattr(part, "text") and part.text:
                            texts.append(part.text)
                    return " ".join(texts)
                if hasattr(content_obj, "text") and content_obj.text:
                    return content_obj.text
            if hasattr(event, "text") and event.text:
                return event.text
            if hasattr(event, "content") and isinstance(event.content, str):
                return event.content
        except Exception:
            pass
        return "Content not available"

    def _get_timestamp(self, session: Session) -> str:
        """Get ISO8601 timestamp for session."""
        try:
            created_at = getattr(session, "created_at", None)
            if created_at is not None:
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                return created_at.isoformat()
        except Exception:
            pass
        return datetime.now(timezone.utc).isoformat()

    @override
    async def add_session_to_memory(self, session: Session) -> None:
        """Add session events to FloTorch memory."""
        try:
            events = getattr(session, "events", None) or getattr(session, "messages", [])
            messages = [
                {"role": self._extract_role(e), "content": self._extract_content_text(e)}
                for e in events
            ]
            metadata = {
                "source": "adk_session",
                "importance": 0.5,
                "category": "conversation",
                "tags": ["adk", "session"],
            }
            self._memory.add(
                messages=messages,
                userId=getattr(session, "user_id", "unknown"),
                appId=getattr(session, "app_name", "unknown"),
                metadata=metadata,
                timestamp=self._get_timestamp(session),
            )
        except Exception as e:
            logger.error("FlotorchMemoryService.add_session_to_memory failed: %s", e)

    @override
    async def search_memory(
        self, *, app_name: str, user_id: str, query: str
    ) -> SearchMemoryResponse:
        """Search FloTorch memory."""
        try:
            result = self._memory.search(
                userId=user_id,
                appId=app_name,
                query=query,
                page=1,
                limit=10,
            )
            raw = result.get("data", [])
            if not raw:
                return SearchMemoryResponse()

            entries = []
            for mem in raw:
                text = mem.get("memory") or mem.get("content") or mem.get("text")
                if not text and isinstance(mem.get("messages"), list) and mem["messages"]:
                    m0 = mem["messages"][0]
                    if isinstance(m0, dict):
                        text = m0.get("content")
                if text:
                    ts = mem.get("timestamp") or mem.get("createdAt") or mem.get("updatedAt")
                    entries.append(
                        MemoryEntry(
                            content=types.Content(
                                parts=[types.Part(text=str(text))],
                                role="user",
                            ),
                            author="user",
                            timestamp=ts,
                        )
                    )
            return SearchMemoryResponse(memories=entries)
        except Exception as e:
            logger.error("FlotorchMemoryService.search_memory failed: %s", e)
            return SearchMemoryResponse(memories=[])
