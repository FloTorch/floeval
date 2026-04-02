"""Minimal FloTorch LLM for OpenAI-compatible chat completions (no tracing)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from floeval.flotorch.sdk.utils.http_utils import async_http_post

logger = logging.getLogger(__name__)


class FlotorchLLM:
    """OpenAI-compatible HTTP chat completion client for agent evaluation."""

    def __init__(
        self,
        model_id: str,
        api_key: str,
        base_url: str,
        chat_endpoint: str = "chat/completions",
        default_headers: Optional[Dict[str, str]] = None,
    ):
        self.model_id = model_id
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.chat_endpoint = chat_endpoint.lstrip("/")
        self._url = f"{self.base_url}/{self.chat_endpoint}"
        self.default_headers = dict(default_headers or {})

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.default_headers)
        return headers

    async def ainvoke(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        response_format: Optional[Dict] = None,
        extra_body: Optional[Dict] = None,
        **kwargs: Any,
    ) -> _LLMResponse:
        """Async chat completion request."""
        extra_headers = kwargs.pop("extra_headers", None)
        headers = self._headers()
        if isinstance(extra_headers, dict):
            headers.update({str(k): str(v) for k, v in extra_headers.items()})
        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            **(extra_body or {}),
        }
        if tools:
            payload["tools"] = tools
        if response_format:
            payload["response_format"] = response_format
        payload.update(kwargs)

        result = await async_http_post(
            url=self._url,
            headers=headers,
            json=payload,
        )
        return _LLMResponse(result)


class _LLMResponse:
    """Wrapper for LLM response with metadata and content."""

    def __init__(self, raw: Dict[str, Any]):
        self._raw = raw
        self.metadata = self._extract_metadata(raw)
        self.content = self._extract_content(raw)

    def _extract_metadata(self, raw: Dict) -> Dict[str, Any]:
        meta: Dict[str, Any] = {"raw_response": raw}
        if "usage" in raw:
            meta["inputTokens"] = str(raw["usage"].get("prompt_tokens", 0))
            meta["outputTokens"] = str(raw["usage"].get("completion_tokens", 0))
            meta["totalTokens"] = str(raw["usage"].get("total_tokens", 0))
        return meta

    def _extract_content(self, raw: Dict) -> str:
        try:
            msg = raw.get("choices", [{}])[0].get("message", {})
            if "content" in msg and msg["content"] is not None:
                return msg["content"] or ""
            return ""
        except (KeyError, IndexError):
            return ""
