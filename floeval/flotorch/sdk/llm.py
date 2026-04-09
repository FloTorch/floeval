"""HTTP client for OpenAI-style chat/completions (used by FlotorchADKLLM)."""

from __future__ import annotations

import logging
import traceback
from typing import Any, Dict, List, Optional

from floeval.flotorch.sdk.utils.http_utils import async_http_post

logger = logging.getLogger(__name__)


class FlotorchLLM:
    """POST JSON to base_url/chat_endpoint."""

    def __init__(
        self,
        model_id: str,
        api_key: str,
        base_url: str,
        chat_endpoint: str = "chat/completions",
    ):
        self.model_id = model_id
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.chat_endpoint = chat_endpoint.lstrip("/")
        self._url = f"{self.base_url}/{self.chat_endpoint}"

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def ainvoke(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        response_format: Optional[Dict] = None,
        extra_body: Optional[Dict] = None,
        **kwargs: Any,
    ) -> _LLMResponse:
        """Chat completion (async)."""
        kw = dict(kwargs)
        kw.pop("return_headers", None)

        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "extra_body": extra_body if extra_body is not None else {},
        }
        if tools:
            payload["tools"] = tools
        if response_format:
            payload["response_format"] = response_format
        payload.update(kw)

        try:
            result = await async_http_post(
                url=self._url,
                headers=self._headers(),
                json=payload,
                timeout=120.0,
            )
        except Exception:
            print(
                f"[floeval-debug] FlotorchLLM.ainvoke failed model={self.model_id} url={self._url}",
                flush=True,
            )
            traceback.print_exc()
            raise
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
