"""FloTorch ADK agent builder for Mode 4 evaluation."""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, cast

from google.adk.agents import LlmAgent

from floeval.flotorch.adk.llm import FlotorchADKLLM
from floeval.flotorch.adk.utils.warning_utils import SuppressOutput
from floeval.flotorch.sdk.utils.http_utils import http_get

logger = logging.getLogger(__name__)


def _gateway_root(base_url: str) -> str:
    """Derive gateway root from base_url for agent/MCP APIs.

    base_url may be .../openai/v1 (LLM base) or gateway root. Returns root only.
    """
    url = base_url.rstrip("/")
    if "/openai" in url:
        return url.split("/openai")[0].rstrip("/")
    return url


def sanitize_name(name: str) -> str:
    """Sanitize agent name to be a valid identifier."""
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if sanitized and not sanitized[0].isalpha() and sanitized[0] != "_":
        sanitized = f"agent_{sanitized}"
    sanitized = re.sub(r"_+", "_", sanitized)
    sanitized = sanitized.strip("_")
    return sanitized or "agent"


def remove_curly_braces(text: str) -> str:
    """Remove curly braces from text."""
    return re.sub(r"[{}]", "", text)


def build_simple_agent(
    base_url: str,
    api_key: str,
    model_id: str,
    instruction: str = "You are a helpful assistant.",
    tools: Optional[List[Any]] = None,
    chat_endpoint: str = "chat/completions",
) -> LlmAgent:
    """Build a minimal LlmAgent from config (no gateway fetch).

    Use for eval when no FloTorch gateway is available.
    """
    llm = FlotorchADKLLM(
        model_id=model_id,
        api_key=api_key,
        base_url=base_url,
        chat_endpoint=chat_endpoint,
    )
    return LlmAgent(
        name="eval_agent",
        model=llm,
        instruction=instruction,
        tools=tools or [],
    )


class FlotorchADKAgent:
    """Build LlmAgent from FloTorch gateway config for Mode 4 evaluation."""

    def __init__(
        self,
        agent_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        custom_tools: Optional[List[Any]] = None,
    ):
        self.agent_name = agent_name
        self.base_url = base_url or os.environ.get("FLOTORCH_BASE_URL")
        self.api_key = api_key or os.environ.get("FLOTORCH_API_KEY")
        self.custom_tools = custom_tools or []
        self.config = self._fetch_agent_config(agent_name)
        self._agent = self._build_agent_from_config(self.config)

    def _fetch_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Fetch agent config from gateway API."""
        if not self.base_url:
            raise ValueError("base_url is required to fetch agent configuration")
        if not self.api_key:
            raise ValueError("api_key is required to fetch agent configuration")
        root = _gateway_root(self.base_url)
        url = f"{root.rstrip('/')}/v1/agents/{agent_name}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            return http_get(url, headers=headers)
        except Exception as e:
            logger.error("Failed to fetch agent config: %s", e)
            raise

    def _build_tools(self, config: Dict[str, Any]) -> List[Any]:
        """Build tools list from config (MCP + custom). Memory excluded per plan."""
        tools: List[Any] = []
        try:
            for tool_cfg in config.get("tools", []):
                if tool_cfg.get("type") == "MCP":
                    mcp_conf = tool_cfg.get("config", {})
                    root = _gateway_root(self.base_url)
                    proxy_url = f"{root.rstrip('/')}/v1/mcps/{tool_cfg['name']}/proxy"
                    with SuppressOutput():
                        from google.adk.tools.mcp_tool.mcp_toolset import (
                            MCPToolset,
                            SseConnectionParams,
                            StreamableHTTPConnectionParams,
                        )

                        headers = dict(mcp_conf.get("headers", {}))
                        if self.api_key:
                            headers["Authorization"] = f"Bearer {self.api_key}"

                        if mcp_conf.get("transport") == "HTTP_STREAMABLE":
                            conn_params = StreamableHTTPConnectionParams(
                                url=proxy_url,
                                headers=headers,
                                timeout=mcp_conf.get("timeout", 30_000) / 1000.0,
                                sse_read_timeout=mcp_conf.get("sse_read_timeout", 300_000) / 1000.0,
                                terminate_on_close=False,
                            )
                        elif mcp_conf.get("transport") == "HTTP_SSE":
                            conn_params = SseConnectionParams(
                                url=proxy_url,
                                headers=headers,
                                timeout=mcp_conf.get("timeout", 30_000) / 1000.0,
                                sse_read_timeout=mcp_conf.get("sse_read_timeout", 300_000) / 1000.0,
                            )
                        else:
                            continue
                        tool_name = sanitize_name(tool_cfg["name"])
                        toolset = MCPToolset(connection_params=conn_params)
                        tools.append(toolset)
        except ImportError:
            logger.warning("MCP toolset not available; skipping MCP tools")
        tools.extend(self.custom_tools)
        return tools

    def _build_agent_from_config(self, config: Dict[str, Any]) -> LlmAgent:
        """Build LlmAgent from config dict."""
        llm_cfg = config.get("llm", {})
        model_id = llm_cfg.get("callableName", "gpt-3.5-turbo")
        chat_endpoint = llm_cfg.get("chatEndpoint", "chat/completions")
        # Pass base_url as-is; FlotorchLLM builds url as base_url + "/" + chat_endpoint
        llm = FlotorchADKLLM(
            model_id=model_id,
            api_key=self.api_key,
            base_url=self.base_url,
            chat_endpoint=chat_endpoint,
        )
        tools = self._build_tools(config)
        return LlmAgent(
            name=sanitize_name(config.get("name", self.agent_name)),
            model=llm,
            instruction=remove_curly_braces(config.get("systemPrompt", "You are helpful.")),
            description=remove_curly_braces(config.get("goal", "")),
            tools=tools,
        )

    def get_agent(self) -> LlmAgent:
        """Return the built LlmAgent."""
        return cast(LlmAgent, self._agent)
