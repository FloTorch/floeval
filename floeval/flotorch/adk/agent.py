"""Build ADK LlmAgent from a FloTorch gateway agent config (agent eval / Mode 4).

Uses session events for traces (no OTel). base_url is OpenAI-style; _gateway_root()
strips path suffixes for /v1/agents and MCP proxy URLs.
"""

from __future__ import annotations

import json
import logging
import os
import traceback
import re
import time
from typing import Any, Dict, List, Optional, Union, cast

from google.adk.agents import LlmAgent
from pydantic import create_model, Field

from floeval.flotorch.adk.llm import FlotorchADKLLM
from floeval.flotorch.adk.utils.warning_utils import SuppressOutput
from floeval.flotorch.sdk.utils.http_utils import http_get

logger = logging.getLogger(__name__)


def _gateway_root(base_url: str) -> str:
    """Strip /openai/v1, /openai, or /v1 from base_url for gateway REST paths."""
    url = base_url.rstrip("/")
    for suffix in ("/openai/v1", "/openai", "/v1"):
        if url.endswith(suffix):
            return url[: -len(suffix)].rstrip("/")
    return url


def sanitize_name(name: str) -> str:
    """Sanitize agent name to be a valid Python/ADK identifier."""
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if sanitized and not sanitized[0].isalpha() and sanitized[0] != "_":
        sanitized = f"agent_{sanitized}"
    sanitized = re.sub(r"_+", "_", sanitized)
    sanitized = sanitized.strip("_")
    return sanitized or "agent"


def remove_curly_braces(text: str) -> str:
    """Remove all curly braces {} from text."""
    return re.sub(r"[{}]", "", text)


def schema_to_pydantic_model(name: str, schema: dict):
    """Dynamically create a Pydantic model from a JSON schema dict."""
    properties = schema.get("properties", {})
    required_fields = set(schema.get("required", []))

    if len(properties) == 1:
        prop_name = next(iter(properties))
        if name.lower().startswith("input"):
            model_name = f"{prop_name.capitalize()}Input"
        elif name.lower().startswith("output"):
            model_name = f"{prop_name.capitalize()}Output"
        else:
            model_name = f"{prop_name.capitalize()}Schema"
    else:
        model_name = name

    fields = {}
    for prop, prop_schema in properties.items():
        field_type = str
        if prop_schema.get("type") == "integer":
            field_type = int
        elif prop_schema.get("type") == "number":
            field_type = float
        elif prop_schema.get("type") == "boolean":
            field_type = bool
        elif prop_schema.get("type") == "object":
            field_type = dict
        description = prop_schema.get("description", "")
        if prop in required_fields:
            fields[prop] = (field_type, Field(..., description=description))
        else:
            fields[prop] = (Optional[field_type], Field(default=None, description=description))

    return create_model(model_name, **fields)


def build_simple_agent(
    base_url: str,
    api_key: str,
    model_id: str,
    instruction: str = "You are a helpful assistant.",
    tools: Optional[List[Any]] = None,
    chat_endpoint: str = "chat/completions",
    default_headers: Optional[Dict[str, str]] = None,
) -> LlmAgent:
    """Build a minimal LlmAgent directly from config (no gateway fetch).

    Used for local eval mode when no FloTorch gateway is available.
    """
    llm = FlotorchADKLLM(
        model_id=model_id,
        api_key=api_key,
        base_url=base_url,
        chat_endpoint=chat_endpoint,
        default_headers=default_headers,
    )
    return LlmAgent(
        name="eval_agent",
        model=llm,
        instruction=instruction,
        tools=tools or [],
    )


class FlotorchADKAgent:
    """Load agent config from the gateway and build an LlmAgent (MCP tools, optional schemas)."""

    def __init__(
        self,
        agent_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        custom_tools: Optional[List[Any]] = None,
        enable_memory: bool = False,
        default_headers: Optional[Dict[str, str]] = None,
    ):
        self.agent_name = agent_name
        self.base_url = base_url or os.environ.get("FLOTORCH_BASE_URL", "")
        self.api_key = api_key or os.environ.get("FLOTORCH_API_KEY", "")
        self.custom_tools = custom_tools or []
        self.enable_memory = enable_memory
        self.default_headers = dict(default_headers or {})
        self.config = self._fetch_agent_config(agent_name)
        self._agent = self._build_agent_from_config(self.config)
        self._last_reload = time.time()

        logger.info(
            "FlotorchADKAgent created: '%s' memory=%s base_url=%s",
            self.agent_name,
            "enabled" if self.enable_memory else "disabled",
            self.base_url,
        )

    def _fetch_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Fetch agent config from the gateway REST API."""
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
        headers.update(self.default_headers)
        try:
            return http_get(url, headers=headers, timeout=30.0)
        except Exception:
            print(f"[floeval-debug] FlotorchADKAgent fetch config failed url={url}", flush=True)
            traceback.print_exc()
            raise

    def _build_tools(self, config: Dict[str, Any]) -> List[Any]:
        """Build tools list from config (memory pre-loader + MCP tools + custom tools)."""
        tools: List[Any] = []

        if self.enable_memory:
            from google.adk.tools import preload_memory
            tools.append(preload_memory)

        root = _gateway_root(self.base_url)
        mcp_tool_cfgs = [t for t in config.get("tools", []) if t.get("type") == "MCP"]

        for tool_cfg in mcp_tool_cfgs:
            mcp_conf = tool_cfg.get("config", {})
            proxy_url = f"{root.rstrip('/')}/v1/mcps/{tool_cfg['name']}/proxy"
            try:
                with SuppressOutput():
                    from google.adk.tools.mcp_tool.mcp_toolset import (
                        MCPToolset,
                        SseConnectionParams,
                        StreamableHTTPConnectionParams,
                    )

                    headers = dict(mcp_conf.get("headers", {}))
                    if self.api_key:
                        headers["Authorization"] = f"Bearer {self.api_key}"
                    headers.update(self.default_headers)

                    timeout = mcp_conf.get("timeout", 30_000) / 1000.0
                    sse_read_timeout = mcp_conf.get("sse_read_timeout", 300_000) / 1000.0

                    transport = mcp_conf.get("transport", "")
                    if transport == "HTTP_STREAMABLE":
                        conn_params = StreamableHTTPConnectionParams(
                            url=proxy_url,
                            headers=headers,
                            timeout=timeout,
                            sse_read_timeout=sse_read_timeout,
                            terminate_on_close=False,
                        )
                    elif transport == "HTTP_SSE":
                        conn_params = SseConnectionParams(
                            url=proxy_url,
                            headers=headers,
                            timeout=timeout,
                            sse_read_timeout=sse_read_timeout,
                        )
                    else:
                        logger.warning(
                            "Unknown MCP transport=%r for tool=%r — skipping",
                            transport,
                            tool_cfg.get("name"),
                        )
                        continue

                    toolset = MCPToolset(connection_params=conn_params)
                    tools.append(toolset)
            except Exception as exc:
                print("[floeval-debug] MCP toolset build exception:", flush=True)
                traceback.print_exc()
                logger.warning(
                    "Failed to build MCP toolset name=%s proxy=%s: %s — skipping",
                    tool_cfg.get("name"),
                    proxy_url,
                    exc,
                )
                continue

        tools.extend(self.custom_tools)
        return tools

    def _build_agent_from_config(self, config: Dict[str, Any]) -> LlmAgent:
        """Build LlmAgent from config dict fetched from the gateway."""
        llm_cfg = config.get("llm", {})
        model_id = llm_cfg.get("callableName", "gpt-3.5-turbo")
        chat_endpoint = llm_cfg.get("chatEndpoint", "chat/completions")

        llm = FlotorchADKLLM(
            model_id=model_id,
            api_key=self.api_key,
            base_url=self.base_url,
            chat_endpoint=chat_endpoint,
            default_headers=self.default_headers,
        )
        tools = self._build_tools(config)

        input_schema = None
        output_schema = None
        if config.get("inputSchema") is not None:
            try:
                input_schema = schema_to_pydantic_model("InputSchema", config["inputSchema"])
            except Exception as exc:
                logger.warning("Failed to build input schema: %s", exc)
        if config.get("outputSchema") is not None:
            try:
                output_schema = schema_to_pydantic_model("OutputSchema", config["outputSchema"])
            except Exception as exc:
                logger.warning("Failed to build output schema: %s", exc)

        agent = LlmAgent(
            name=sanitize_name(config.get("name", self.agent_name)),
            model=llm,
            instruction=remove_curly_braces(config.get("systemPrompt", "You are helpful.")),
            description=remove_curly_braces(config.get("goal", "")),
            tools=tools,
            input_schema=input_schema,
            output_schema=output_schema,
        )
        return agent

    def _get_synced_agent(self) -> LlmAgent:
        """Return current agent, reloading config if sync is enabled and interval passed."""
        sync_enabled = self.config.get("syncEnabled", False)
        if not sync_enabled:
            return self._agent

        sync_interval = self.config.get("syncInterval", 1_000_000)
        if time.time() - self._last_reload > sync_interval:
            logger.info("Sync: reloading config for agent '%s'", self.agent_name)
            try:
                new_config = self._fetch_agent_config(self.agent_name)
                if new_config and new_config != self.config:
                    self.config = new_config
                    self._agent = self._build_agent_from_config(self.config)
                    logger.info("Sync: agent '%s' reloaded", self.agent_name)
            except Exception as exc:
                logger.warning(
                    "Sync: failed to reload config for '%s': %s — using previous",
                    self.agent_name,
                    exc,
                )
            finally:
                self._last_reload = time.time()
        return self._agent

    def get_agent(self) -> LlmAgent:
        """Return the LlmAgent (wrapped so gateway sync reload stays transparent)."""
        return cast(LlmAgent, AgentProxy(self))


class AgentProxy(LlmAgent):
    """Delegates to the live LlmAgent from the manager (sync reload support). Subclasses LlmAgent for typing only."""

    def __init__(self, manager: "FlotorchADKAgent"):
        self._manager = manager

    def __getattr__(self, item: str):
        return getattr(self._manager._get_synced_agent(), item)

    def __setattr__(self, key: str, value: Any) -> None:
        if key == "_manager":
            return object.__setattr__(self, key, value)
        return setattr(self._manager._get_synced_agent(), key, value)
