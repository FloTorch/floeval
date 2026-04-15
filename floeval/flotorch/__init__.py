"""FloTorch gateway helpers for agent evaluation (optional import)."""

import os
from typing import Any

try:
    from floeval.flotorch.adk.agent import FlotorchADKAgent
    from floeval.flotorch.adk.utils.adk_utils import process_session_events
    from floeval.flotorch.dag import (
        DAG,
        Edge,
        Node,
        NodeType,
        aggregate_parent_results,
        initial_ready,
    )
    from floeval.flotorch.runner import FloTorchRunner
    from floeval.flotorch.workflow_executor import WorkflowExecutor
    from floeval.flotorch.adk.workflow_runner import WorkflowRunner

    def create_flotorch_runner(
        agent_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        llm_config: Any | None = None,
        run_headers: dict[str, str] | None = None,
    ) -> FloTorchRunner:
        """Build FloTorchRunner; gateway URL/key from args, llm_config, or FLOTORCH_* env."""
        if base_url is None and llm_config is not None:
            base_url = getattr(llm_config, "base_url", None)
        if base_url is None:
            base_url = os.environ.get("FLOTORCH_BASE_URL")
        if api_key is None and llm_config is not None:
            api_key = getattr(llm_config, "api_key", None)
        if api_key is None:
            api_key = os.environ.get("FLOTORCH_API_KEY")
        if not base_url or not api_key:
            raise ValueError(
                "FloTorch gateway base_url and api_key required. "
                "Pass llm_config, or set FLOTORCH_BASE_URL and FLOTORCH_API_KEY."
            )
        adk = FlotorchADKAgent(
            agent_name=agent_name,
            base_url=base_url,
            api_key=api_key,
            default_headers=run_headers,
        )
        return FloTorchRunner(adk.get_agent())

    __all__ = [
        "create_flotorch_runner",
        "FloTorchRunner",
        "process_session_events",
        "DAG",
        "Node",
        "NodeType",
        "Edge",
        "initial_ready",
        "aggregate_parent_results",
        "WorkflowExecutor",
        "WorkflowRunner",
    ]
except ImportError:
    __all__ = []
