"""High-level workflow runner for agent evaluation.

Wraps WorkflowExecutor to run workflows on test datasets.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from floeval.config.schemas.io.agent_dataset import (
    AgentSample,
    AgentTrace,
    PartialAgentSample,
    _to_display_str,
)
from floeval.config.schemas.io.llm import OpenAIProviderConfig
from floeval.flotorch.dag import DAG
from floeval.flotorch.workflow_executor import WorkflowExecutor

logger = logging.getLogger(__name__)


class WorkflowRunner:
    """Run agent workflow on multiple test cases, capture all traces."""

    def __init__(
        self,
        dag_config: dict[str, Any],
        llm_config: OpenAIProviderConfig,
        app_name: str = "floeval-workflow",
        run_headers: dict[str, str] | None = None,
    ):
        """Initialize with DAG config (nodes, edges). Builds fresh DAG per sample with unique session."""
        self.dag_config = dag_config
        self.llm_config = llm_config
        self.app_name = app_name
        self.run_headers = dict(run_headers or {})

    def _build_dag_for_sample(self) -> DAG:
        """Build DAG with unique invocationId per workflow run (one session per sample)."""
        raw = {
            "config": self.dag_config,
            "invocationId": f"eval-{uuid4()}",
        }
        return DAG.from_builder_json(raw)

    async def run_on_dataset(
        self,
        partial_samples: list[PartialAgentSample],
    ) -> list[AgentSample]:
        """Execute workflow on each test case, return full samples with traces.

        Args:
            partial_samples: Test cases with user_input + reference_outcome.

        Returns:
            Full AgentSample instances with agent_traces and trace (last agent).
        """
        full_samples: list[AgentSample] = []

        for idx, partial in enumerate(partial_samples):
            logger.info("Running workflow for sample %d/%d", idx + 1, len(partial_samples))

            if isinstance(partial.user_input, str):
                wf_input: dict[str, Any] = {"text": partial.user_input}
            elif isinstance(partial.user_input, dict):
                wf_input = partial.user_input
            else:
                wf_input = {"text": str(partial.user_input)}

            dag = self._build_dag_for_sample()
            try:
                executor = WorkflowExecutor(
                    dag=dag,
                    llm_config=self.llm_config,
                    app_name=self.app_name,
                    user_id=f"eval-user-{idx}",
                    run_headers=self.run_headers,
                )

                result = await executor.execute(wf_input)
                agent_traces = result.get("agent_traces", [])
                agent_summaries = result.get("agent_summaries", [])
                final_output = result.get("final_output", "")

                trace: AgentTrace
                if agent_traces:
                    trace = agent_traces[-1]
                else:
                    trace = AgentTrace.from_simple_response(
                        _to_display_str(partial.user_input),
                        final_output or "[NO_RESPONSE]",
                    )

                metadata = dict(partial.metadata or {})
                if agent_summaries:
                    metadata["workflow_execution"] = {
                        "agents": agent_summaries,
                        "final_response": final_output,
                    }

                full_sample = AgentSample(
                    user_input=partial.user_input,
                    trace=trace,
                    reference_outcome=partial.reference_outcome,
                    reference_tool_calls=partial.reference_tool_calls,
                    agent_traces=agent_traces if agent_traces else None,
                    metadata=metadata,
                )
                full_samples.append(full_sample)
                logger.info("Sample %d completed successfully", idx + 1)

            except Exception as exc:
                logger.exception("Sample %d failed: %s", idx + 1, exc)
                full_samples.append(
                    AgentSample(
                        user_input=partial.user_input,
                        trace=AgentTrace.from_simple_response(
                            _to_display_str(partial.user_input),
                            f"[ERROR: {exc}]",
                        ),
                        reference_outcome=partial.reference_outcome,
                        metadata={**(partial.metadata or {}), "error": str(exc)},
                    )
                )

        return full_samples
