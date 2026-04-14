"""WorkflowEvaluation — evaluates multi-agent workflow executions.

Same API style as AgentEvaluation. Single metrics=[] list.
Internally routes each metric to the correct execution scope:
  execution_scope="per_sample"   → metric runs on each agent's AgentSample (one per trace)
  execution_scope="per_workflow" → metric runs once on the full WorkflowExecution

Users specify only: metrics=["builtin:task_completion", "builtin:agent_handoff_quality"]
The system handles routing automatically — users never need to know about
execution_scope.

Note on metric key format: WorkflowEvaluation uses "provider:metric_name" keys
(same as Evaluation) rather than bare "metric_name" keys (as AgentEvaluation does).
This is intentional to stay consistent with the newer convention.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Mapping

from pydantic import BaseModel, Field

import floeval.metric_providers  # noqa: F401 — trigger provider registration
from floeval.api.metrics.base import BaseMetric, MetricResult
from floeval.api.metrics.registry import MetricRegistry
from floeval.config.schemas.io.agent_dataset import (
    AgentSample,
    WorkflowExecution,
    _to_display_str,
)
from floeval.config.schemas.io.llm import OpenAIProviderConfig
from floeval.core.execution.llm_executor import OpenAIProvider
from floeval.utils.asyncio_compat import run_coroutine_sync
from floeval.utils.metric_constructor_kwargs import filter_kwargs_for_metric_factory

logger = logging.getLogger(__name__)

MetricSpec = BaseMetric | str | dict[str, Any]


class WorkflowEvaluationResult(BaseModel):
    """Result of a workflow evaluation run."""

    # Per-agent metric results: one entry per agent trace
    # Each entry: {"agent_name": str, "metrics": {key: {"score": float, "metadata": dict}}}
    per_agent_results: list[dict[str, Any]] = Field(default_factory=list)

    # Workflow-level metric results: {"metric_key": {"score": float, "metadata": dict}}
    workflow_results: dict[str, Any] = Field(default_factory=dict)

    # Aggregate: per-sample metrics averaged across agents + per-workflow metric scores
    aggregate_scores: dict[str, float] = Field(default_factory=dict)

    # Summary with per-agent breakdowns and workflow-level metadata
    summary: dict[str, Any] = Field(default_factory=dict)


class WorkflowEvaluation:
    """Evaluate a multi-agent workflow execution.

    Usage is identical in style to AgentEvaluation:

        result = WorkflowEvaluation(
            execution=execution,
            metrics=[
                "builtin:task_completion",          # per_sample: runs on each agent's trace
                "builtin:agent_handoff_quality",    # per_workflow: runs once on full execution
                "builtin:workflow_completion_rate", # per_workflow: deterministic
                "ragas:agent_goal_accuracy",        # per_sample: runs on each agent's trace
            ],
            llm_config=llm_config,
        ).run()

    The system infers execution scope from each metric's `execution_scope` class
    attribute. Users never configure this separately.

    Args:
        execution: WorkflowExecution produced by WorkflowExecutor.execute_and_build().
        metrics: Metric specs in any format supported by AgentEvaluation / Evaluation.
        llm_config: LLM configuration forwarded to metrics that need it.
        default_provider: Provider used when a metric spec has no provider prefix.
        metric_params: Per-metric parameter overrides (keyed by "metric" or "provider:metric").
        run_headers: Custom HTTP headers forwarded to all LLM providers.
    """

    def __init__(
        self,
        execution: WorkflowExecution,
        metrics: list[MetricSpec],
        llm_config: OpenAIProviderConfig | None = None,
        default_provider: str | None = "builtin",
        metric_params: Mapping[str, dict[str, Any]] | None = None,
        run_headers: dict[str, str] | None = None,
    ):
        self.execution = execution
        self.llm_config = llm_config
        self.default_provider = default_provider or "builtin"
        self.metric_params = dict(metric_params or {})
        self.run_headers = dict(run_headers or {})
        self._registry = MetricRegistry()

        self._all_metrics = self._resolve_metrics(metrics)
        # Internal routing — not exposed to users
        self._per_sample_metrics = [
            m for m in self._all_metrics if getattr(m, "execution_scope", "per_sample") != "per_workflow"
        ]
        self._per_workflow_metrics = [
            m for m in self._all_metrics if getattr(m, "execution_scope", "per_sample") == "per_workflow"
        ]

    # ── Metric resolution (identical pattern to AgentEvaluation._resolve_metrics) ──

    def _resolve_metrics(self, specs: list[MetricSpec]) -> list[BaseMetric]:
        resolved: list[BaseMetric] = []
        for spec in specs:
            if isinstance(spec, BaseMetric):
                self._inject_context(spec)
                resolved.append(spec)
                continue

            if isinstance(spec, str):
                if ":" in spec:
                    provider, metric_id = spec.split(":", 1)
                else:
                    metric_id = spec
                    provider = self._registry.resolve_best(metric_id, self.default_provider)
                params: dict[str, Any] = {}
            elif isinstance(spec, dict):
                metric_id = spec.get("id")
                if not metric_id:
                    raise ValueError("Metric dict spec must include 'id'.")
                provider = spec.get("provider") or self.default_provider
                params = spec.get("params", {}) or {}
            else:
                raise TypeError(f"Invalid metric spec: {spec!r}")

            metric = self._create_metric(provider, metric_id, params)
            if self.run_headers and hasattr(metric, "extra_headers"):
                metric.extra_headers = self.run_headers
            resolved.append(metric)
        return resolved

    def _inject_context(self, metric: BaseMetric) -> None:
        if self.llm_config and hasattr(metric, "llm_config") and metric.llm_config is None:
            metric.llm_config = self.llm_config
        if self.run_headers and hasattr(metric, "extra_headers"):
            metric.extra_headers = self.run_headers

    def _merge_params(self, provider: str, metric_id: str, params: dict[str, Any]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        merged.update(self.metric_params.get(metric_id, {}))
        merged.update(self.metric_params.get(f"{provider}:{metric_id}", {}))
        merged.update(params)
        return merged

    def _create_metric(self, provider: str, metric_id: str, params: dict[str, Any]) -> BaseMetric:
        merged = self._merge_params(provider, metric_id, params)

        factory = self._registry.get_class(provider, metric_id)
        if factory is None:
            available = self._registry.list_metrics(provider)
            raise KeyError(f"Unknown metric: {provider}:{metric_id}. Available: {available}")

        # Inject llm_config and llm_provider based on constructor signature
        try:
            sig = inspect.signature(factory.__init__ if isinstance(factory, type) else factory)
            if self.llm_config is not None:
                if "llm_config" in sig.parameters and "llm_config" not in merged:
                    merged["llm_config"] = self.llm_config
                if "llm_provider" in sig.parameters and "llm_provider" not in merged:
                    merged["llm_provider"] = OpenAIProvider(
                        config_name=f"{provider}:{metric_id}",
                        extra_headers=self.run_headers or None,
                        **self.llm_config.model_dump(),
                    )
        except (TypeError, ValueError, AttributeError):
            pass

        if self.run_headers and "extra_headers" not in merged:
            merged["extra_headers"] = self.run_headers

        merged = filter_kwargs_for_metric_factory(merged, factory)
        return self._registry.create(provider, metric_id, **merged)

    # ── Execution ──────────────────────────────────────────────────────────────

    def _build_agent_sample(self, trace, agent_name: str) -> AgentSample:
        """Build an AgentSample for per-sample metrics from one agent's trace.

        Per-agent user_input is derived from the first HumanMessage in the
        agent's trace — that is the actual input this agent received (which
        may differ from the workflow-level user_input for downstream agents).

        Example in a 2-agent workflow:
          agent 1 (summarizer):    user_input = "Summarize this document: ..."
          agent 2 (email_drafter): user_input = "Draft an email with: <summary>"

        Falling back to execution.user_input only when the trace has no
        HumanMessage (e.g. a pure tool-calling node with no human turn).
        """
        from floeval.config.schemas.io.agent_dataset import HumanMessage as _HumanMessage

        agent_user_input: str = _to_display_str(self.execution.user_input)
        for msg in trace.messages:
            if isinstance(msg, _HumanMessage) and msg.content:
                agent_user_input = msg.content
                break

        return AgentSample(
            user_input=agent_user_input,
            trace=trace,
            reference_outcome=self.execution.reference_outcome,
            metadata={
                "agent_name": agent_name,
                "workflow_goal": _to_display_str(self.execution.user_input),
                "workflow_session": self.execution.session_id,
            },
        )

    async def _run_per_sample_on_agent(
        self,
        trace,
        agent_name: str,
        metrics: list[BaseMetric],
    ) -> dict[str, Any]:
        """Run all per-sample metrics on one agent's trace."""
        sample = self._build_agent_sample(trace, agent_name)
        metric_results: dict[str, Any] = {}

        for metric in metrics:
            provider = getattr(metric, "provider", "unknown")
            key = f"{provider}:{metric.name}"
            try:
                if asyncio.iscoroutinefunction(getattr(metric, "aevaluate", None)):
                    result: MetricResult = await metric.aevaluate(sample)
                else:
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(
                        None, lambda m=metric: m.evaluate(sample)
                    )
            except Exception as e:
                logger.error(
                    "Metric %s failed for agent %s: %s", key, agent_name, e, exc_info=True
                )
                result = MetricResult(
                    score=None,
                    metadata={"error": str(e), "metric_name": metric.name},
                )

            metric_results[key] = {"score": result.score, "metadata": result.metadata}

        return {"agent_name": agent_name, "metrics": metric_results}

    async def _run_per_workflow(self, metrics: list[BaseMetric]) -> dict[str, Any]:
        """Run all per-workflow metrics once on the WorkflowExecution."""
        results: dict[str, Any] = {}
        for metric in metrics:
            provider = getattr(metric, "provider", "unknown")
            key = f"{provider}:{metric.name}"
            try:
                if asyncio.iscoroutinefunction(getattr(metric, "aevaluate", None)):
                    result: MetricResult = await metric.aevaluate(self.execution)
                else:
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(
                        None, lambda m=metric: m.evaluate(self.execution)
                    )
            except Exception as e:
                logger.error("Workflow metric %s failed: %s", key, e, exc_info=True)
                result = MetricResult(
                    score=None,
                    metadata={"error": str(e), "metric_name": metric.name},
                )
            results[key] = {"score": result.score, "metadata": result.metadata}
        return results

    async def arun(self) -> WorkflowEvaluationResult:
        """Run workflow evaluation asynchronously."""
        agent_names = self.execution.agent_names or [
            f"agent_{i}" for i in range(len(self.execution.agent_traces))
        ]

        # Run per-sample metrics on each agent trace in parallel
        per_agent_results: list[dict[str, Any]] = []
        if self._per_sample_metrics and self.execution.agent_traces:
            tasks = [
                self._run_per_sample_on_agent(trace, name, self._per_sample_metrics)
                for trace, name in zip(self.execution.agent_traces, agent_names)
            ]
            per_agent_results = list(await asyncio.gather(*tasks, return_exceptions=False))

        # Run per-workflow metrics once
        workflow_results: dict[str, Any] = {}
        if self._per_workflow_metrics:
            workflow_results = await self._run_per_workflow(self._per_workflow_metrics)

        aggregate_scores = self._aggregate(per_agent_results, workflow_results)
        summary = self._build_summary(per_agent_results, workflow_results, aggregate_scores)

        return WorkflowEvaluationResult(
            per_agent_results=per_agent_results,
            workflow_results=workflow_results,
            aggregate_scores=aggregate_scores,
            summary=summary,
        )

    def run(self) -> WorkflowEvaluationResult:
        """Run workflow evaluation synchronously."""
        return run_coroutine_sync(lambda: self.arun())

    def _aggregate(
        self,
        per_agent_results: list[dict[str, Any]],
        workflow_results: dict[str, Any],
    ) -> dict[str, float]:
        """Aggregate per-agent scores. Strategy driven by each metric's workflow_aggregate attribute:
          "mean" → average (quality/rate metrics)
          "sum"  → total  (count/volume metrics)
        """
        scores: dict[str, list[float]] = {}
        for row in per_agent_results:
            for key, data in (row.get("metrics") or {}).items():
                if data.get("score") is not None:
                    scores.setdefault(key, []).append(float(data["score"]))

        # Build key → metric lookup so we can read workflow_aggregate per metric
        metric_by_key = {
            f"{getattr(m, 'provider', 'unknown')}:{m.name}": m
            for m in self._per_sample_metrics
        }

        aggregated: dict[str, float] = {}
        for key, values in scores.items():
            if not values:
                continue
            strategy = getattr(metric_by_key.get(key), "workflow_aggregate", "mean")
            aggregated[key] = round(sum(values) if strategy == "sum" else sum(values) / len(values), 4)

        for key, data in workflow_results.items():
            if data.get("score") is not None:
                aggregated[key] = round(float(data["score"]), 4)

        return aggregated

    def _build_summary(
        self,
        per_agent_results: list[dict[str, Any]],
        workflow_results: dict[str, Any],
        aggregate_scores: dict[str, float],
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "total_agents": len(per_agent_results),
            "workflow_completion_rate": self.execution.workflow_completion_rate,
            "aggregate_scores": aggregate_scores,
            "per_agent_scores": {
                row["agent_name"]: {k: v.get("score") for k, v in row.get("metrics", {}).items()}
                for row in per_agent_results
            },
            "workflow_scores": {k: v.get("score") for k, v in workflow_results.items()},
        }
        # Identify agents that may have failed (task_completion below 0.5)
        failing = [
            row["agent_name"]
            for row in per_agent_results
            if row.get("metrics", {}).get("builtin:task_completion", {}).get("score", 1.0) < 0.5
        ]
        if failing:
            summary["failing_agents"] = failing
        return summary
