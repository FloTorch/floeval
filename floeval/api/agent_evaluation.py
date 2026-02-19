"""Agent evaluation orchestrator."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field

import floeval.metric_providers  # noqa: F401 - trigger metric registration
from floeval.api.metrics.base import BaseMetric, MetricResult
from floeval.api.metrics.registry import MetricRegistry
from floeval.config.schemas.io.agent_dataset import AgentDataset, AgentSample
from floeval.config.schemas.io.llm import OpenAIProviderConfig
from floeval.core.execution.llm_executor import OpenAIProvider

logger = logging.getLogger(__name__)

MetricSpec = BaseMetric | str | dict[str, Any]


class AgentEvaluationResult(BaseModel):
    """Agent evaluation results."""

    sample_results: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


class AgentEvaluation:
    """Agent evaluation orchestrator.

    Supports Mode 1 (pre-captured traces), Mode 2 (partial + agent callable),
    Mode 4 (partial + agent_runner e.g. FloTorchRunner).
    """

    def __init__(
        self,
        dataset: AgentDataset,
        metrics: list[MetricSpec],
        llm_config: OpenAIProviderConfig | None = None,
        agent: Callable[[str], str | Any] | Callable[[str], Awaitable[str | Any]]
        | None = None,
        agent_runner: Any | None = None,
        default_provider: str | None = "builtin",
        metric_params: dict[str, dict[str, Any]] | None = None,
    ):
        self.dataset = dataset
        self.metrics = metrics
        self.llm_config = llm_config
        self.agent = agent
        self.agent_runner = agent_runner
        self.default_provider = default_provider or "builtin"
        self.metric_params = dict(metric_params or {})
        self._registry = MetricRegistry()
        self._resolved_metrics = self._resolve_metrics(metrics)

    def _resolve_metrics(self, specs: list[MetricSpec]) -> list[BaseMetric]:
        """Resolve metric specs to instances."""
        resolved: list[BaseMetric] = []

        for spec in specs:
            if isinstance(spec, BaseMetric):
                resolved.append(spec)
                continue

            if isinstance(spec, dict):
                metric_id = spec.get("id")
                if not metric_id:
                    raise ValueError("Metric dict spec must include 'id'")
                provider = spec.get("provider") or self.default_provider
                params = spec.get("params", {}) or {}
                metric = self._create_metric(provider, metric_id, params)
                resolved.append(metric)
                continue

            if isinstance(spec, str):
                if ":" in spec:
                    provider, metric_id = spec.split(":", 1)
                else:
                    metric_id = spec
                    provider = self._registry.resolve_best(
                        metric_id, self.default_provider
                    )
                metric = self._create_metric(provider, metric_id, {})
                resolved.append(metric)
                continue

            raise TypeError(f"Invalid metric spec: {spec!r}")

        return resolved

    def _create_metric(
        self, provider: str, metric_id: str, params: dict[str, Any]
    ) -> BaseMetric:
        """Create metric instance. Injects llm_provider for goal_achievement."""
        merged = dict(self.metric_params.get(metric_id, {}))
        merged.update(self.metric_params.get(f"{provider}:{metric_id}", {}))
        merged.update(params)

        if provider == "builtin" and metric_id == "goal_achievement":
            if "llm_provider" not in merged and self.llm_config is not None:
                merged["llm_provider"] = OpenAIProvider(
                    config_name="goal_achievement",
                    **self.llm_config.model_dump(),
                )
        elif self.llm_config is not None and "llm_config" not in merged:
            merged["llm_config"] = self.llm_config

        return self._registry.create(provider, metric_id, **merged)

    def _ensure_full_samples(self) -> list[AgentSample]:
        """Ensure all samples have traces. Run agent/runner if partial."""
        if not self.dataset.is_partial:
            return self.dataset.all_full

        partial = self.dataset.all_partial
        if not partial:
            return []

        if self.agent_runner is not None:
            if hasattr(self.agent_runner, "run_on_dataset"):
                return self.agent_runner.run_on_dataset(partial)
            full = []
            for p in partial:
                trace = self.agent_runner.run(p.user_input)
                full.append(AgentSample.from_partial(p, trace))
            return full

        if self.agent is not None:
            from floeval.utils.agent_trace import TraceCollector

            collector = TraceCollector(self.agent)
            return collector.collect(partial)

        raise ValueError(
            "Dataset has partial samples but no agent or agent_runner provided. "
            "Pass agent (callable) for Mode 2 or agent_runner for Mode 4."
        )

    def run(self) -> AgentEvaluationResult:
        """Run evaluation synchronously."""
        return asyncio.run(self.arun())

    async def arun(self) -> AgentEvaluationResult:
        """Run evaluation asynchronously."""
        full_samples = self._ensure_full_samples()

        # Print captured traces (from partial -> full) for debugging/verification
        if full_samples and self.dataset.is_partial:
            captured = [
                {
                    "user_input": s.user_input,
                    "trace": {
                        "messages": [m.model_dump() for m in s.trace.messages],
                        "final_response": s.trace.final_response,
                    },
                }
                for s in full_samples
            ]
            print("Captured traces (full dataset for evaluation):")
            print(json.dumps(captured, indent=2, default=str))

        if not full_samples:
            return AgentEvaluationResult(
                sample_results=[],
                summary={"error": "No full samples to evaluate"},
            )

        sample_results: list[dict[str, Any]] = []
        metric_scores: dict[str, list[float]] = {}

        for sample in full_samples:
            row: dict[str, Any] = {
                "user_input": sample.user_input,
                "final_response": sample.trace.final_response,
                "metrics": {},
            }

            for metric in self._resolved_metrics:
                name = getattr(metric, "name", metric.__class__.__name__)
                try:
                    if asyncio.iscoroutinefunction(getattr(metric, "aevaluate", None)):
                        result = await metric.aevaluate(sample)
                    else:
                        result = metric.evaluate(sample)
                except Exception as e:
                    logger.exception("Metric %s failed for sample: %s", name, e)
                    result = MetricResult(score=None, metadata={"error": str(e)})

                row["metrics"][name] = {
                    "score": result.score,
                    "metadata": result.metadata,
                }
                if result.score is not None:
                    metric_scores.setdefault(name, []).append(result.score)

            sample_results.append(row)

        summary = {}
        for name, scores in metric_scores.items():
            if scores:
                summary[name] = sum(scores) / len(scores)

        return AgentEvaluationResult(
            sample_results=sample_results,
            summary=summary,
        )
