"""Agent evaluation orchestrator."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from typing import Any, Awaitable, Callable, Mapping

from pydantic import BaseModel, Field

import floeval.metric_providers  # noqa: F401 - trigger metric registration
from floeval.api.metrics.base import BaseMetric, MetricResult
from floeval.api.metrics.registry import MetricRegistry
from floeval.config.schemas.io.agent_dataset import (
    AgentDataset,
    AgentSample,
    _to_display_str,
)
from floeval.config.schemas.io.llm import OpenAIProviderConfig
from floeval.core.execution.llm_executor import OpenAIProvider
from floeval.utils.asyncio_compat import run_coroutine_sync
from floeval.utils.metric_constructor_kwargs import filter_kwargs_for_metric_factory

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
        agent: Callable[[str], str | Any] | Callable[[str], Awaitable[str | Any]] | None = None,
        agent_runner: Any | None = None,
        default_provider: str | None = "builtin",
        metric_params: Mapping[str, dict[str, Any]] | None = None,
        run_headers: dict[str, str] | None = None,
    ):
        self.dataset = dataset
        self.metrics = metrics
        self.llm_config = llm_config
        self.agent = agent
        self.agent_runner = agent_runner
        self.default_provider = default_provider or "builtin"
        self.metric_params = dict(metric_params or {})
        self.run_headers: dict[str, str] = dict(run_headers or {})
        self._registry = MetricRegistry()
        self._resolved_metrics = self._resolve_metrics(metrics)

    def _resolve_metrics(self, specs: list[MetricSpec]) -> list[BaseMetric]:
        """Resolve metric specs to instances."""
        resolved: list[BaseMetric] = []

        for spec in specs:
            if isinstance(spec, BaseMetric):
                if (
                    self.llm_config is not None
                    and hasattr(spec, "llm_config")
                    and spec.llm_config is None
                ):
                    spec.llm_config = self.llm_config
                if self.run_headers and hasattr(spec, "extra_headers"):
                    spec.extra_headers = self.run_headers
                resolved.append(spec)
                continue

            if isinstance(spec, dict):
                metric_id = spec.get("id")
                if not metric_id:
                    raise ValueError("Metric dict spec must include 'id'")
                provider = (
                    spec.get("provider")
                    or self.default_provider
                    or self._registry.resolve_best(metric_id, self.default_provider)
                )
                params = spec.get("params", {}) or {}
                metric = self._create_metric(provider, metric_id, params)
                if self.run_headers and hasattr(metric, "extra_headers"):
                    metric.extra_headers = self.run_headers
                resolved.append(metric)
                continue

            if isinstance(spec, str):
                if ":" in spec:
                    provider, metric_id = spec.split(":", 1)
                else:
                    metric_id = spec
                    provider = self._registry.resolve_best(metric_id, self.default_provider)
                metric = self._create_metric(provider, metric_id, params={})
                if self.run_headers and hasattr(metric, "extra_headers"):
                    metric.extra_headers = self.run_headers
                resolved.append(metric)
                continue

            raise TypeError(f"Invalid metric spec: {spec!r}")

        return resolved

    def _merge_params(
        self, provider: str, metric_id: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge user-provided params with evaluation-level defaults.

        Precedence (highest to lowest):
        1) Explicit params passed in the metric spec dict
        2) metric_params mapping (keyed by "provider:metric" or "metric")
        3) llm_config when metrics expect it
        """
        merged: dict[str, Any] = {}
        merged.update(self.metric_params.get(metric_id, {}))
        merged.update(self.metric_params.get(f"{provider}:{metric_id}", {}))
        merged.update(params)
        return merged

    def _inject_context_params(
        self, provider: str, metric_id: str, merged: dict[str, Any]
    ) -> dict[str, Any]:
        """Inject context-derived params based on metric constructor signature.

        Uses introspection to avoid hardcoding metric names. Injects:
        - llm_config: when metric accepts it and not in merged
        - llm_provider: when metric accepts it, not in merged, and we have llm_config
        """
        metric_factory = self._registry.get_class(provider, metric_id)
        if metric_factory is None:
            return merged

        try:
            if callable(metric_factory) and not isinstance(metric_factory, type):
                sig = inspect.signature(metric_factory)
            else:
                sig = inspect.signature(metric_factory.__init__)
        except (TypeError, ValueError, AttributeError):
            return merged

        if self.llm_config is None:
            return merged

        has_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )

        can_accept_llm_config = "llm_config" in sig.parameters or has_var_kw
        can_accept_llm_provider = "llm_provider" in sig.parameters or has_var_kw

        if can_accept_llm_config and "llm_config" not in merged:
            merged["llm_config"] = self.llm_config

        if can_accept_llm_provider and "llm_provider" not in merged:
            merged["llm_provider"] = OpenAIProvider(
                config_name=f"{provider}:{metric_id}",
                extra_headers=self.run_headers or None,
                **self.llm_config.model_dump(),
            )

        return merged

    def _create_metric(
        self, provider: str, metric_id: str, params: dict[str, Any]
    ) -> BaseMetric:
        """Create metric instance with dynamic dependency injection."""
        merged = self._merge_params(provider, metric_id, params)
        merged = self._inject_context_params(provider, metric_id, merged)
        if self.run_headers and "extra_headers" not in merged:
            merged["extra_headers"] = self.run_headers

        metric_factory = self._registry.get_class(provider, metric_id)
        if metric_factory is None:
            available = self._registry.list_metrics(provider)
            raise KeyError(f"Unknown metric: {provider}:{metric_id}. Available: {available}")
        merged = filter_kwargs_for_metric_factory(merged, metric_factory)

        try:
            return self._registry.create(provider, metric_id, **merged)
        except TypeError as e:
            if "llm_config" in merged or "llm_provider" in merged or "adapter" in merged:
                fallback = {k: v for k, v in merged.items() if k not in ("llm_config", "llm_provider", "adapter")
                }
                fallback = filter_kwargs_for_metric_factory(fallback, metric_factory)
                try:
                    return self._registry.create(provider, metric_id, **fallback)
                except TypeError:
                    pass
            raise e

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
                text = _to_display_str(p.user_input)
                trace = self.agent_runner.run(text)
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

    async def _ensure_full_samples_async(self) -> list[AgentSample]:
        """Async variant: run agent/runner without nested sync wrappers."""
        if not self.dataset.is_partial:
            return self.dataset.all_full

        partial = self.dataset.all_partial
        if not partial:
            return []

        if self.agent_runner is not None:
            runner = self.agent_runner
            if hasattr(runner, "run_on_dataset_async"):
                return await runner.run_on_dataset_async(partial)
            if hasattr(runner, "run_on_dataset"):
                # run_on_dataset is sync; run in executor to avoid blocking async loop.
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(
                    None, lambda: runner.run_on_dataset(partial)
                )
            full = []
            for p in partial:
                text = _to_display_str(p.user_input)
                if hasattr(runner, "arun") and asyncio.iscoroutinefunction(runner.arun):
                    trace = await runner.arun(text)
                else:
                    loop = asyncio.get_running_loop()
                    trace = await loop.run_in_executor(None, lambda t=text: runner.run(t))
                full.append(AgentSample.from_partial(p, trace))
            return full

        if self.agent is not None:
            from floeval.utils.agent_trace import TraceCollector

            collector = TraceCollector(self.agent)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: collector.collect(partial))

        raise ValueError(
            "Dataset has partial samples but no agent or agent_runner provided. "
            "Pass agent (callable) for Mode 2 or agent_runner for Mode 4."
        )

    def run(self) -> AgentEvaluationResult:
        """Run evaluation synchronously."""
        return run_coroutine_sync(lambda: self.arun())

    async def arun(self) -> AgentEvaluationResult:
        """Run evaluation asynchronously."""
        full_samples = await self._ensure_full_samples_async()

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
            logger.debug(
                "Captured traces (full dataset for evaluation): %s",
                json.dumps(captured, indent=2, default=str),
            )

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
                "reference_outcome": sample.reference_outcome,
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
