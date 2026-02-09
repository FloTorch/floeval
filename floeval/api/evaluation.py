"""Evaluation orchestrator."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from pydantic import BaseModel, Field

from floeval.api.dataset import Dataset
from floeval.api.metrics.base import BaseMetric, MetricResult
from floeval.api.metrics.registry import MetricRegistry
from floeval.config import GatewayConfig
from floeval.metric_providers.ragas.adapter import RAGASAdapter

MetricSpec = BaseMetric | str | Dict[str, Any]


class EvaluationResult(BaseModel):
    """Evaluation results."""

    sample_results: List[Dict[str, Any]]
    aggregate_scores: Dict[str, float]
    summary: Dict[str, Any] = Field(default_factory=dict)


class Evaluation:
    """
    Main evaluation orchestrator.

    Supports three metric formats :
    1) Instance: BaseMetric instance
    2) String: "answer_relevancy" or "ragas:answer_relevancy"
    3) Dict: {"id": "answer_relevancy", "provider": "ragas", "params": {...}}

    Note: For gateway config, pass it via dict params:
      {"id": "answer_relevancy", "provider": "ragas", "params": {"gateway_config": ...}}
    """

    def __init__(
        self,
        dataset: Dataset,
        metrics: List[MetricSpec],
        default_provider: Optional[str] = None,
        gateway_config: Optional[Any] = None,
        metric_params: Optional[Mapping[str, Dict[str, Any]]] = None,
    ):
        import floeval.metric_providers

        self.dataset = dataset
        self.default_provider = default_provider
        self.gateway_config = gateway_config
        self.metric_params = dict(metric_params or {})
        self._registry = MetricRegistry()

        # Cache adapters per provider to avoid duplicate initialization
        # Note: DeepEval adapters are now initialized internally by metrics
        # Initialize BEFORE resolving metrics (which may use adapters)
        self._provider_adapters: Dict[str, Any] = {
            "ragas": {"adapter": None},
        }

        self.metrics = self._resolve_metrics(metrics)

    def _get_ragas_adapter(self, gateway_config: Optional[GatewayConfig]) -> RAGASAdapter:
        """Get or create RAGAS adapter (cached per gateway config)."""
        if self._provider_adapters["ragas"]["adapter"] is None:
            self._provider_adapters["ragas"]["adapter"] = RAGASAdapter(config=gateway_config)
        return self._provider_adapters["ragas"]["adapter"]

    def _merge_params(
        self, provider: str, metric_id: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge user-provided params with Evaluation-level defaults.

        Precedence (highest to lowest):
        1) Explicit params passed in the metric spec dict
        2) metric_params mapping (keyed by "provider:metric" or "metric")
        3) Evaluation.gateway_config (injected as "gateway_config")
        """
        merged: Dict[str, Any] = {}

        # 2) metric_params mapping
        merged.update(self.metric_params.get(metric_id, {}))
        merged.update(self.metric_params.get(f"{provider}:{metric_id}", {}))

        # 3) Evaluation-level gateway config (if metric doesn't override it)
        if self.gateway_config is not None and "gateway_config" not in merged:
            merged["gateway_config"] = self.gateway_config

        # 1) Spec params override everything
        merged.update(params)
        return merged

    def _create_metric_instance(
        self, provider: str, metric_id: str, params: Dict[str, Any]
    ) -> BaseMetric:
        """
        Instantiate a metric robustly.

        Injects cached adapters for providers that support reuse (RAGAS, DeepEval).
        """
        merged = self._merge_params(provider, metric_id, params)

        # Inject cached adapters for providers that support reuse
        gateway_config = merged.get("gateway_config")

        if provider == "ragas" and "adapter" not in merged:
            # RAGAS: Inject cached adapter if not provided
            merged["adapter"] = self._get_ragas_adapter(gateway_config)
        elif provider == "deepeval" and "gateway_config" not in merged and self.gateway_config:
            # DeepEval: Ensure gateway_config is available for adapter initialization
            merged["gateway_config"] = self.gateway_config

        try:
            return self._registry.create(provider, metric_id, **merged)
        except TypeError as e:
            # If metric doesn't accept gateway_config or adapter, retry without them
            if "gateway_config" in merged or "adapter" in merged:
                merged2 = dict(merged)
                merged2.pop("gateway_config", None)
                merged2.pop("adapter", None)
                return self._registry.create(provider, metric_id, **merged2)
            raise e

    def _resolve_metrics(self, specs: List[MetricSpec]) -> List[BaseMetric]:
        """Resolve metric specs to instances."""
        resolved: List[BaseMetric] = []

        for spec in specs:
            # Case 1: Already an instance
            if isinstance(spec, BaseMetric):
                # Inject gateway_config if not already set
                if self.gateway_config:
                    if not hasattr(spec, 'gateway_config') or spec.gateway_config is None:
                        spec.gateway_config = self.gateway_config
                        # Sync LLM helper config (e.g., criteria-based metrics use llm_helper)
                        if hasattr(spec, 'llm_helper') and getattr(spec, 'llm_helper', None) is not None:
                            spec.llm_helper.gateway_config = self.gateway_config
                resolved.append(spec)
                continue

            # Case 2: Dict
            if isinstance(spec, dict):
                metric_id = spec.get("id")
                if not metric_id:
                    raise ValueError("Metric dict spec must include 'id'.")

                provider = spec.get("provider")
                params = spec.get("params", {}) or {}

                if not provider:
                    provider = self.default_provider or self._registry.resolve_best(
                        metric_id, self.default_provider
                    )

                metric = self._create_metric_instance(provider, metric_id, params)
                resolved.append(metric)
                continue

            # Case 3: String
            if isinstance(spec, str):
                if ":" in spec:
                    provider, metric_id = spec.split(":", 1)
                else:
                    metric_id = spec
                    provider = self.default_provider or self._registry.resolve_best(
                        metric_id, self.default_provider
                    )

                metric = self._create_metric_instance(provider, metric_id, params={})
                resolved.append(metric)
                continue

            raise TypeError(f"Invalid metric spec: {spec!r}")

        return resolved

    def run(self) -> EvaluationResult:
        """Run evaluation."""
        sample_results: List[Dict[str, Any]] = []

        for sample in self.dataset:
            metric_results: Dict[str, Any] = {}

            for metric in self.metrics:
                provider = getattr(metric, "provider", "unknown")
                metric_name = getattr(metric, "name", metric.__class__.__name__)
                key = f"{provider}:{metric_name}"

                # Wrap each metric evaluation in try/except for fault isolation
                try:
                    # evaluate() -> compute(). Works for both sync and async metrics
                    # (async metrics run via ThreadPoolExecutor internally)
                    # All metrics handle their own adapters internally - no provider-specific logic needed
                    result: MetricResult = metric.evaluate(sample)

                    # passed may be None if threshold wasn't provided
                    passed = result.metadata.get("passed")
                    if passed is not None:
                        passed = bool(passed)
                    reason = result.metadata.get("reason") or result.metadata.get("error")

                    metric_results[key] = {
                        "score": result.score,
                        "passed": passed,
                        "reason": reason,
                        "provider": provider,
                        "metadata": result.metadata,
                    }
                except Exception as e:
                    # Isolate metric failures - continue evaluation for other metrics
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(
                        f"Metric {key} failed for sample: {e}",
                        exc_info=True
                    )
                    metric_results[key] = {
                        "score": None,
                        "passed": False,
                        "reason": str(e),
                        "provider": provider,
                        "metadata": {"error": str(e), "metric_name": metric_name},
                    }

            # Samples are Pydantic models; include the raw inputs/ground_truth for readability.
            sample_results.append(sample.model_dump() | {"metrics": metric_results})

        aggregate_scores = self._aggregate(sample_results)
        summary = self._summarize(sample_results, aggregate_scores)

        return EvaluationResult(
            sample_results=sample_results,
            aggregate_scores=aggregate_scores,
            summary=summary,
        )

    async def arun(self) -> EvaluationResult:
        """
        Run evaluation asynchronously (concurrent metric execution per sample).
        
        Uses metric.aevaluate() so metrics can run concurrently per sample.
        Optional for users who want true async concurrency; most users use run().
        """
        import asyncio

        sample_results: List[Dict[str, Any]] = []
        for sample in self.dataset:
            metric_results: Dict[str, Any] = {}
            tasks = []
            metric_list = list(self.metrics)
            for metric in metric_list:
                tasks.append(metric.aevaluate(sample))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for metric, res in zip(metric_list, results):
                provider = getattr(metric, "provider", "unknown")
                metric_name = getattr(metric, "name", metric.__class__.__name__)
                key = f"{provider}:{metric_name}"
                if isinstance(res, Exception):
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"Metric {key} failed for sample: {res}", exc_info=True)
                    metric_results[key] = {
                        "score": None,
                        "passed": False,
                        "reason": str(res),
                        "provider": provider,
                        "metadata": {"error": str(res), "metric_name": metric_name},
                    }
                else:
                    result = res
                    passed = result.metadata.get("passed")
                    if passed is not None:
                        passed = bool(passed)
                    reason = result.metadata.get("reason") or result.metadata.get("error")
                    metric_results[key] = {
                        "score": result.score,
                        "passed": passed,
                        "reason": reason,
                        "provider": provider,
                        "metadata": result.metadata,
                    }
            # Use model_dump() to get all sample fields, consistent with run() method
            sample_results.append(sample.model_dump() | {"metrics": metric_results})
        aggregate_scores = self._aggregate(sample_results)
        summary = self._summarize(sample_results, aggregate_scores)
        return EvaluationResult(
            sample_results=sample_results,
            aggregate_scores=aggregate_scores,
            summary=summary,
        )

    def _aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate scores across samples. Skips None scores (failed evaluations)."""
        scores: Dict[str, List[float]] = {}
        for result in results:
            for key, data in result["metrics"].items():
                score = data["score"]
                # Only aggregate non-None scores (None indicates evaluation failure)
                if score is not None:
                    scores.setdefault(key, []).append(float(score))
        return {key: (sum(vals) / len(vals)) for key, vals in scores.items() if vals}

    def _summarize(self, results: List[Dict[str, Any]], agg: Dict[str, float]) -> Dict[str, Any]:
        """Compute summary (PRD-style)."""
        total = len(results)
        passes: Dict[str, int] = {}
        providers = set()

        for result in results:
            for key, data in result["metrics"].items():
                passes.setdefault(key, 0)
                # Only count passes if threshold was provided (passed is not None)
                if data.get("passed") is True:
                    passes[key] += 1
                providers.add(data.get("provider", "unknown"))

        return {
            "total_samples": total,
            "providers_used": sorted(p for p in providers if p),
            "pass_rates": {k: (v / total if total else 0.0) for k, v in passes.items()},
            "aggregate_scores": agg,
        }
