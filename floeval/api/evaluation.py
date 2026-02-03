"""Evaluation orchestrator."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Mapping, Optional, Union

from pydantic import BaseModel, Field

from floeval.metric_providers.deepeval.adapter import DeepEvalGatewayConfig, DeepEvalLLMAdapter

from .dataset import Dataset
from .metrics.base import BaseMetric, MetricResult
from .metrics.registry import MetricRegistry

MetricSpec = Union[BaseMetric, str, Dict[str, Any]]
SUPPORTED_METRIC_PROVIDERS = ["deepeval", "ragas"]


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
        self.metrics = self._resolve_metrics(metrics)

        # TODO: Add support for per-provider LLM adapters
        self._provider_llm_adapters_mapping: Dict[str, Dict[str, None | DeepEvalLLMAdapter]] = {
            "deepeval": {"llm_model": None, "embeddings_model": None},
            "ragas": {"llm_model": None, "embeddings_model": None},
        }

    # TODO: Generalize to other providers
    def _get_deepeval_adapter(
        self, gateway_config: DeepEvalGatewayConfig, model_type: Literal["llm", "embedding"]
    ) -> DeepEvalLLMAdapter:
        """Create DeepEval LLM adapter."""
        if self._provider_llm_adapters_mapping["deepeval"][f"{model_type}_model"] is not None:
            _adapter = self._provider_llm_adapters_mapping["deepeval"][f"{model_type}_model"]
            assert _adapter is not None, (
                "Provider LLM/Embedding model Adapter should not be None here."
            )
            return _adapter

        deepeval_llm_adapter = DeepEvalLLMAdapter(
            model_name=gateway_config.llm_model, config=gateway_config
        )
        if gateway_config.llm_model:
            self._provider_llm_adapters_mapping["deepeval"]["llm_model"] = deepeval_llm_adapter
        if gateway_config.embedding_model:
            raise NotImplementedError("DeepEval embedding adapter not implemented yet.")
        return deepeval_llm_adapter

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

        If a metric does not accept `gateway_config`, we retry without it.
        """
        merged = self._merge_params(provider, metric_id, params)
        try:
            return self._registry.create(provider, metric_id, **merged)
        except TypeError as e:
            if "gateway_config" in merged:
                merged2 = dict(merged)
                merged2.pop("gateway_config", None)
                return self._registry.create(provider, metric_id, **merged2)
            raise e

    def _resolve_metrics(self, specs: List[MetricSpec]) -> List[BaseMetric]:
        """Resolve metric specs to instances."""
        resolved: List[BaseMetric] = []

        for spec in specs:
            # Case 1: Already an instance
            if isinstance(spec, BaseMetric):
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
                # PRD calls `evaluate()`. In this repo, evaluate() is an alias to compute().
                if metric.provider == "deepeval":
                    llm_adapter = self._get_deepeval_adapter(self.gateway_config, "llm")
                    result: MetricResult = metric.evaluate(sample, llm_adapter=llm_adapter)
                else:
                    # TODO: Implement other provider-specific adapter retrievals
                    result: MetricResult = metric.evaluate(sample)

                provider = getattr(metric, "provider", "unknown")
                metric_name = getattr(metric, "name", metric.__class__.__name__)
                key = f"{provider}:{metric_name}"

                passed = bool(result.metadata.get("passed", False))
                reason = result.metadata.get("reason") or result.metadata.get("error")

                metric_results[key] = {
                    "score": result.score,
                    "passed": passed,
                    "reason": reason,
                    "provider": provider,
                    "metadata": result.metadata,
                }

            # Samples are Pydantic models; include the raw inputs/ground_truth for readability.
            sample_results.append(
                {
                    "inputs": sample.inputs,
                    "ground_truth": sample.ground_truth,
                    "metrics": metric_results,
                }
            )

        aggregate_scores = self._aggregate(sample_results)
        summary = self._summarize(sample_results, aggregate_scores)

        return EvaluationResult(
            sample_results=sample_results,
            aggregate_scores=aggregate_scores,
            summary=summary,
        )

    def _aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate scores across samples."""
        scores: Dict[str, List[float]] = {}
        for result in results:
            for key, data in result["metrics"].items():
                scores.setdefault(key, []).append(float(data["score"]))
        return {key: (sum(vals) / len(vals)) for key, vals in scores.items() if vals}

    def _summarize(self, results: List[Dict[str, Any]], agg: Dict[str, float]) -> Dict[str, Any]:
        """Compute summary (PRD-style)."""
        total = len(results)
        passes: Dict[str, int] = {}
        providers = set()

        for result in results:
            for key, data in result["metrics"].items():
                passes.setdefault(key, 0)
                if data.get("passed"):
                    passes[key] += 1
                providers.add(data.get("provider", "unknown"))

        return {
            "total_samples": total,
            "providers_used": sorted(p for p in providers if p),
            "pass_rates": {k: (v / total if total else 0.0) for k, v in passes.items()},
            "aggregate_scores": agg,
        }
