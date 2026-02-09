"""Evaluation orchestrator."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

from pydantic import BaseModel, Field

from floeval.api.metrics.base import BaseMetric, MetricResult
from floeval.api.metrics.registry import MetricRegistry
from floeval.config import GatewayConfig
from floeval.config.schemas.io.dataset import Dataset
from floeval.metric_providers.ragas.adapter import RAGASAdapter

logger = logging.getLogger(__name__)

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

    def _group_metrics_by_strategy(self) -> Dict[str, List[BaseMetric]]:
        """
        Group metrics by execution strategy (execute_via attribute).
        
        Returns:
            Dictionary with keys "standalone", "ragas", "deepeval" mapping to metric lists
        """
        grouped: Dict[str, List[BaseMetric]] = {
            "standalone": [],
            "ragas": [],
            "deepeval": []
        }
        
        for metric in self.metrics:
            execute_via = getattr(metric, 'execute_via', None)
            if execute_via == "ragas":
                grouped["ragas"].append(metric)
            elif execute_via == "deepeval":
                grouped["deepeval"].append(metric)
            else:
                # Default to standalone (including None and any other values)
                grouped["standalone"].append(metric)
        
        return grouped
    
    def _run_standalone(self, metrics: List[BaseMetric]) -> List[Dict[str, Any]]:
        """Execute standalone metrics."""
        sample_results: List[Dict[str, Any]] = []

        for sample in self.dataset.samples:
            metric_results: Dict[str, Any] = {}
            
            for metric in metrics:
                provider = getattr(metric, "provider", "custom")
                metric_name = getattr(metric, "name", metric.__class__.__name__)
                key = f"{provider}:{metric_name}"
                
                try:
                    result: MetricResult = metric.evaluate(sample)
                    
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
            
            sample_results.append(sample.model_dump() | {"metrics": metric_results})
        
        return sample_results
    
    def _run_via_ragas_sync(self, metrics: List[BaseMetric]) -> List[Dict[str, Any]]:
        """Execute metrics via RAGAS native infrastructure."""
        from floeval.metric_providers.ragas.custom_adapter import RAGASCustomMetricAdapter
        from ragas import evaluate as ragas_evaluate
        
        adapter = RAGASCustomMetricAdapter(self.gateway_config)
        
        ragas_metrics = []
        metric_mapping = []
        
        for metric in metrics:
            try:
                ragas_class = adapter.transform_metric(metric)
                ragas_metric_instance = ragas_class(llm=adapter.llm, name=metric.name)
                ragas_metrics.append(ragas_metric_instance)
                metric_mapping.append((ragas_metric_instance, metric))
            except Exception as e:
                logger.error(f"Failed to transform metric {metric.name} to RAGAS: {e}", exc_info=True)
        
        if not ragas_metrics:
            return []
        
        ragas_dataset = adapter.transform_dataset(self.dataset)
        
        try:
            ragas_eval_result = ragas_evaluate(
                dataset=ragas_dataset,
                metrics=ragas_metrics,
                llm=adapter.llm,
                embeddings=adapter.embeddings
            )
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}", exc_info=True)
            return []
        
        if hasattr(ragas_eval_result, 'to_pandas'):
            ragas_results = ragas_eval_result.to_pandas()
        elif hasattr(ragas_eval_result, 'columns'):
            ragas_results = ragas_eval_result
        else:
            logger.error(f"RAGAS results cannot be converted to DataFrame. Type: {type(ragas_eval_result)}")
            return []
        
        if not hasattr(ragas_results, 'columns'):
            logger.error(f"RAGAS results is not a DataFrame after conversion. Type: {type(ragas_results)}")
            return []
        
        available_columns = list(ragas_results.columns)
        logger.debug(f"RAGAS results columns: {available_columns}")
        
        sample_results: List[Dict[str, Any]] = []
        
        for i, sample in enumerate(self.dataset.samples):
            metric_results: Dict[str, Any] = {}
            
            for idx, (ragas_metric_instance, floeval_metric) in enumerate(metric_mapping):
                provider = "ragas"
                metric_name = floeval_metric.name
                key = f"{provider}:{metric_name}"
                ragas_metric_name = getattr(ragas_metric_instance, 'name', metric_name)
                metric_class_name = ragas_metric_instance.__class__.__name__
                
                try:
                    score = None
                    
                    if ragas_metric_name in available_columns:
                        score = float(ragas_results[ragas_metric_name].iloc[i])
                    elif metric_name in available_columns:
                        score = float(ragas_results[metric_name].iloc[i])
                    elif metric_class_name in available_columns:
                        score = float(ragas_results[metric_class_name].iloc[i])
                    else:
                        for col in available_columns:
                            col_lower = col.lower()
                            metric_lower = metric_name.lower()
                            ragas_lower = ragas_metric_name.lower()
                            class_lower = metric_class_name.lower()
                            if (ragas_lower in col_lower or col_lower in ragas_lower or
                                metric_lower in col_lower or col_lower in metric_lower or
                                class_lower in col_lower or col_lower in class_lower):
                                score = float(ragas_results[col].iloc[i])
                                break
                    
                    if score is None:
                        if len(ragas_metrics) == 1 and len(available_columns) > 0:
                            score_col = available_columns[-1]
                            score = float(ragas_results[score_col].iloc[i])
                            logger.debug(f"Using fallback: extracted score from column {score_col} for metric {metric_name}")
                        elif len(ragas_metrics) > 1 and idx < len(available_columns):
                            score_col = available_columns[idx]
                            score = float(ragas_results[score_col].iloc[i])
                            logger.debug(f"Using index-based fallback: extracted score from column {score_col} (index {idx}) for metric {metric_name}")
                    
                    if score is None:
                        raise ValueError(
                            f"Could not find score for metric {metric_name} (ragas name: {ragas_metric_name}, "
                            f"class: {metric_class_name}) in RAGAS results. Available columns: {available_columns}"
                        )
                    
                    threshold = getattr(floeval_metric, 'threshold', 0.5)
                    
                    metric_results[key] = {
                        "score": score,
                        "passed": score >= threshold if score is not None else False,
                        "reason": None,
                        "provider": provider,
                        "metadata": {
                            "threshold": threshold,
                            "execution_provider": "ragas"
                        },
                    }
                except Exception as e:
                    logger.error(
                        f"Failed to extract RAGAS result for {key}: {e}",
                        exc_info=True
                    )
                    metric_results[key] = {
                        "score": None,
                        "passed": False,
                        "reason": str(e),
                        "provider": provider,
                        "metadata": {"error": str(e), "metric_name": metric_name},
                    }
            
            sample_results.append(sample.model_dump() | {"metrics": metric_results})
        
        return sample_results
    
    def run(self) -> EvaluationResult:
        """Run evaluation with provider routing."""
        grouped = self._group_metrics_by_strategy()
        all_sample_results: List[Dict[str, Any]] = []
        
        if grouped["standalone"]:
            standalone_results = self._run_standalone(grouped["standalone"])
            all_sample_results.extend(standalone_results)
        
        if grouped["ragas"]:
            ragas_results = self._run_via_ragas_sync(grouped["ragas"])
            if not all_sample_results:
                all_sample_results.extend(ragas_results)
            else:
                self._merge_provider_results(all_sample_results, ragas_results)
        
        if grouped["deepeval"]:
            deepeval_results = self._run_via_deepeval_sync(grouped["deepeval"])
            if not all_sample_results:
                all_sample_results.extend(deepeval_results)
            else:
                self._merge_provider_results(all_sample_results, deepeval_results)
        
        # Aggregate and summarize
        aggregate_scores = self._aggregate(all_sample_results)
        summary = self._summarize(all_sample_results, aggregate_scores)
        
        return EvaluationResult(
            sample_results=all_sample_results,
            aggregate_scores=aggregate_scores,
            summary=summary,
        )
    
    def _merge_provider_results(
        self,
        existing_results: List[Dict[str, Any]],
        new_results: List[Dict[str, Any]]
    ) -> None:
        """Merge provider-specific results into existing results by sample index."""
        if len(existing_results) != len(new_results):
            logger.warning(
                f"Mismatch in result lengths: existing={len(existing_results)}, "
                f"new={len(new_results)}. Results may not align correctly."
            )
        
        for i, new_result in enumerate(new_results):
            if i < len(existing_results):
                # Merge metrics from new_result into existing_result
                existing_metrics = existing_results[i].get("metrics", {})
                new_metrics = new_result.get("metrics", {})
                existing_metrics.update(new_metrics)
                existing_results[i]["metrics"] = existing_metrics
            else:
                # Append new result if beyond existing length
                existing_results.append(new_result)
    
    def _run_via_deepeval_sync(self, metrics: List[BaseMetric]) -> List[Dict[str, Any]]:
        """Execute metrics via DeepEval native infrastructure."""
        # Lazy import to avoid circular dependencies
        from floeval.metric_providers.deepeval.custom_adapter import DeepEvalCustomMetricAdapter
        from deepeval.evaluate import evaluate as deepeval_evaluate
        
        adapter = DeepEvalCustomMetricAdapter(self.gateway_config)
        
        # Transform metrics to DeepEval classes
        deepeval_metric_classes = []
        for metric in metrics:
            try:
                deepeval_class = adapter.transform_metric(metric)
                deepeval_metric_classes.append((deepeval_class, metric))
            except Exception as e:
                logger.error(
                    f"Failed to transform metric {metric.name} to DeepEval: {e}",
                    exc_info=True
                )
        
        if not deepeval_metric_classes:
            # No metrics successfully transformed
            return []
        
        # Execute per-sample (simpler than batch, matches existing pattern)
        sample_results: List[Dict[str, Any]] = []
        
        for sample in self.dataset.samples:
            metric_results: Dict[str, Any] = {}
            
            test_case = adapter.transform_sample(sample)
            
            # Execute each DeepEval metric
            for deepeval_class, floeval_metric in deepeval_metric_classes:
                provider = "deepeval"
                metric_name = floeval_metric.name
                key = f"{provider}:{metric_name}"
                
                try:
                    deepeval_metric_instance = deepeval_class()
                    # DeepEval expects list of metrics and list of test cases
                    result = deepeval_evaluate(
                        metrics=[deepeval_metric_instance],
                        test_cases=[test_case]
                    )
                    
                    # Extract result from DeepEval evaluation
                    if result.test_results and len(result.test_results) > 0:
                        test_result = result.test_results[0]
                        if test_result.metrics_data and len(test_result.metrics_data) > 0:
                            metric_data = test_result.metrics_data[0]
                            score = metric_data.score
                            success = metric_data.success if hasattr(metric_data, 'success') else None
                            
                            threshold = getattr(floeval_metric, 'threshold', 0.5)
                            
                            # Use metric instance state if available
                            if deepeval_metric_instance.score is not None:
                                score = deepeval_metric_instance.score
                            if deepeval_metric_instance.success is not None:
                                success = deepeval_metric_instance.success
                            
                            if score is None:
                                score = 0.0
                            
                            if success is None:
                                success = score >= threshold
                            
                            metric_results[key] = {
                                "score": score,
                                "passed": success,
                                "reason": getattr(deepeval_metric_instance, 'reason', None),
                                "provider": provider,
                                "metadata": {
                                    "threshold": threshold,
                                    "execution_provider": "deepeval"
                                },
                            }
                        else:
                            raise ValueError("No metric data in DeepEval result")
                    else:
                        raise ValueError("No test results from DeepEval evaluation")
                
                except Exception as e:
                    logger.error(
                        f"DeepEval metric {key} failed for sample: {e}",
                        exc_info=True
                    )
                    metric_results[key] = {
                        "score": None,
                        "passed": False,
                        "reason": str(e),
                        "provider": provider,
                        "metadata": {"error": str(e), "metric_name": metric_name},
                    }
            
            sample_results.append(sample.model_dump() | {"metrics": metric_results})
        
        return sample_results

    async def arun(self) -> EvaluationResult:
        """
        Run evaluation asynchronously (concurrent metric execution per sample).
        
        Uses metric.aevaluate() so metrics can run concurrently per sample.
        Optional for users who want true async concurrency; most users use run().
        """
        import asyncio

        sample_results: List[Dict[str, Any]] = []
        for sample in self.dataset.samples:
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
