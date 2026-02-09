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
        """
        Execute standalone metrics (default Floeval execution).
        
        Args:
            metrics: List of metrics to execute standalone
        
        Returns:
            List of sample results in Floeval format
        """
        sample_results: List[Dict[str, Any]] = []
        
        for sample in self.dataset:
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
            
            sample_results.append(sample.model_dump() | {"metrics": metric_results})
        
        return sample_results
    
    def _run_via_ragas_sync(self, metrics: List[BaseMetric]) -> List[Dict[str, Any]]:
        """
        Execute metrics through RAGAS native infrastructure.
        
        Process:
        1. Transform Floeval metrics → RAGAS metrics
        2. Transform Floeval dataset → RAGAS dataset
        3. Execute via ragas.evaluate() (per-sample)
        4. Transform results back to Floeval format
        
        Args:
            metrics: List of Floeval metrics with execute_via="ragas"
        
        Returns:
            List of sample results in Floeval format
        """
        import asyncio
        
        # Lazy import to avoid circular dependencies
        from floeval.metric_providers.ragas.custom_adapter import RAGASCustomMetricAdapter
        
        adapter = RAGASCustomMetricAdapter(self.gateway_config)
        
        # Transform metrics to RAGAS classes
        ragas_metric_classes = []
        for metric in metrics:
            try:
                ragas_class = adapter.transform_metric(metric)
                ragas_metric_classes.append((ragas_class, metric))
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Failed to transform metric {metric.name} to RAGAS: {e}", exc_info=True)
                # Skip this metric - will be handled in result transformation
        
        if not ragas_metric_classes:
            # No metrics successfully transformed
            return []
        
        # Execute per-sample (simpler than batch, matches existing pattern)
        sample_results: List[Dict[str, Any]] = []
        
        for sample in self.dataset:
            metric_results: Dict[str, Any] = {}
            
            # Transform sample to RAGAS format
            ragas_sample = adapter.transform_sample(sample)
            
            # Execute each RAGAS metric
            for ragas_class, floeval_metric in ragas_metric_classes:
                provider = "ragas"
                metric_name = floeval_metric.name
                key = f"{provider}:{metric_name}"
                
                try:
                    # Instantiate RAGAS metric with LLM
                    ragas_metric_instance = ragas_class(llm=adapter.llm, name=metric_name)
                    
                    # Execute RAGAS metric (async)
                    score = asyncio.run(
                        ragas_metric_instance._single_turn_ascore(ragas_sample, callbacks=None)
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
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(
                        f"RAGAS metric {key} failed for sample: {e}",
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
        """
        Run evaluation with provider routing.
        
        Groups metrics by execute_via attribute and routes to appropriate execution engine:
        - standalone: Direct Floeval execution (default)
        - ragas: RAGAS native execution
        - deepeval: DeepEval native execution
        """
        # Group metrics by execution strategy
        grouped = self._group_metrics_by_strategy()
        
        # Collect results from all execution paths
        all_sample_results: List[Dict[str, Any]] = []
        
        # Execute standalone metrics (existing logic)
        if grouped["standalone"]:
            standalone_results = self._run_standalone(grouped["standalone"])
            all_sample_results.extend(standalone_results)
        
        # Execute RAGAS metrics (new)
        if grouped["ragas"]:
            ragas_results = self._run_via_ragas_sync(grouped["ragas"])
            if not all_sample_results:
                # No existing results - just use provider results directly
                all_sample_results.extend(ragas_results)
            else:
                # Merge RAGAS results with existing results (by sample index)
                self._merge_provider_results(all_sample_results, ragas_results)
        
        # Execute DeepEval metrics
        if grouped["deepeval"]:
            deepeval_results = self._run_via_deepeval_sync(grouped["deepeval"])
            if not all_sample_results:
                # No existing results - just use provider results directly
                all_sample_results.extend(deepeval_results)
            else:
                # Merge DeepEval results with existing results (by sample index)
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
        """
        Merge provider-specific results into existing results by sample index.
        
        Modifies existing_results in-place by adding metrics from new_results.
        Assumes both lists have same length and correspond to same samples.
        
        Args:
            existing_results: Existing sample results (modified in-place)
            new_results: New provider-specific results to merge
        """
        if len(existing_results) != len(new_results):
            import logging
            logger = logging.getLogger(__name__)
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
        """
        Execute metrics through DeepEval native infrastructure.
        
        Process:
        1. Transform Floeval metrics → DeepEval metrics
        2. Transform Floeval dataset → DeepEval test cases
        3. Execute via deepeval.evaluate() (per-sample)
        4. Transform results back to Floeval format
        
        Args:
            metrics: List of Floeval metrics with execute_via="deepeval"
        
        Returns:
            List of sample results in Floeval format
        """
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
                import logging
                logger = logging.getLogger(__name__)
                logger.error(
                    f"Failed to transform metric {metric.name} to DeepEval: {e}",
                    exc_info=True
                )
                # Skip this metric - will be handled in result transformation
        
        if not deepeval_metric_classes:
            # No metrics successfully transformed
            return []
        
        # Execute per-sample (simpler than batch, matches existing pattern)
        sample_results: List[Dict[str, Any]] = []
        
        for sample in self.dataset:
            metric_results: Dict[str, Any] = {}
            
            # Transform sample to DeepEval format
            test_case = adapter.transform_sample(sample)
            
            # Execute each DeepEval metric
            for deepeval_class, floeval_metric in deepeval_metric_classes:
                provider = "deepeval"
                metric_name = floeval_metric.name
                key = f"{provider}:{metric_name}"
                
                try:
                    # Instantiate DeepEval metric
                    deepeval_metric_instance = deepeval_class()
                    
                    # Execute DeepEval metric via evaluate()
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
                    import logging
                    logger = logging.getLogger(__name__)
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
