"""Evaluation orchestrator."""

from __future__ import annotations

import inspect
import logging
from typing import Any, Mapping, cast

from pydantic import BaseModel, Field

from floeval.api.metrics.base import BaseMetric, MetricResult
from floeval.api.metrics.registry import MetricRegistry
from floeval.config.schemas.io.dataset import Dataset, PartialDataset
from floeval.config.schemas.io.llm import (
    LLMProviderConfig,
    OpenAIProviderConfig,
    _normalize_openai_base_url,
)
from floeval.core.execution.llm_executor import OpenAIProvider
from floeval.core.execution.response_synthesizer import populate_llm_responses
from floeval.metric_providers.deepeval.custom_adapter import (
    DeepEvalCustomMetricAdapter,
)
from floeval.metric_providers.ragas.adapter import RAGASAdapter
from floeval.utils.loaders import load_prompts_file
from floeval.utils.ragas_results import extract_ragas_score

logger = logging.getLogger(__name__)

MetricSpec = BaseMetric | str | dict[str, Any]


class EvaluationResult(BaseModel):
    """Evaluation results."""

    sample_results: list[dict[str, Any]]
    aggregate_scores: dict[str, float]
    summary: dict[str, Any] = Field(default_factory=dict)


class Evaluation:
    """Main evaluation orchestrator.

    Supports three metric formats :
    1) Instance: BaseMetric instance
    2) String: "answer_relevancy" or "ragas:answer_relevancy"
    3) Dict: {"id": "answer_relevancy", "provider": "ragas", "params": {...}}

    """

    def __init__(
        self,
        dataset: Dataset | PartialDataset,
        metrics: list[MetricSpec],
        default_provider: str | None = None,
        llm_config: Any | None = None,
        metric_params: Mapping[str, dict[str, Any]] | None = None,
        dataset_generator_model: str | None = None,
        prompts_file: str | None = None,
    ):
        self.dataset_generator_model = dataset_generator_model
        self.default_provider = default_provider
        self.llm_config: OpenAIProviderConfig | LLMProviderConfig | None = llm_config
        self.prompts_file = prompts_file
        self.dataset = self._prepare_dataset(dataset)
        self.metric_params = dict(metric_params or {})
        self._registry = MetricRegistry()

        # Cache adapters per provider
        self._provider_adapters: dict[str, Any] = {
            "ragas": {"adapter": None},
        }

        self.metrics = self._resolve_metrics(metrics)

    def _prepare_dataset(self, dataset: Dataset | PartialDataset) -> Dataset:
        """Prepare dataset for evaluation (e.g., populate LLM responses if needed)."""
        if isinstance(dataset, Dataset):
            return dataset

        if self.llm_config is None or not self.dataset_generator_model:
            raise ValueError(
                "llm_config must be provided to Evaluation() when using a "
                "PartialDataset. dataset_generator_model is also required."
            )

        _partial_dataset = dataset
        config_dict = (
            self.llm_config.model_dump()
            if hasattr(self.llm_config, "model_dump")
            else dict(self.llm_config)
        )
        if config_dict.get("base_url"):
            config_dict["base_url"] = _normalize_openai_base_url(config_dict["base_url"])

        llm_provider = OpenAIProvider(
            config_name=f"{self.dataset_generator_model}_generation",
            **(self.llm_config.model_dump() | {"chat_model": self.dataset_generator_model}),
        )

        # Load prompts file if specified
        prompts = None
        if self.prompts_file:
            prompts = load_prompts_file(self.prompts_file)

        dataset = populate_llm_responses(
            partial_dataset=_partial_dataset,
            llm_provider=llm_provider,
            prompts=prompts,
        )
        return dataset

    def _get_ragas_adapter(self, llm_config: Any | None) -> RAGASAdapter:
        """Return cached RAGAS adapter, creating it if needed."""
        if self._provider_adapters["ragas"]["adapter"] is None:
            self._provider_adapters["ragas"]["adapter"] = RAGASAdapter(config=llm_config)
        return self._provider_adapters["ragas"]["adapter"]

    def _merge_params(
        self, provider: str, metric_id: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge user-provided params with Evaluation-level defaults.

        Precedence (highest to lowest):
        1) Explicit params passed in the metric spec dict
        2) metric_params mapping (keyed by "provider:metric" or "metric")
        3) Evaluation.llm_config (injected as "llm_config" for metrics)
        """
        merged: dict[str, Any] = {}

        # 2) metric_params mapping
        merged.update(self.metric_params.get(metric_id, {}))
        merged.update(self.metric_params.get(f"{provider}:{metric_id}", {}))

        # 3) Evaluation-level llm config (metrics expect "llm_config" key)
        if self.llm_config is not None and "llm_config" not in merged:
            merged["llm_config"] = self.llm_config

        # 1) Spec params override everything
        merged.update(params)
        return merged

    def _create_metric_instance(
        self, provider: str, metric_id: str, params: dict[str, Any]
    ) -> BaseMetric:
        """Build merged params using constructor introspection, then create metric."""
        merged = self._merge_params(provider, metric_id, params)
        llm_config = merged.get("llm_config")
        if provider == "ragas" and "adapter" not in merged:
            merged["adapter"] = self._get_ragas_adapter(llm_config)
        elif provider == "deepeval" and "llm_config" not in merged and self.llm_config:
            merged["llm_config"] = self.llm_config

        metric_factory = self._registry.get_class(provider, metric_id)
        if metric_factory is None:
            available = self._registry.list_metrics(provider)
            raise KeyError(f"Unknown metric: {provider}:{metric_id}. Available: {available}")

        # Filter params by constructor signature when possible
        try:
            if callable(metric_factory) and not isinstance(metric_factory, type):
                sig = inspect.signature(metric_factory)
            else:
                sig = inspect.signature(metric_factory.__init__)
            accepted = set(sig.parameters) - {"self"}
            has_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            if not has_var_kw:
                merged = {k: v for k, v in merged.items() if k in accepted}
        except (TypeError, ValueError, AttributeError):
            pass

        return self._registry.create(provider, metric_id, **merged)

    def _resolve_metrics(self, specs: list[MetricSpec]) -> list[BaseMetric]:
        """Resolve metric specs to instances."""
        resolved: list[BaseMetric] = []

        for spec in specs:
            # Case 1: Already an instance
            if isinstance(spec, BaseMetric):
                if (
                    self.llm_config is not None
                    and hasattr(spec, "llm_config")
                    and spec.llm_config is None
                ):
                    spec.llm_config = self.llm_config
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

    def _group_metrics_by_strategy(self) -> dict[str, list[BaseMetric]]:
        """Group metrics by execution strategy. Returns standalone, ragas, deepeval lists."""
        grouped: dict[str, list[BaseMetric]] = {
            "standalone": [],
            "ragas": [],
            "deepeval": [],
        }

        for metric in self.metrics:
            execute_via = getattr(metric, "execute_via", None)
            if execute_via == "ragas":
                grouped["ragas"].append(metric)
            elif execute_via == "deepeval":
                grouped["deepeval"].append(metric)
            else:
                # Default to standalone (including None and any other values)
                grouped["standalone"].append(metric)

        return grouped

    def _run_standalone(self, metrics: list[BaseMetric]) -> list[dict[str, Any]]:
        """Execute standalone metrics."""
        sample_results: list[dict[str, Any]] = []

        for sample in self.dataset.samples:
            metric_results: dict[str, Any] = {}

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
                    logger.error(f"Metric {key} failed for sample: {e}", exc_info=True)
                    metric_results[key] = {
                        "score": None,
                        "passed": False,
                        "reason": str(e),
                        "provider": provider,
                        "metadata": {"error": str(e), "metric_name": metric_name},
                    }

            sample_results.append(sample.model_dump() | {"metrics": metric_results})

        return sample_results

    def _run_via_ragas_sync(self, metrics: list[BaseMetric]) -> list[dict[str, Any]]:
        """Run metrics via RAGAS evaluate(); map results back to sample dicts."""
        from ragas import evaluate as ragas_evaluate

        from floeval.metric_providers.ragas.custom_adapter import (
            RAGASCustomMetricAdapter,
        )

        ragas_adapter = self._get_ragas_adapter(self.llm_config)
        adapter = RAGASCustomMetricAdapter(self.llm_config, ragas_adapter=ragas_adapter)

        ragas_metrics = []
        metric_mapping = []

        for metric in metrics:
            try:
                ragas_class = adapter.transform_metric(metric)
                ragas_metric_instance = ragas_class(llm=adapter.llm, name=metric.name)
                ragas_metrics.append(ragas_metric_instance)
                metric_mapping.append((ragas_metric_instance, metric))
            except Exception as e:
                logger.error(
                    f"Failed to transform metric {metric.name} to RAGAS: {e}", exc_info=True
                )

        if not ragas_metrics:
            return []

        ragas_dataset = adapter.transform_dataset(self.dataset)

        try:
            ragas_eval_result = ragas_evaluate(
                dataset=ragas_dataset,
                metrics=ragas_metrics,
                llm=adapter.llm,
                embeddings=adapter.embeddings,
            )
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}", exc_info=True)
            return []

        if hasattr(ragas_eval_result, "to_pandas"):
            ragas_results = ragas_eval_result.to_pandas()
        elif hasattr(ragas_eval_result, "columns"):
            ragas_results = ragas_eval_result
        else:
            logger.error(
                f"RAGAS results cannot be converted to DataFrame. Type: {type(ragas_eval_result)}"
            )
            return []

        if not hasattr(ragas_results, "columns"):
            logger.error(
                f"RAGAS results is not a DataFrame after conversion. Type: {type(ragas_results)}"
            )
            return []

        available_columns = list(ragas_results.columns)
        logger.debug(f"RAGAS results columns: {available_columns}")

        sample_results: list[dict[str, Any]] = []

        for i, sample in enumerate(self.dataset.samples):
            metric_results: dict[str, Any] = {}

            for idx, (ragas_metric_instance, floeval_metric) in enumerate(metric_mapping):
                provider = "ragas"
                metric_name = floeval_metric.name
                key = f"{provider}:{metric_name}"
                try:
                    score = extract_ragas_score(
                        ragas_results,
                        i,
                        available_columns,
                        ragas_metric_instance,
                        metric_name,
                        len(ragas_metrics),
                        idx,
                    )
                    threshold = getattr(floeval_metric, "threshold", 0.5)

                    metric_results[key] = {
                        "score": score,
                        "passed": score >= threshold if score is not None else False,
                        "reason": None,
                        "provider": provider,
                        "metadata": {"threshold": threshold, "execution_provider": "ragas"},
                    }
                except Exception as e:
                    logger.error(f"Failed to extract RAGAS result for {key}: {e}", exc_info=True)
                    metric_results[key] = {
                        "score": None,
                        "passed": False,
                        "reason": str(e),
                        "provider": provider,
                        "metadata": {"error": str(e), "metric_name": metric_name},
                    }

            sample_results.append(sample.model_dump() | {"metrics": metric_results})

        return sample_results

    def _collect_results_sync(self, grouped: dict[str, list[BaseMetric]]) -> list[dict[str, Any]]:
        """Collect results synchronously. Shared routing for run()."""
        results: list[dict[str, Any]] = []
        if grouped["standalone"]:
            results = self._run_standalone(grouped["standalone"])
        if grouped["ragas"]:
            ragas_results = self._run_via_ragas_sync(grouped["ragas"])
            if not results:
                results.extend(ragas_results)
            else:
                self._merge_provider_results(results, ragas_results)
        if grouped["deepeval"]:
            deepeval_results = self._run_via_deepeval_sync(grouped["deepeval"])
            if not results:
                results.extend(deepeval_results)
            else:
                self._merge_provider_results(results, deepeval_results)
        return results

    def run(self) -> EvaluationResult:
        """Run evaluation with provider routing (pure sync)."""
        grouped = self._group_metrics_by_strategy()
        results = self._collect_results_sync(grouped)
        aggregate_scores = self._aggregate(results)
        summary = self._summarize(results, aggregate_scores)
        return EvaluationResult(
            sample_results=results,
            aggregate_scores=aggregate_scores,
            summary=summary,
        )

    def _merge_provider_results(
        self,
        existing_results: list[dict[str, Any]],
        new_results: list[dict[str, Any]],
    ) -> None:
        """Merge new_results into existing_results by index (metrics dict per sample)."""
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

    def _run_via_deepeval_sync(self, metrics: list[BaseMetric]) -> list[dict[str, Any]]:
        """Run metrics via deepeval.evaluate() per sample; map results to sample dicts."""
        from deepeval.evaluate import evaluate as deepeval_evaluate

        adapter = DeepEvalCustomMetricAdapter(self.llm_config)

        # Transform metrics to DeepEval classes
        deepeval_metric_classes = []
        for metric in metrics:
            try:
                deepeval_class = adapter.transform_metric(metric)
                deepeval_metric_classes.append((deepeval_class, metric))
            except Exception as e:
                logger.error(
                    f"Failed to transform metric {metric.name} to DeepEval: {e}", exc_info=True
                )

        if not deepeval_metric_classes:
            return []

        sample_results: list[dict[str, Any]] = []
        for sample in self.dataset.samples:
            metric_results: dict[str, Any] = {}
            test_case = adapter.transform_sample(sample)
            for deepeval_class, floeval_metric in deepeval_metric_classes:
                provider = "deepeval"
                metric_name = floeval_metric.name
                key = f"{provider}:{metric_name}"

                try:
                    deepeval_metric_instance = deepeval_class()
                    result = deepeval_evaluate(
                        metrics=[deepeval_metric_instance], test_cases=[test_case]
                    )
                    if result.test_results and len(result.test_results) > 0:
                        test_result = result.test_results[0]
                        if test_result.metrics_data and len(test_result.metrics_data) > 0:
                            metric_data = test_result.metrics_data[0]
                            score = metric_data.score
                            success = (
                                metric_data.success if hasattr(metric_data, "success") else None
                            )
                            threshold = getattr(floeval_metric, "threshold", 0.5)
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
                                "reason": getattr(deepeval_metric_instance, "reason", None),
                                "provider": provider,
                                "metadata": {
                                    "threshold": threshold,
                                    "execution_provider": "deepeval",
                                },
                            }
                        else:
                            raise ValueError("No metric data in DeepEval result")
                    else:
                        raise ValueError("No test results from DeepEval evaluation")

                except Exception as e:
                    logger.error(f"DeepEval metric {key} failed for sample: {e}", exc_info=True)
                    metric_results[key] = {
                        "score": None,
                        "passed": False,
                        "reason": str(e),
                        "provider": provider,
                        "metadata": {"error": str(e), "metric_name": metric_name},
                    }

            sample_results.append(sample.model_dump() | {"metrics": metric_results})

        return sample_results

    async def _collect_results_async(
        self, grouped: dict[str, list[BaseMetric]]
    ) -> list[dict[str, Any]]:
        """Collect results asynchronously. Shared routing for arun()."""
        import asyncio

        results: list[dict[str, Any]] = []
        if grouped["standalone"]:
            results = await self._arun_standalone(grouped["standalone"])
        loop = asyncio.get_running_loop()
        if grouped["ragas"]:
            ragas_results = await loop.run_in_executor(
                None, lambda: self._run_via_ragas_sync(grouped["ragas"])
            )
            if not results:
                results.extend(ragas_results)
            else:
                self._merge_provider_results(results, ragas_results)
        if grouped["deepeval"]:
            deepeval_results = await loop.run_in_executor(
                None, lambda: self._run_via_deepeval_sync(grouped["deepeval"])
            )
            if not results:
                results.extend(deepeval_results)
            else:
                self._merge_provider_results(results, deepeval_results)
        return results

    async def arun(self) -> EvaluationResult:
        """Run evaluation asynchronously with strategy routing.

        Uses the same _group_metrics_by_strategy() as run() for consistency.
        """
        grouped = self._group_metrics_by_strategy()
        results = await self._collect_results_async(grouped)
        aggregate_scores = self._aggregate(results)
        summary = self._summarize(results, aggregate_scores)
        return EvaluationResult(
            sample_results=results,
            aggregate_scores=aggregate_scores,
            summary=summary,
        )

    async def _arun_standalone(self, metrics: list[BaseMetric]) -> list[dict[str, Any]]:
        """Run standalone metrics asynchronously with concurrent execution per sample."""
        import asyncio

        sem = asyncio.Semaphore(8)  # Limit concurrent samples to prevent overwhelming API

        async def evaluate_sample(sample):
            async with sem:
                metric_results: dict[str, Any] = {}
                tasks = [metric.aevaluate(sample) for metric in metrics]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for metric, res in zip(metrics, results, strict=True):
                    provider = getattr(metric, "provider", "unknown")
                    metric_name = getattr(metric, "name", metric.__class__.__name__)
                    key = f"{provider}:{metric_name}"

                    if isinstance(res, Exception):
                        logger.error(f"Metric {key} failed: {res}", exc_info=True)
                        metric_results[key] = {
                            "score": None,
                            "passed": False,
                            "reason": str(res),
                            "provider": provider,
                            "metadata": {"error": str(res), "metric_name": metric_name},
                        }
                    else:
                        mres = cast(MetricResult, res)
                        passed = mres.metadata.get("passed")
                        if passed is not None:
                            passed = bool(passed)
                        reason = mres.metadata.get("reason") or mres.metadata.get("error")
                        metric_results[key] = {
                            "score": mres.score,
                            "passed": passed,
                            "reason": reason,
                            "provider": provider,
                            "metadata": mres.metadata,
                        }
                return sample.model_dump() | {"metrics": metric_results}

        sample_tasks = [evaluate_sample(s) for s in self.dataset.samples]
        return await asyncio.gather(*sample_tasks)

    def _aggregate(self, results: list[dict[str, Any]]) -> dict[str, float]:
        """Aggregate scores across samples. Skips None scores (failed evaluations)."""
        scores: dict[str, list[float]] = {}
        for result in results:
            for key, data in result["metrics"].items():
                score = data["score"]
                # Only aggregate non-None scores (None indicates evaluation failure)
                if score is not None:
                    scores.setdefault(key, []).append(float(score))
        return {key: (sum(vals) / len(vals)) for key, vals in scores.items() if vals}

    def _summarize(self, results: list[dict[str, Any]], agg: dict[str, float]) -> dict[str, Any]:
        """Compute summary (PRD-style)."""
        total = len(results)
        passes: dict[str, int] = {}
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
