"""Evaluation orchestrator."""

import logging
import os
import time
from typing import Any, Mapping, cast

from pydantic import BaseModel, Field
from ragas import aevaluate as ragas_aevaluate, evaluate as ragas_evaluate
from ragas.run_config import RunConfig

from floeval.api.metrics.base import BaseMetric, MetricResult
from floeval.api.metrics.registry import MetricRegistry
from floeval.config.schemas.io.conversational_dataset import (
    ConversationalDataset,
    PartialConversationalDataset,
)
from floeval.config.schemas.io.dataset import Dataset, PartialDataset
from floeval.config.schemas.io.llm import (
    LLMProviderConfig,
    OpenAIProviderConfig,
    _normalize_openai_base_url,
)
from floeval.core.execution.llm_executor import OpenAIProvider
from floeval.core.execution.response_synthesizer import (
    apopulate_llm_responses,
    populate_llm_responses,
)
from floeval.metric_providers.deepeval.batch_eval import (
    partition_deepeval_metrics,
    run_deepeval_conversational_batch,
    run_deepeval_llm_batch,
)
from floeval.metric_providers.ragas.adapter import RAGASAdapter
from floeval.metric_providers.ragas.custom_adapter import RAGASCustomMetricAdapter
from floeval.utils.job_status import log_job_status, log_job_status_error
from floeval.utils.loaders import load_prompts_file
from floeval.utils.metric_constructor_kwargs import filter_kwargs_for_metric_factory
from floeval.utils.ragas_results import extract_ragas_score

logger = logging.getLogger(__name__)


def _emit_progress(msg: str, *args: Any, extra: dict[str, Any]) -> None:
    # Worker may attach a file handler to floeval.job_status; otherwise log_job_status is a no-op.
    logger.info(msg, *args, extra=extra)
    log_job_status(msg, *args, extra=extra)


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
        dataset: Dataset | PartialDataset | ConversationalDataset | PartialConversationalDataset,
        metrics: list[MetricSpec],
        default_provider: str | None = None,
        llm_config: Any | None = None,
        metric_params: Mapping[str, dict[str, Any]] | None = None,
        dataset_generator_model: str | None = None,
        prompts_file: str | None = None,
        ragas_max_workers: int | None = None,
        run_headers: dict[str, Any] | None = None,
    ):
        self.dataset_generator_model = dataset_generator_model
        self.default_provider = default_provider
        self.llm_config: OpenAIProviderConfig | LLMProviderConfig | None = llm_config
        self.prompts_file = prompts_file
        self.dataset: (
            Dataset | PartialDataset | ConversationalDataset | PartialConversationalDataset
        ) = dataset
        self.metric_params = dict(metric_params or {})
        self._ragas_max_workers = ragas_max_workers
        self._run_headers: dict[str, Any] = dict(run_headers or {})
        self._registry = MetricRegistry()

        # Cache adapters per provider
        self._provider_adapters: dict[str, Any] = {
            "ragas": {"adapter": None},
        }

        if isinstance(self.dataset, PartialDataset) and (
            self.llm_config is None or not self.dataset_generator_model
        ):
            raise ValueError(
                "llm_config must be provided to Evaluation() when using a "
                "PartialDataset. dataset_generator_model is also required."
            )

        self.metrics = self._resolve_metrics(metrics)

    def _build_dataset_generation_context(self) -> tuple[OpenAIProvider, Any]:
        """Build provider and prompt context for PartialDataset generation."""
        if self.llm_config is None or not self.dataset_generator_model:
            raise ValueError(
                "llm_config must be provided to Evaluation() when using a "
                "PartialDataset. dataset_generator_model is also required."
            )

        config_dict = (
            self.llm_config.model_dump()
            if hasattr(self.llm_config, "model_dump")
            else dict(self.llm_config)
        )
        if config_dict.get("base_url"):
            config_dict["base_url"] = _normalize_openai_base_url(config_dict["base_url"])

        llm_provider = OpenAIProvider(
            config_name=f"{self.dataset_generator_model}_generation",
            extra_headers=self._run_headers or None,
            **(config_dict | {"chat_model": self.dataset_generator_model}),
        )

        # Load prompts file if specified
        prompts = None
        if self.prompts_file:
            prompts = load_prompts_file(self.prompts_file)

        return llm_provider, prompts

    def _ensure_dataset_sync(self) -> None:
        """Ensure partial dataset is expanded in synchronous flows."""
        if isinstance(self.dataset, PartialConversationalDataset):
            raise ValueError(
                "PartialConversationalDataset is not supported for automatic completion; "
                "provide a full ConversationalDataset."
            )
        if isinstance(self.dataset, (Dataset, ConversationalDataset)):
            return

        llm_provider, prompts = self._build_dataset_generation_context()
        self.dataset = populate_llm_responses(
            partial_dataset=self.dataset,
            llm_provider=llm_provider,
            prompts=prompts,
        )

    async def _ensure_dataset_async(self) -> None:
        """Ensure partial dataset is expanded in asynchronous flows."""
        if isinstance(self.dataset, PartialConversationalDataset):
            raise ValueError(
                "PartialConversationalDataset is not supported for automatic completion; "
                "provide a full ConversationalDataset."
            )
        if isinstance(self.dataset, (Dataset, ConversationalDataset)):
            return

        llm_provider, prompts = self._build_dataset_generation_context()
        self.dataset = await apopulate_llm_responses(
            partial_dataset=self.dataset,
            llm_provider=llm_provider,
            prompts=prompts,
        )

    def _get_ragas_adapter(self, llm_config: Any | None) -> RAGASAdapter:
        """Return cached RAGAS adapter, creating it if needed."""
        if self._provider_adapters["ragas"]["adapter"] is None:
            self._provider_adapters["ragas"]["adapter"] = RAGASAdapter(
                config=llm_config, extra_headers=self._run_headers or None
            )
        return cast(RAGASAdapter, self._provider_adapters["ragas"]["adapter"])

    def _require_row_dataset(self) -> Dataset | ConversationalDataset:
        """Return prepared single-turn or conversational dataset after expand."""
        d = self.dataset
        if isinstance(d, (Dataset, ConversationalDataset)):
            return d
        raise RuntimeError(
            "Dataset was not prepared. Call run() or arun() before metric execution."
        )

    def _row_samples(self) -> list[Any]:
        d = self._require_row_dataset()
        if isinstance(d, Dataset):
            return d.samples
        return d.samples

    def _validate_metric_dataset_compatibility(self, grouped: dict[str, list[BaseMetric]]) -> None:
        conv = isinstance(self.dataset, ConversationalDataset)
        if conv:
            for m in grouped["deepeval"]:
                if getattr(m, "deepeval_test_case_kind", "llm") != "conversational":
                    msg = (
                        f"ConversationalDataset cannot run LLMTestCase DeepEval metric "
                        f"{m.name!r}; use conversational metrics or a single-turn Dataset."
                    )
                    raise ValueError(msg)
            for m in grouped["ragas"]:
                if getattr(m, "ragas_sample_kind", "single_turn") != "multi_turn":
                    msg = (
                        f"ConversationalDataset cannot run single-turn RAGAS metric {m.name!r}; "
                        f"use multi_turn metrics (e.g. topic_adherence) or a Dataset."
                    )
                    raise ValueError(msg)
            return
        for m in grouped["deepeval"]:
            if getattr(m, "deepeval_test_case_kind", "llm") == "conversational":
                msg = (
                    f"Single-turn Dataset cannot run conversational DeepEval metric {m.name!r}; "
                    f"use ConversationalDataset."
                )
                raise ValueError(msg)
        for m in grouped["ragas"]:
            if getattr(m, "ragas_sample_kind", "single_turn") == "multi_turn":
                msg = (
                    f"Single-turn Dataset cannot run multi-turn RAGAS metric {m.name!r}; "
                    f"use ConversationalDataset."
                )
                raise ValueError(msg)

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
        if self._run_headers and "extra_headers" not in merged:
            merged["extra_headers"] = self._run_headers

        metric_factory = self._registry.get_class(provider, metric_id)
        if metric_factory is None:
            available = self._registry.list_metrics(provider)
            raise KeyError(f"Unknown metric: {provider}:{metric_id}. Available: {available}")

        merged = filter_kwargs_for_metric_factory(merged, metric_factory)

        return cast(BaseMetric, self._registry.create(provider, metric_id, **merged))

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
                if self._run_headers and hasattr(spec, "extra_headers"):
                    spec.extra_headers = self._run_headers
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
                if self._run_headers and hasattr(metric, "extra_headers"):
                    metric.extra_headers = self._run_headers
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
                if self._run_headers and hasattr(metric, "extra_headers"):
                    metric.extra_headers = self._run_headers
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

        for sample in self._row_samples():
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
        ragas_adapter = self._get_ragas_adapter(self.llm_config)
        adapter = RAGASCustomMetricAdapter(self.llm_config, ragas_adapter=ragas_adapter)
        dataset = self._require_row_dataset()

        ragas_metrics = []
        metric_mapping = []

        for metric in metrics:
            try:
                if getattr(metric, "ragas_sample_kind", "single_turn") == "multi_turn":
                    native = getattr(metric, "ragas_multiturn_metric", None)
                    if native is None:
                        logger.error(
                            "Multi-turn RAGAS metric %s has no ragas_multiturn_metric",
                            metric.name,
                        )
                        continue
                    native.llm = adapter.llm
                    ragas_metrics.append(native)
                    metric_mapping.append((native, metric))
                    continue
                ragas_class = adapter.transform_metric(metric)
                ragas_metric_instance = ragas_class(llm=adapter.llm, name=metric.name)
                ragas_metrics.append(ragas_metric_instance)
                metric_mapping.append((ragas_metric_instance, metric))
            except Exception as e:
                logger.error(
                    f"Failed to transform metric {metric.name} to RAGAS: {e}",
                    exc_info=True,
                )

        if not ragas_metrics:
            return []

        if isinstance(dataset, ConversationalDataset):
            ragas_dataset = adapter.transform_conversational_dataset(dataset)
            n_samples = len(dataset.samples)
        else:
            ragas_dataset = adapter.transform_dataset(dataset)
            n_samples = len(dataset.samples)
        logger.info(
            "RAGAS sync evaluate() path (run/executor): metrics=%d samples=%d",
            len(ragas_metrics),
            n_samples,
            extra={
                "phase": "floeval_ragas_sync",
                "eval_phase": "ragas_sync_evaluate",
                "metric_count": len(ragas_metrics),
                "sample_count": n_samples,
            },
        )

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
                "RAGAS results cannot be converted to DataFrame. Type: %s",
                type(ragas_eval_result),
            )
            return []

        if not hasattr(ragas_results, "columns"):
            logger.error(
                "RAGAS results is not a DataFrame after conversion. Type: %s",
                type(ragas_results),
            )
            return []

        available_columns = list(ragas_results.columns)
        logger.debug(f"RAGAS results columns: {available_columns}")

        sample_results: list[dict[str, Any]] = []

        for i, sample in enumerate(self._row_samples()):
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
        self._ensure_dataset_sync()
        grouped = self._group_metrics_by_strategy()
        self._validate_metric_dataset_compatibility(grouped)
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
        """Run all DeepEval metrics on all samples in one batched evaluate() call.

        Uses AsyncConfig(run_async=True) so DeepEval processes test cases and metrics
        concurrently, instead of the previous per-sample×per-metric nested loop.
        """
        row_ds = self._require_row_dataset()
        llm_metrics, conv_metrics = partition_deepeval_metrics(metrics)
        chunks: list[list[dict[str, Any]]] = []
        if isinstance(row_ds, ConversationalDataset):
            if llm_metrics:
                raise ValueError(
                    "Internal error: LLM DeepEval metrics on ConversationalDataset "
                    "(should have been rejected in validation)."
                )
            if conv_metrics:
                chunks.append(
                    run_deepeval_conversational_batch(
                        conversational_rows=row_ds.samples,
                        floeval_metrics=conv_metrics,
                        llm_config=self.llm_config,
                        emit_progress=_emit_progress,
                    )
                )
        else:
            if llm_metrics:
                chunks.append(
                    run_deepeval_llm_batch(
                        dataset=row_ds,
                        floeval_metrics=llm_metrics,
                        llm_config=self.llm_config,
                        emit_progress=_emit_progress,
                    )
                )
            if conv_metrics:
                raise ValueError(
                    "Internal error: conversational DeepEval metrics on single-turn Dataset."
                )
        if not chunks:
            return []
        out = chunks[0]
        for extra in chunks[1:]:
            self._merge_provider_results(out, extra)
        return out

    async def _run_via_ragas_async(self, metrics: list[BaseMetric]) -> list[dict[str, Any]]:
        """Run RAGAS metrics via aevaluate() in the current event loop."""
        ragas_adapter = self._get_ragas_adapter(self.llm_config)
        adapter = RAGASCustomMetricAdapter(self.llm_config, ragas_adapter=ragas_adapter)
        dataset = self._require_row_dataset()

        ragas_metrics = []
        metric_mapping = []
        for metric in metrics:
            try:
                if getattr(metric, "ragas_sample_kind", "single_turn") == "multi_turn":
                    native = getattr(metric, "ragas_multiturn_metric", None)
                    if native is None:
                        logger.error(
                            "Multi-turn RAGAS metric %s has no ragas_multiturn_metric",
                            metric.name,
                        )
                        continue
                    native.llm = adapter.llm
                    ragas_metrics.append(native)
                    metric_mapping.append((native, metric))
                    continue
                ragas_class = adapter.transform_metric(metric)
                ragas_metric_instance = ragas_class(llm=adapter.llm, name=metric.name)
                ragas_metrics.append(ragas_metric_instance)
                metric_mapping.append((ragas_metric_instance, metric))
            except Exception as e:
                logger.error(
                    "Failed to transform metric %s to RAGAS: %s", metric.name, e, exc_info=True
                )

        if not ragas_metrics:
            return []

        if isinstance(dataset, ConversationalDataset):
            ragas_dataset = adapter.transform_conversational_dataset(dataset)
            n_s = len(dataset.samples)
        else:
            ragas_dataset = adapter.transform_dataset(dataset)
            n_s = len(dataset.samples)
        if self._ragas_max_workers is not None:
            rw = max(1, int(self._ragas_max_workers))
        else:
            s = os.environ.get("FLOEVAL_RAGAS_MAX_WORKERS") or os.environ.get(
                "WORKER_RAGAS_MAX_WORKERS", "32"
            )
            s = s or "32"
            try:
                rw = max(1, int(s))
            except ValueError:
                rw = 32
        run_config = RunConfig(max_workers=rw, timeout=180, max_wait=60)

        _ragas_start_extra = {
            "phase": "floeval_ragas_async",
            "eval_phase": "ragas_aevaluate",
            "metric_count": len(ragas_metrics),
            "sample_count": n_s,
            "ragas_max_workers": rw,
        }
        _emit_progress(
            "RAGAS async aevaluate() starting: metrics=%d samples=%d ragas_max_workers=%d",
            len(ragas_metrics),
            n_s,
            rw,
            extra=_ragas_start_extra,
        )

        t_ragas = time.monotonic()
        try:
            ragas_eval_result = await ragas_aevaluate(
                dataset=ragas_dataset,
                metrics=ragas_metrics,
                llm=adapter.llm,
                embeddings=adapter.embeddings,
                run_config=run_config,
            )
        except Exception as e:
            elapsed = time.monotonic() - t_ragas
            logger.error(
                "RAGAS async evaluation failed after %.1fs: %s",
                elapsed,
                e,
                exc_info=True,
            )
            log_job_status_error(
                "RAGAS aevaluate() FAILED after %.1fs: [%s] %s",
                elapsed,
                type(e).__name__,
                str(e)[:600],
                extra={
                    "phase": "ragas_eval_failed",
                    "elapsed_s": round(elapsed, 1),
                    "error_type": type(e).__name__,
                    "error": str(e)[:600],
                },
            )
            return []

        ragas_elapsed = time.monotonic() - t_ragas
        _ragas_done_extra = {
            "phase": "floeval_ragas_async_done",
            "elapsed_s": round(ragas_elapsed, 1),
            "metric_count": len(ragas_metrics),
            "sample_count": n_s,
        }
        _emit_progress(
            "RAGAS async aevaluate() done in %.1fs metrics=%d samples=%d",
            ragas_elapsed,
            len(ragas_metrics),
            n_s,
            extra=_ragas_done_extra,
        )

        if hasattr(ragas_eval_result, "to_pandas"):
            ragas_results = ragas_eval_result.to_pandas()
        elif hasattr(ragas_eval_result, "columns"):
            ragas_results = ragas_eval_result
        else:
            logger.error(
                "RAGAS results cannot be converted to DataFrame. Type: %s",
                type(ragas_eval_result),
            )
            return []

        available_columns = list(ragas_results.columns)
        sample_results: list[dict[str, Any]] = []

        for i, sample in enumerate(self._row_samples()):
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
                    logger.error("Failed to extract RAGAS result for %s: %s", key, e, exc_info=True)
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

        n = len(self._row_samples())
        _routing_extra = {
            "phase": "floeval_arun_routing",
            "standalone_count": len(grouped["standalone"]),
            "ragas_count": len(grouped["ragas"]),
            "deepeval_count": len(grouped["deepeval"]),
            "sample_count": n,
        }
        _emit_progress(
            "floeval arun routing: standalone=%d ragas=%d deepeval=%d samples=%d",
            len(grouped["standalone"]),
            len(grouped["ragas"]),
            len(grouped["deepeval"]),
            n,
            extra=_routing_extra,
        )

        results: list[dict[str, Any]] = []
        if grouped["standalone"]:
            results = await self._arun_standalone(grouped["standalone"])
        # RAGAS: use aevaluate() directly — no thread/executor overhead.
        if grouped["ragas"]:
            ragas_results = await self._run_via_ragas_async(grouped["ragas"])
            if not results:
                results.extend(ragas_results)
            else:
                self._merge_provider_results(results, ragas_results)
        # DeepEval: still sync internally (manages its own event loop via AsyncConfig).
        if grouped["deepeval"]:
            loop = asyncio.get_running_loop()
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
        await self._ensure_dataset_async()
        grouped = self._group_metrics_by_strategy()
        self._validate_metric_dataset_compatibility(grouped)
        results = await self._collect_results_async(grouped)
        aggregate_scores = self._aggregate(results)
        summary = self._summarize(results, aggregate_scores)

        # Log per-metric summary to experiment JSONL for quick post-eval debugging.
        if results:
            scores_by_metric: dict[str, list[float]] = {}
            for r in results:
                for key, m in (r.get("metrics") or {}).items():
                    if m.get("score") is not None:
                        scores_by_metric.setdefault(key, []).append(float(m["score"]))
            metric_summary = {
                k: {
                    "avg": round(sum(v) / len(v), 4),
                    "pass_rate": round(
                        sum(1 for r in results if (r.get("metrics") or {}).get(k, {}).get("passed"))
                        / len(results),
                        4,
                    ),
                    "n": len(v),
                }
                for k, v in scores_by_metric.items()
            }
            if metric_summary:
                log_job_status(
                    "Evaluation metrics summary: %s",
                    metric_summary,
                    extra={
                        "phase": "eval_metrics_summary",
                        "sample_count": len(results),
                        "metrics": metric_summary,
                    },
                )
            else:
                log_job_status_error(
                    "Evaluation produced %d results but no metric scores were collected "
                    "(all providers may have failed or returned no scores)",
                    len(results),
                    extra={
                        "phase": "eval_metrics_summary",
                        "sample_count": len(results),
                        "metrics": {},
                    },
                )

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

        sample_tasks = [evaluate_sample(s) for s in self._row_samples()]
        return cast(list[dict[str, Any]], await asyncio.gather(*sample_tasks))

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
