"""Batched DeepEval ``evaluate()`` runs for Floeval (LLM and conversational test cases)."""

import logging
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol, cast

from deepeval.evaluate import AsyncConfig, DisplayConfig, ErrorConfig, evaluate as deepeval_evaluate
from deepeval.test_case import ToolCall as DeepEvalToolCall
from pydantic import BaseModel

from floeval.api.metrics.base import BaseMetric
from floeval.config.schemas.io.agent_dataset import AgentSample, ToolCall as AgentToolCall
from floeval.config.schemas.io.conversation import ToolCallPayload
from floeval.config.schemas.io.conversational_dataset import ConversationalSample
from floeval.config.schemas.io.dataset import Dataset
from floeval.config.schemas.io.llm import LLMProviderConfig
from floeval.metric_providers.deepeval.custom_adapter import DeepEvalCustomMetricAdapter
from floeval.metric_providers.deepeval.multiturn_adapter import (
    agent_sample_to_conversational_test_case,
    conversational_sample_to_deepeval,
)
from floeval.utils.job_status import log_job_status, log_job_status_error

logger = logging.getLogger(__name__)

ProgressFn = Callable[..., None]


class _SupportsConversationalDeepevalBatch(Protocol):
    """DeepEval metrics that build a batched conversational metric instance."""

    def create_deepeval_metric_instance(self) -> Any: ...


def _collect_relevant_topics(rows: Sequence[ConversationalSample | AgentSample]) -> list[str]:
    seen: set[str] = set()
    topics: list[str] = []
    for row in rows:
        row_topics = getattr(row, "reference_topics", None)
        if not row_topics:
            continue
        for topic in row_topics:
            if topic in seen:
                continue
            seen.add(topic)
            topics.append(topic)
    return topics


def _to_deepeval_available_tool(
    tool: ToolCallPayload | AgentToolCall | dict[str, Any],
) -> DeepEvalToolCall:
    if isinstance(tool, AgentToolCall):
        return DeepEvalToolCall(name=tool.name, input_parameters=tool.args)
    payload = tool if isinstance(tool, ToolCallPayload) else ToolCallPayload.model_validate(tool)
    return DeepEvalToolCall(
        name=payload.name,
        input_parameters=payload.args,
        output=payload.output,
    )


def _collect_available_tools(
    rows: Sequence[ConversationalSample | AgentSample],
) -> list[DeepEvalToolCall]:
    dedup: set[tuple[str, str]] = set()
    available_tools: list[DeepEvalToolCall] = []
    for row in rows:
        calls = getattr(row, "reference_tool_calls", None)
        if not calls:
            continue
        for raw in calls:
            tool = _to_deepeval_available_tool(raw)
            tool_key = (tool.name, str(tool.input_parameters))
            if tool_key in dedup:
                continue
            dedup.add(tool_key)
            available_tools.append(tool)
    return available_tools


def _hydrate_conversational_metric_params(
    floeval_metrics: list[BaseMetric],
    rows: Sequence[ConversationalSample | AgentSample],
) -> None:
    """Fill metric constructor kwargs from the dataset before batched DeepEval ``evaluate()``.

    In the conversational batch path, Floeval builds **one** DeepEval metric instance per
    Floeval metric and runs it across **all** rows. Some DeepEval conversational metrics
    expect certain lists on the **metric object** (constructor kwargs), while Floeval's
    dataset schema stores the corresponding ground truth on **each row**:

    - ``topic_adherence`` — DeepEval wants ``relevant_topics``; rows may carry
      ``reference_topics`` (see ``ConversationalSample`` / ``AgentSample``).
    - ``tool_use`` — DeepEval wants ``available_tools``; rows may carry
      ``reference_tool_calls``.

    This function **mutates** each metric's ``_metric_params`` in place when that key is
    missing or empty, by aggregating values from ``rows`` (deduped union). If the user
    already set ``relevant_topics`` / ``available_tools`` in YAML or metric params, those
    values are left unchanged.

    It does **not** replace row-level fields that map onto ``ConversationalTestCase`` (e.g.
    ``scenario``, ``reference_outcome``, transcript turns); those are handled when building
    test cases, not here.

    Args:
        floeval_metrics: Floeval metrics about to be passed to
            ``create_deepeval_metric_instance()`` for a single batched run.
        rows: All conversational or agent-trace rows in that batch.

    """
    for metric in floeval_metrics:
        params = getattr(metric, "_metric_params", None)
        if not isinstance(params, dict):
            continue
        if metric.name == "topic_adherence" and not params.get("relevant_topics"):
            params["relevant_topics"] = _collect_relevant_topics(rows)
        if metric.name == "tool_use" and not params.get("available_tools"):
            params["available_tools"] = _collect_available_tools(rows)


def _default_progress(msg: str, *args: Any, extra: dict[str, Any]) -> None:
    logger.info(msg, *args, extra=extra)
    log_job_status(msg, *args, extra=extra)


def partition_deepeval_metrics(
    metrics: list[BaseMetric],
) -> tuple[list[BaseMetric], list[BaseMetric]]:
    """Split DeepEval-route metrics into LLM test-case vs conversational batches."""
    llm_route: list[BaseMetric] = []
    conversational: list[BaseMetric] = []
    for m in metrics:
        if getattr(m, "deepeval_test_case_kind", "llm") == "conversational":
            conversational.append(m)
        else:
            llm_route.append(m)
    return llm_route, conversational


def _instantiate_deepeval_for_batch(
    adapter: DeepEvalCustomMetricAdapter,
    floeval_metric: BaseMetric,
):
    """Build one DeepEval metric instance for a batched ``evaluate()`` call."""
    kind: str = getattr(floeval_metric, "deepeval_test_case_kind", "llm")
    if kind == "conversational":
        return cast(
            _SupportsConversationalDeepevalBatch, floeval_metric
        ).create_deepeval_metric_instance()
    deepeval_class = adapter.transform_metric(floeval_metric)
    return deepeval_class()


def _metrics_instances_for_batch(
    adapter: DeepEvalCustomMetricAdapter,
    floeval_metrics: list[BaseMetric],
) -> tuple[list[Any], list[BaseMetric]]:
    """Return DeepEval metric objects and parallel Floeval metric list."""
    deepeval_metric_instances: list[Any] = []
    floeval_metrics_ordered: list[BaseMetric] = []
    for floeval_metric in floeval_metrics:
        try:
            inst = _instantiate_deepeval_for_batch(adapter, floeval_metric)
            deepeval_metric_instances.append(inst)
            floeval_metrics_ordered.append(floeval_metric)
        except Exception as e:
            logger.error(
                "Failed to transform metric %s to DeepEval: %s",
                floeval_metric.name,
                e,
                exc_info=True,
            )
    return deepeval_metric_instances, floeval_metrics_ordered


def _map_evaluate_result_to_sample_results(
    *,
    row_models: Sequence[BaseModel],
    result: Any,
    deepeval_metric_instances: list[Any],
    floeval_metrics_ordered: list[BaseMetric],
) -> list[dict[str, Any]]:
    """Turn DeepEval aggregate result into Floeval sample dicts with metrics."""
    sample_results: list[dict[str, Any]] = []
    for i, sample in enumerate(row_models):
        metric_results: dict[str, Any] = {}

        test_result = (
            result.test_results[i]
            if result.test_results and i < len(result.test_results)
            else None
        )

        for j, (metric_instance, floeval_metric) in enumerate(
            zip(deepeval_metric_instances, floeval_metrics_ordered, strict=True)
        ):
            provider = "deepeval"
            metric_name = floeval_metric.name
            key = f"{provider}:{metric_name}"
            threshold = getattr(floeval_metric, "threshold", 0.5)

            try:
                metric_data = (
                    test_result.metrics_data[j]
                    if test_result
                    and test_result.metrics_data
                    and j < len(test_result.metrics_data)
                    else None
                )
                if metric_data is None:
                    raise ValueError(f"No metric data at index {j} for sample {i}")

                score = metric_data.score
                if getattr(metric_instance, "score", None) is not None:
                    score = metric_instance.score
                if score is None:
                    score = 0.0

                success = getattr(metric_data, "success", None)
                if getattr(metric_instance, "success", None) is not None:
                    success = metric_instance.success
                if success is None:
                    success = score >= threshold

                metric_results[key] = {
                    "score": score,
                    "passed": success,
                    "reason": getattr(metric_instance, "reason", None),
                    "provider": provider,
                    "metadata": {
                        "threshold": threshold,
                        "execution_provider": "deepeval",
                    },
                }
            except Exception as e:
                logger.error(
                    "DeepEval metric %s failed for sample %d: %s", key, i, e, exc_info=True
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


def _evaluate_deepeval_batch(
    *,
    all_test_cases: list[Any],
    deepeval_metric_instances: list[Any],
    floeval_metrics_ordered: list[BaseMetric],
    row_models: Sequence[BaseModel],
    emit_progress: ProgressFn,
    phase_batched: str,
    phase_done: str,
) -> list[dict[str, Any]]:
    """Run ``deepeval_evaluate`` and map results to per-sample dicts."""
    max_concurrent = min(len(all_test_cases), 10)

    emit_progress(
        "DeepEval batched evaluate() starting: metrics=%d samples=%d max_concurrent=%d",
        len(deepeval_metric_instances),
        len(all_test_cases),
        max_concurrent,
        extra={
            "phase": phase_batched,
            "eval_phase": "deepeval_async_config",
            "metric_count": len(deepeval_metric_instances),
            "sample_count": len(all_test_cases),
            "deepeval_max_concurrent": max_concurrent,
        },
    )

    t_deepeval = time.monotonic()
    try:
        result = deepeval_evaluate(
            test_cases=all_test_cases,
            metrics=deepeval_metric_instances,
            async_config=AsyncConfig(run_async=True, max_concurrent=max_concurrent),
            display_config=DisplayConfig(print_results=False, show_indicator=False),
            error_config=ErrorConfig(ignore_errors=True),
        )
    except Exception as e:
        elapsed = time.monotonic() - t_deepeval
        logger.error(
            "DeepEval batched evaluation failed after %.1fs: %s", elapsed, e, exc_info=True
        )
        log_job_status_error(
            "DeepEval evaluation FAILED after %.1fs: [%s] %s",
            elapsed,
            type(e).__name__,
            str(e)[:600],
            extra={
                "phase": "deepeval_eval_failed",
                "elapsed_s": round(elapsed, 1),
                "error_type": type(e).__name__,
                "error": str(e)[:600],
            },
        )
        return []

    deepeval_elapsed = time.monotonic() - t_deepeval
    emit_progress(
        "DeepEval batched evaluate() done in %.1fs metrics=%d samples=%d",
        deepeval_elapsed,
        len(deepeval_metric_instances),
        len(all_test_cases),
        extra={
            "phase": phase_done,
            "elapsed_s": round(deepeval_elapsed, 1),
            "metric_count": len(deepeval_metric_instances),
            "sample_count": len(all_test_cases),
        },
    )

    return _map_evaluate_result_to_sample_results(
        row_models=row_models,
        result=result,
        deepeval_metric_instances=deepeval_metric_instances,
        floeval_metrics_ordered=floeval_metrics_ordered,
    )


def run_deepeval_llm_batch(
    *,
    dataset: Dataset,
    floeval_metrics: list[BaseMetric],
    llm_config: LLMProviderConfig | None,
    emit_progress: ProgressFn | None = None,
) -> list[dict[str, Any]]:
    """Run Floeval DeepEval-route metrics on ``LLMTestCase`` rows in one batched call."""
    progress = emit_progress or _default_progress
    adapter = DeepEvalCustomMetricAdapter(llm_config)

    deepeval_metric_instances, floeval_metrics_ordered = _metrics_instances_for_batch(
        adapter, floeval_metrics
    )
    if not deepeval_metric_instances:
        return []

    all_test_cases = [adapter.transform_sample(s) for s in dataset.samples]
    return _evaluate_deepeval_batch(
        all_test_cases=all_test_cases,
        deepeval_metric_instances=deepeval_metric_instances,
        floeval_metrics_ordered=floeval_metrics_ordered,
        row_models=dataset.samples,
        emit_progress=progress,
        phase_batched="floeval_deepeval_batched",
        phase_done="floeval_deepeval_batched_done",
    )


def run_deepeval_conversational_batch(
    *,
    conversational_rows: list[ConversationalSample],
    floeval_metrics: list[BaseMetric],
    llm_config: LLMProviderConfig | None,
    emit_progress: ProgressFn | None = None,
) -> list[dict[str, Any]]:
    """Run conversational DeepEval metrics on ``ConversationalTestCase`` rows."""
    progress = emit_progress or _default_progress
    adapter = DeepEvalCustomMetricAdapter(llm_config)
    _hydrate_conversational_metric_params(floeval_metrics, conversational_rows)

    deepeval_metric_instances, floeval_metrics_ordered = _metrics_instances_for_batch(
        adapter, floeval_metrics
    )
    if not deepeval_metric_instances:
        return []

    all_test_cases = [conversational_sample_to_deepeval(s) for s in conversational_rows]
    return _evaluate_deepeval_batch(
        all_test_cases=all_test_cases,
        deepeval_metric_instances=deepeval_metric_instances,
        floeval_metrics_ordered=floeval_metrics_ordered,
        row_models=conversational_rows,
        emit_progress=progress,
        phase_batched="floeval_deepeval_conversational_batched",
        phase_done="floeval_deepeval_conversational_batched_done",
    )


def run_deepeval_conversational_batch_for_agent_samples(
    *,
    agent_rows: list[AgentSample],
    floeval_metrics: list[BaseMetric],
    llm_config: LLMProviderConfig | None,
    emit_progress: ProgressFn | None = None,
) -> list[dict[str, Any]]:
    """Conversational DeepEval batch where rows are ``AgentSample`` (trace-backed)."""
    progress = emit_progress or _default_progress
    adapter = DeepEvalCustomMetricAdapter(llm_config)
    _hydrate_conversational_metric_params(floeval_metrics, agent_rows)

    deepeval_metric_instances, floeval_metrics_ordered = _metrics_instances_for_batch(
        adapter, floeval_metrics
    )
    if not deepeval_metric_instances:
        return []

    all_test_cases = [agent_sample_to_conversational_test_case(s) for s in agent_rows]
    return _evaluate_deepeval_batch(
        all_test_cases=all_test_cases,
        deepeval_metric_instances=deepeval_metric_instances,
        floeval_metrics_ordered=floeval_metrics_ordered,
        row_models=agent_rows,
        emit_progress=progress,
        phase_batched="floeval_deepeval_conversational_agent_batched",
        phase_done="floeval_deepeval_conversational_agent_batched_done",
    )
