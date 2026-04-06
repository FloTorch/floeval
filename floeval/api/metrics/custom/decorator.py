"""Custom metric decorator for function-based metrics."""

from __future__ import annotations

import asyncio
import inspect
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from floeval.api.metrics.base import BaseMetric, MetricResult
from floeval.api.metrics.registry import MetricRegistry
from floeval.config.schemas.io.dataset import Sample
from floeval.config.schemas.io.llm import OpenAIProviderConfig

from .context import MetricContext
from .llm_helper import SimpleLLMHelper

logger = logging.getLogger(__name__)


class FunctionBasedMetric(BaseMetric):
    """Module-level metric wrapping a user-decorated function.

    Supports both sync and async user functions.
    Sync evaluate(): never touches an event loop; async user func uses
    context-manager ThreadPoolExecutor (no stored executor).
    Async aevaluate(): awaits natively, no executor stored as state.
    """

    def __init__(
        self,
        user_func: Callable[..., Any],
        metric_name: str,
        threshold: float,
        execute_via: str | None,
        metadata_params: dict[str, Any],
        sig: inspect.Signature,
        is_async: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=metric_name, **kwargs)
        self._user_func = user_func
        self._threshold = threshold
        self.execute_via = execute_via
        self._metadata = metadata_params
        self._sig = sig
        self._is_async = is_async
        self.provider = "custom"
        self.threshold = threshold
        self.llm_config: OpenAIProviderConfig | None = kwargs.get("llm_config")
        self.chat_model: str | None = kwargs.get("chat_model")
        self.extra_headers: dict[str, str] = dict(kwargs.get("extra_headers") or {})

    def evaluate(self, sample: Any, **kwargs: Any) -> MetricResult:
        """Execute metric synchronously.

        If user_func is async, runs it in a fresh thread with a fresh event loop.
        ThreadPoolExecutor used as context manager — shutdown(wait=True) guaranteed.
        """
        args = self._prepare_args(sample)
        try:
            if self._is_async:
                result = self._run_async_in_clean_thread(args)
            else:
                result = self._user_func(**args)
            return self._normalize_result(result)
        except Exception as e:  # noqa: BLE001
            logger.error("FunctionBasedMetric %s failed: %s", self.name, e, exc_info=True)
            return MetricResult(
                score=None,
                metadata={
                    "error": str(e),
                    "passed": False,
                    "threshold": self._threshold,
                    **self._metadata,
                },
            )

    async def aevaluate(self, sample: Any, **kwargs: Any) -> MetricResult:
        """Execute metric asynchronously.

        If user_func is async, awaits it natively. If user_func is sync,
        offloads to the running loop's default executor.
        """
        args = self._prepare_args(sample)
        try:
            if self._is_async:
                result = await self._user_func(**args)
            else:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, lambda: self._user_func(**args))
            return self._normalize_result(result)
        except Exception as e:  # noqa: BLE001
            logger.error(
                "FunctionBasedMetric %s failed (async): %s",
                self.name,
                e,
                exc_info=True,
            )
            return MetricResult(
                score=None,
                metadata={
                    "error": str(e),
                    "passed": False,
                    "threshold": self._threshold,
                    **self._metadata,
                },
            )

    def _run_async_in_clean_thread(self, args: dict[str, Any]) -> Any:
        """Run an async user_func from sync context with no impact on any existing loop.

        Uses ThreadPoolExecutor as context manager: the executor is created,
        used for exactly one task, then destroyed (shutdown wait=True).
        No executor is ever stored as instance state.
        """
        func = self._user_func

        def _target() -> Any:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(func(**args))
            finally:
                loop.close()

        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(_target).result()

    def _prepare_args(self, sample: Sample) -> dict[str, Any]:
        """Map sample fields to the user function's declared parameters."""
        args: dict[str, Any] = {}
        for param_name in self._sig.parameters:
            if param_name in ("response", "answer", "actual_output"):
                args[param_name] = getattr(sample, "llm_response", "") or ""
            elif param_name in ("question", "input", "query"):
                args[param_name] = getattr(sample, "user_input", "") or ""
            elif param_name == "contexts":
                args[param_name] = getattr(sample, "contexts", []) or []
            elif param_name == "context":
                args[param_name] = MetricContext(sample=sample, llm_config=self.llm_config)
            elif param_name == "llm":
                if not self.llm_config or not isinstance(self.llm_config, OpenAIProviderConfig):
                    raise ValueError(
                        f"Metric '{self.name}' requires 'llm' parameter but no valid "
                        "OpenAIProviderConfig was provided."
                    )
                args[param_name] = SimpleLLMHelper(
                    self.llm_config,
                    chat_model=self.chat_model,
                    extra_headers=self.extra_headers or None,
                )
            elif param_name == "sample":
                args[param_name] = sample
        return args

    def _normalize_result(self, result: Any) -> MetricResult:
        """Convert user function return to MetricResult."""
        if isinstance(result, MetricResult):
            return result
        if isinstance(result, dict):
            score = result.get("score")
            meta = dict(result.get("metadata", {}))
            meta.update(
                {
                    "passed": ((score or 0) >= self._threshold if score is not None else False),
                    "threshold": self._threshold,
                    **self._metadata,
                }
            )
            return MetricResult(score=score, metadata=meta)
        if isinstance(result, (int, float)):
            score = float(result)
            return MetricResult(
                score=score,
                metadata={
                    "passed": score >= self._threshold,
                    "threshold": self._threshold,
                    **self._metadata,
                },
            )
        raise TypeError(
            f"Metric '{self.name}' returned {type(result).__name__}. "
            "Expected float, dict, or MetricResult."
        )


def custom_metric(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    threshold: float = 0.5,
    execute_via: str | None = None,
    **metadata: Any,
) -> Callable[..., Any]:
    """Decorator that registers a function as a custom FloEval metric.

    Supports both @custom_metric and @custom_metric(name="x", threshold=0.7).

    Args:
        func: Decorated function (when used without parentheses).
        name: Metric name. Defaults to func.__name__.
        threshold: Pass/fail threshold (0.0-1.0).
        execute_via: Execution backend ('ragas', 'deepeval', or None).
        **metadata: Extra key-value pairs stored in MetricResult.metadata.
    """

    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        metric_name = name or f.__name__
        sig = inspect.signature(f)
        is_async = inspect.iscoroutinefunction(f)

        def factory(**kw: Any) -> FunctionBasedMetric:
            return FunctionBasedMetric(
                user_func=f,
                metric_name=metric_name,
                threshold=threshold,
                execute_via=execute_via,
                metadata_params=metadata,
                sig=sig,
                is_async=is_async,
                **kw,
            )

        MetricRegistry().register("custom", metric_name, factory, allow_override=True)
        return f

    return decorator if func is None else decorator(func)
