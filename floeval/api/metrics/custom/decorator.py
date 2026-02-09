"""Custom metric decorator for function-based metrics."""

import asyncio
import inspect
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from floeval.api.metrics.base import BaseMetric, MetricResult
from floeval.api.metrics.registry import MetricRegistry

# Module-level logger
logger = logging.getLogger(__name__)

# Lazy imports (avoid circular imports):
# - MetricContext: From same package; lazy to avoid import cycles
# - SimpleLLMHelper: From same package; lazy to avoid import cycles


def custom_metric(
    func: Callable | None = None,
    *,
    name: str | None = None,
    threshold: float = 0.5,
    execute_via: str | None = None,
    **metadata
) -> Callable:
    """
    Convert function to custom metric.
    
    Automatically maps function parameters: response/answer → llm_response,
    question/input → user_input, contexts → contexts, context → MetricContext,
    sample → Sample, llm → SimpleLLMHelper.
    
    Args:
        func: Function to convert
        name: Metric name (defaults to function name)
        threshold: Pass/fail threshold (0-1)
        execute_via: Execution provider (ragas/deepeval/None)
        **metadata: Additional metadata
    """
    
    def decorator(f: Callable) -> Callable:
        metric_name = name or f.__name__
        
        sig = inspect.signature(f)
        is_async = inspect.iscoroutinefunction(f)
        
        metric_class = _generate_function_metric_class(
            user_func=f,
            metric_name=metric_name,
            threshold=threshold,
            execute_via=execute_via,
            metadata_params=metadata,
            sig=sig,
            is_async=is_async
        )
        
        MetricRegistry().register("custom", metric_name, metric_class, allow_override=True)
        return f
    
    # Handle both @custom_metric and @custom_metric(...)
    if func is None:
        return decorator
    else:
        return decorator(func)


def _generate_function_metric_class(
    user_func: Callable,
    metric_name: str,
    threshold: float,
    execute_via: str | None,
    metadata_params: dict[str, Any],
    sig: inspect.Signature,
    is_async: bool
) -> type[BaseMetric]:
    """Generate BaseMetric subclass from function."""
    
    class FunctionBasedMetric(BaseMetric):
        """Generated metric class for function-based custom metric."""
        
        def __init__(self, **kwargs):
            super().__init__(name=metric_name, **kwargs)
            self.user_func = user_func
            self.threshold = threshold
            self.execute_via = execute_via
            self.metadata_params = metadata_params
            self.sig = sig
            self.is_async_func = is_async
            self.provider = "custom"
            self.gateway_config = kwargs.get('gateway_config')
            self.llm_model = kwargs.get('llm_model')
            self._executor = ThreadPoolExecutor(max_workers=1)
        
        def evaluate(self, sample: Any, **kwargs: Any) -> MetricResult:
            """Execute metric synchronously. Async functions run in thread executor."""
            args = self._prepare_args(sample)
            try:
                if self.is_async_func:
                    future = self._executor.submit(
                        self._run_async_user_in_thread,
                        args,
                    )
                    result = future.result()
                else:
                    result = self.user_func(**args)
                return self._normalize_result(result)
            except Exception as e:
                logger.error(f"Metric {self.name} failed for sample: {e}", exc_info=True)
                return MetricResult(
                    score=None,
                    metadata={
                        "error": str(e),
                        "passed": False,
                        "threshold": self.threshold,
                        **self.metadata_params
                    }
                )
        
        def _run_async_user_in_thread(self, args: dict[str, Any]) -> Any:
            """Run async user function in isolated thread with its own event loop."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.user_func(**args))
            finally:
                loop.close()
        
        async def aevaluate(self, sample: Any, **kwargs: Any) -> MetricResult:
            """Execute metric asynchronously. Sync functions run in executor."""
            args = self._prepare_args(sample)
            
            try:
                if self.is_async_func:
                    # Async function - await directly
                    result = await self.user_func(**args)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, lambda: self.user_func(**args))
                
                return self._normalize_result(result)
            except Exception as e:
                logger.error(f"Metric {self.name} failed for sample: {e}", exc_info=True)
                return MetricResult(
                    score=None,
                    metadata={
                        "error": str(e),
                        "passed": False,
                        "threshold": self.threshold,
                        **self.metadata_params
                    }
                )
        
        def _prepare_args(self, sample: Any) -> dict[str, Any]:
            """Map sample fields to function parameters."""
            args: dict[str, Any] = {}
            
            for param_name, param in self.sig.parameters.items():
                if param_name in ["response", "answer", "actual_output"]:
                    args[param_name] = getattr(sample, "llm_response", "") or ""
                elif param_name in ["question", "input", "query"]:
                    args[param_name] = getattr(sample, "user_input", "") or ""
                elif param_name == "contexts":
                    args[param_name] = getattr(sample, "contexts", []) or []
                elif param_name == "context":
                    from .context import MetricContext
                    gateway_config = getattr(self, 'gateway_config', None)
                    args[param_name] = MetricContext(sample=sample, gateway_config=gateway_config)
                elif param_name == "llm":
                    from .llm_helper import SimpleLLMHelper
                    gateway_config = getattr(self, 'gateway_config', None)
                    llm_model = getattr(self, 'llm_model', None)
                    args[param_name] = SimpleLLMHelper(gateway_config, llm_model)
                elif param_name == "sample":
                    args[param_name] = sample
            
            return args
        
        def _normalize_result(self, result: Any) -> MetricResult:
            """Convert function result to MetricResult."""
            if isinstance(result, MetricResult):
                return result
            
            elif isinstance(result, dict):
                score = result.get("score")
                metadata = result.get("metadata", {})
                metadata.update({
                    "passed": score >= self.threshold if score is not None else False,
                    "threshold": self.threshold,
                    **self.metadata_params
                })
                return MetricResult(score=score, metadata=metadata)
            
            elif isinstance(result, (int, float)):
                score = float(result)
                return MetricResult(
                    score=score,
                    metadata={
                        "passed": score >= self.threshold,
                        "threshold": self.threshold,
                        **self.metadata_params
                    }
                )
            
            else:
                raise ValueError(
                    f"Invalid metric result type: {type(result)}. "
                    f"Expected: float, dict, or MetricResult"
                )
    
    # Set class metadata
    FunctionBasedMetric.__name__ = f"{metric_name}_metric"
    FunctionBasedMetric.__qualname__ = f"{metric_name}_metric"
    FunctionBasedMetric.__doc__ = user_func.__doc__
    
    return FunctionBasedMetric
