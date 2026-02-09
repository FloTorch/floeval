"""
Custom metric decorator for function-based metrics.

Applies Decorator Pattern to wrap user functions in BaseMetric classes.
Supports both sync and async user functions; compute() and acompute() both work.
"""

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
    Decorator to convert functions into custom metrics.
    
    Design Patterns:
    - Decorator: Wraps function in metric class
    - Factory: Generates metric class dynamically
    - Registry: Auto-registers metric
    
    PARAMETER MAPPING:
    -----------------
    The decorator automatically maps function parameters to sample fields:
    
    Standard Mappings:
        - response, answer, actual_output → sample.inputs["answer"] or ["response"]
        - question, input, query → sample.inputs["question"] or ["input"]
        - contexts → sample.inputs["contexts"]
        - context (MetricContext) → Provides full sample access
        - sample (Sample) → Raw sample object
    
    Args:
        func: Function to convert (when used as @custom_metric)
        name: Metric name (defaults to function name)
        threshold: Pass/fail threshold (0-1)
        execute_via: Provider for execution (ragas/deepeval/None for standalone)
        **metadata: Additional metadata to include in results
    
    Returns:
        Original function (for testing) or decorator
    
    Examples:
        # Simple: Automatic mapping
        @custom_metric
        def politeness(response: str) -> float:
            return len(response) / 100
        
        # With options
        @custom_metric(threshold=0.8, execute_via="ragas")
        def custom_faithfulness(response: str, contexts: list) -> float:
            return faithfulness_score(response, contexts)
        
        # Advanced: Custom field access
        @custom_metric
        def my_metric(response: str, context: MetricContext) -> float:
            custom_field = context.get_input("my_field")
            return compute(response, custom_field)
    """
    
    def decorator(f: Callable) -> Callable:
        metric_name = name or f.__name__
        
        # Inspect function signature
        sig = inspect.signature(f)
        is_async = inspect.iscoroutinefunction(f)
        
        # Generate metric class (Factory Pattern)
        metric_class = _generate_function_metric_class(
            user_func=f,
            metric_name=metric_name,
            threshold=threshold,
            execute_via=execute_via,
            metadata_params=metadata,
            sig=sig,
            is_async=is_async
        )
        
        # Auto-register (Registry Pattern)
        MetricRegistry().register("custom", metric_name, metric_class, allow_override=True)
        
        # Return original function for testability
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
    """
    Factory: Generate BaseMetric subclass from function.
    
    Design Principles:
    - Single Responsibility: Only generates class, doesn't execute
    - Open/Closed: Extensible via inheritance
    - Liskov Substitution: Generated class is valid BaseMetric
    """
    
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
            # Store gateway_config and llm_model from kwargs (injected by Evaluation)
            self.gateway_config = kwargs.get('gateway_config')
            self.llm_model = kwargs.get('llm_model')
            # One worker for running async user functions in sync context (isolated thread)
            self._executor = ThreadPoolExecutor(max_workers=1)
        
        def compute(self, sample: Any, **kwargs: Any) -> MetricResult:
            """
            Synchronous execution of custom metric function.
            
            Sync user functions: called directly.
            Async user functions: run in isolated thread via ThreadPoolExecutor (same pattern as SimpleLLMHelper).
            Both work with Evaluation.run().
            """
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
        
        async def acompute(self, sample: Any, **kwargs: Any) -> MetricResult:
            """
            Asynchronous execution of custom metric function.
            
            Contract:
            - Async user functions: Await directly
            - Sync user functions: Run in executor to avoid blocking event loop
            - LLM-based metrics: Use agenerate() inside async function
            
            Args:
                sample: Sample object containing inputs and ground truth.
                **kwargs: Additional keyword arguments (typically unused).
            
            Returns:
                MetricResult: Normalized result with score and metadata.
            """
            args = self._prepare_args(sample)
            
            try:
                if self.is_async_func:
                    # Async function - await directly
                    result = await self.user_func(**args)
                else:
                    # Sync function - run in executor to avoid blocking event loop
                    # This allows sync metrics to work in async evaluation contexts
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
            """
            Parameter mapping: Sample fields → Function parameters.
            
            Design: Convention over Configuration
            - Standard parameter names mapped automatically
            - Extensible for new parameter types
            
            Mappings:
            - response/answer/actual_output → sample.llm_response
            - question/input/query → sample.user_input
            - contexts → sample.contexts
            - context → MetricContext(sample)
            - sample → sample (full object)
            - llm → LLMHelper instance (injected from gateway_config)
            """
            args: dict[str, Any] = {}
            
            for param_name, param in self.sig.parameters.items():
                # Response parameter
                if param_name in ["response", "answer", "actual_output"]:
                    args[param_name] = getattr(sample, "llm_response", "") or ""
                
                # Question parameter
                elif param_name in ["question", "input", "query"]:
                    args[param_name] = getattr(sample, "user_input", "") or ""
                
                # Contexts parameter
                elif param_name == "contexts":
                    args[param_name] = getattr(sample, "contexts", []) or []
                
                # Context object parameter
                elif param_name == "context":
                    # Lazy import: From same package; lazy to avoid import cycles
                    from .context import MetricContext
                    gateway_config = getattr(self, 'gateway_config', None)
                    args[param_name] = MetricContext(
                        sample=sample,
                        gateway_config=gateway_config
                    )
                
                # LLM helper parameter
                elif param_name == "llm":
                    # Lazy import: From same package; lazy to avoid import cycles
                    from .llm_helper import SimpleLLMHelper
                    gateway_config = getattr(self, 'gateway_config', None)
                    llm_model = getattr(self, 'llm_model', None)
                    args[param_name] = SimpleLLMHelper(gateway_config, llm_model)
                
                # Full sample parameter
                elif param_name == "sample":
                    args[param_name] = sample
            
            return args
        
        def _normalize_result(self, result: Any) -> MetricResult:
            """
            Convert function result to MetricResult.
            
            Handles multiple return types:
            - MetricResult: Passthrough (already normalized)
            - dict: Must contain "score" key; optional "metadata" dict merged
            - float/int: Converted to score, metadata includes threshold/passed
            
            Args:
                result: Function return value (MetricResult, dict, or numeric).
            
            Returns:
                MetricResult: Normalized result with score and metadata.
            
            Raises:
                ValueError: If result type is not supported.
            """
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
