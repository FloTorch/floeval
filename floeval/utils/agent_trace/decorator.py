"""
DeepEval-style decorator for universal agent support.

Inspired by deepeval's @observe pattern but adapted for floeval.
"""

import asyncio
import functools
import logging
from typing import Any, Callable

from floeval.config.schemas.io.agent_dataset import AgentTrace
from floeval.utils.agent_trace.trace_context import (
    TraceContext,
    clear_current_trace,
    get_current_trace,
    set_current_trace,
)

logger = logging.getLogger(__name__)


def capture_trace(func: Callable | None = None, *, name: str | None = None) -> Callable:
    """
    Decorator for capturing agent execution traces.

    Works like DeepEval's @observe but adapted for floeval:
    - Creates trace context automatically if not already present
    - Allows manual span updates via log_turn()
    - Converts string results to AgentTrace automatically
    - Works standalone OR inside AgentEvaluation
    - Supports both sync and async agents

    Usage:
        # Simple case
        @capture_trace
        def my_agent(user_input: str) -> str:
            response = my_llm(user_input)
            log_turn(response)  # Optional manual capture
            return response

        # Async agent
        @capture_trace
        async def my_async_agent(user_input: str) -> str:
            response = await my_llm(user_input)
            log_turn(response)
            return response

        # With name
        @capture_trace(name="anthropic_agent")
        def my_agent(user_input: str) -> str:
            return anthropic_call(user_input)

    Args:
        func: Function to decorate (when used as @capture_trace)
        name: Optional display name for this span

    Returns:
        Decorated function that returns AgentTrace
    """

    def decorator(fn: Callable) -> Callable:
        display_name = name or fn.__name__

        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(
                user_input: str, *args: Any, **kwargs: Any
            ) -> AgentTrace:
                existing_trace = get_current_trace()

                if existing_trace is not None:
                    result = await fn(user_input, *args, **kwargs)
                    if isinstance(result, AgentTrace):
                        return result
                    return existing_trace.to_trace()

                trace_ctx = TraceContext(user_input=user_input)
                trace_ctx.metadata["component_name"] = display_name
                set_current_trace(trace_ctx)

                try:
                    logger.debug("Starting trace capture for: %s", display_name)
                    result = await fn(user_input, *args, **kwargs)

                    if isinstance(result, AgentTrace):
                        return result
                    if isinstance(result, str):
                        if len(trace_ctx.messages) > 1:
                            trace_ctx.final_response = result
                            return trace_ctx.to_trace()
                        trace_ctx.log_ai_turn(result)
                        return trace_ctx.to_trace()
                    logger.warning(
                        "Function %s returned %s. Expected AgentTrace or str. Creating minimal trace.",
                        display_name,
                        type(result),
                    )
                    trace_ctx.log_ai_turn(str(result))
                    return trace_ctx.to_trace()
                finally:
                    clear_current_trace()

            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(user_input: str, *args: Any, **kwargs: Any) -> AgentTrace:
            existing_trace = get_current_trace()

            if existing_trace is not None:
                result = fn(user_input, *args, **kwargs)
                if isinstance(result, AgentTrace):
                    return result
                return existing_trace.to_trace()

            trace_ctx = TraceContext(user_input=user_input)
            trace_ctx.metadata["component_name"] = display_name
            set_current_trace(trace_ctx)

            try:
                logger.debug("Starting trace capture for: %s", display_name)
                result = fn(user_input, *args, **kwargs)

                if isinstance(result, AgentTrace):
                    return result
                if isinstance(result, str):
                    if len(trace_ctx.messages) > 1:
                        trace_ctx.final_response = result
                        return trace_ctx.to_trace()
                    trace_ctx.log_ai_turn(result)
                    return trace_ctx.to_trace()
                logger.warning(
                    "Function %s returned %s. Expected AgentTrace or str. Creating minimal trace.",
                    display_name,
                    type(result),
                )
                trace_ctx.log_ai_turn(str(result))
                return trace_ctx.to_trace()
            finally:
                clear_current_trace()

        return sync_wrapper

    if func is None:
        return decorator
    return decorator(func)
