"""
Helper functions for manual trace capture.

Inspired by DeepEval's update_current_span() pattern.
"""

import logging

from floeval.config.schemas.io.agent_dataset import ToolCall
from floeval.utils.agent_trace.trace_context import get_current_trace

logger = logging.getLogger(__name__)


def log_turn(
    output: str,
    tool_calls: list[ToolCall] | None = None,
) -> None:
    """
    One-line helper to manually log an AI turn.

    Inspired by DeepEval's update_current_span() but simpler.
    Only works when trace context is active (inside @capture_trace
    or AgentEvaluation).

    Usage:
        @capture_trace
        def my_agent(user_input: str):
            response = anthropic.messages.create(...)
            log_turn(response.content[0].text)  # One line!
            return response.content[0].text

    Args:
        output: The AI's response text
        tool_calls: Optional list of tool calls made
    """
    trace = get_current_trace()

    if trace is None:
        logger.warning(
            "log_turn() called outside trace context. "
            "Use @capture_trace decorator or run inside AgentEvaluation."
        )
        return

    trace.log_ai_turn(output, tool_calls)
    logger.debug("Logged AI turn: %s...", (output[:50] if len(output) > 50 else output))


def log_tool_result(tool_name: str, result: str) -> None:
    """
    Log a tool execution result.

    Usage:
        @capture_trace
        def my_agent(user_input: str):
            tool_output = execute_search(query)
            log_tool_result("search", tool_output)  # Log tool result

            response = generate_response(tool_output)
            log_turn(response)  # Log AI response
            return response

    Args:
        tool_name: Name of the tool that was executed
        result: Tool's output
    """
    trace = get_current_trace()

    if trace is None:
        logger.warning(
            "log_tool_result() called outside trace context. "
            "Use @capture_trace decorator or run inside AgentEvaluation."
        )
        return

    trace.log_tool_result(tool_name, result)
    logger.debug(
        "Logged tool result from %s: %s...",
        tool_name,
        (result[:50] if len(result) > 50 else result),
    )


def create_span(
    input: str,
    output: str,
    tool_calls: list[ToolCall] | None = None,
) -> None:
    """
    DeepEval-style span creation API.

    Alternative to log_turn() that matches DeepEval's update_current_span() signature.

    Usage:
        @capture_trace
        def my_agent(user_input: str):
            response = my_llm(user_input)
            create_span(input=user_input, output=response)  # DeepEval style
            return response

    Args:
        input: The input to this span (usually user_input)
        output: The output from this span
        tool_calls: Optional tool calls made
    """
    log_turn(output, tool_calls)
