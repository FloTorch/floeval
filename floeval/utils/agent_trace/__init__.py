"""
Agent trace utilities for floeval.

Provides DeepEval-style decorator and helpers for universal agent support.
"""

from floeval.utils.agent_trace.decorator import capture_trace
from floeval.utils.agent_trace.helpers import create_span, log_tool_result, log_turn
from floeval.utils.agent_trace.langchain_adapter import wrap_langchain_agent
from floeval.utils.agent_trace.trace_collector import TraceCollector
from floeval.utils.agent_trace.trace_context import (
    TraceContext,
    clear_current_trace,
    get_current_trace,
    set_current_trace,
)

__all__ = [
    # Simple API (recommended)
    "capture_trace",
    "log_turn",
    "log_tool_result",
    "create_span",
    "wrap_langchain_agent",
    # Advanced API (for power users)
    "TraceCollector",
    "TraceContext",
    "get_current_trace",
    "set_current_trace",
    "clear_current_trace",
]
