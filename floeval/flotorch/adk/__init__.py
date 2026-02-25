"""FloTorch ADK - agent builder and utilities for Mode 4 evaluation.

Mode 4 uses gateway agents via create_flotorch_runner (floeval.flotorch).
build_simple_agent is for local/config-based agents (future Mode 3).
"""

from floeval.flotorch.adk.agent import (
    FlotorchADKAgent,
    build_simple_agent,
)
from floeval.flotorch.adk.llm import FlotorchADKLLM
from floeval.flotorch.adk.utils.adk_utils import process_session_events

__all__ = [
    "FlotorchADKAgent",
    "FlotorchADKLLM",
    "build_simple_agent",
    "process_session_events",
]

try:
    from floeval.flotorch.adk.memory import (
        FlotorchADKVectorMemoryService,
        FlotorchMemoryService,
    )

    __all__ = list(__all__) + [
        "FlotorchADKVectorMemoryService",
        "FlotorchMemoryService",
    ]
except ImportError:
    pass
