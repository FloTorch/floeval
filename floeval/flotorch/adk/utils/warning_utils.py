"""Warning suppression utilities for FloTorch ADK components."""

import io
import logging
import os
import sys
import warnings


def suppress_adk_warnings() -> None:
    """Suppress ADK, MCP, and async related warnings."""
    warnings.filterwarnings("ignore", message=".*EXPERIMENTAL.*BaseAuthenticatedTool.*")
    warnings.filterwarnings("ignore", message=".*auth_config.*")
    warnings.filterwarnings("ignore", message=".*authentication.*")
    warnings.filterwarnings("ignore", message=".*FunctionTool.*")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)


def suppress_adk_logging() -> None:
    """Suppress noisy logging from ADK components."""
    logging.getLogger("mcp").setLevel(logging.CRITICAL)
    logging.getLogger("anyio").setLevel(logging.CRITICAL)
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)
    logging.getLogger("streamable_http").setLevel(logging.CRITICAL)


class SuppressOutput:
    """Context manager to suppress stdout/stderr."""

    def __init__(self) -> None:
        self._original_stdout: object = None
        self._original_stderr: object = None
        self._devnull: io.TextIOWrapper | io.StringIO = io.StringIO()

    def __enter__(self) -> "SuppressOutput":
        try:
            self._devnull = open(os.devnull, "w")
        except OSError:
            self._devnull = io.StringIO()
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = self._devnull
        sys.stderr = self._devnull
        return self

    def __exit__(self, *args: object) -> None:
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        try:
            self._devnull.close()
        except Exception:
            pass


def setup_adk_environment() -> None:
    """One-time setup to suppress ADK-related warnings and noise."""
    suppress_adk_warnings()
    suppress_adk_logging()


if not os.getenv("FLOTORCH_NO_AUTO_SUPPRESS"):
    setup_adk_environment()
