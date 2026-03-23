"""Utilities for running async coroutines from sync code."""

import asyncio
import concurrent.futures
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


def run_coroutine_sync(coro_factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """Run a coroutine factory from sync code.

    If no event loop is active in the current thread, this uses ``asyncio.run``.
    If a loop is already running (for example in Jupyter), it runs the coroutine
    in a worker thread and blocks for completion.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())

    def _run_in_thread() -> T:
        return asyncio.run(coro_factory())

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_run_in_thread).result()
