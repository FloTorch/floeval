"""Session service for FloTorch Mode 4 evaluation.

Uses InMemorySessionService from google.adk for eval (no gateway/persistence).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google.adk.sessions.in_memory_session_service import InMemorySessionService


def get_in_memory_session_service() -> "InMemorySessionService":
    """Return InMemorySessionService for eval runs."""
    from google.adk.sessions.in_memory_session_service import InMemorySessionService

    return InMemorySessionService()
