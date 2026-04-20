"""Run an ADK agent and build AgentTrace from session events (Mode 4)."""

from __future__ import annotations

import logging
import traceback
import uuid
from typing import TYPE_CHECKING

from floeval.config.schemas.io.agent_dataset import (
    AgentSample,
    AgentTrace,
    PartialAgentSample,
    _to_display_str,
)
from floeval.flotorch.adk.utils.adk_utils import process_session_events
from floeval.utils.asyncio_compat import run_coroutine_sync

if TYPE_CHECKING:
    from google.adk.agents import BaseAgent
    from google.genai import types

logger = logging.getLogger(__name__)


def _make_user_content(text: str) -> "types.Content":
    """Create Content for user message."""
    from google.genai import types

    return types.Content(
        role="user",
        parts=[types.Part.from_text(text=text)],
    )


class FloTorchRunner:
    """In-memory ADK runner; trace from session.events after each run."""

    APP_NAME = "floeval_agent_eval"
    USER_ID = "eval_user"

    def __init__(self, agent: "BaseAgent"):
        """Initialize runner with ADK agent (e.g. LlmAgent)."""
        self.agent = agent
        self._session_service = None
        self._runner = None

    async def aclose(self) -> None:
        """Close MCP tool transports (call after batch eval)."""
        if self.agent and hasattr(self.agent, "tools"):
            for tool in self.agent.tools or []:
                if hasattr(tool, "close"):
                    try:
                        await tool.close()
                    except Exception as exc:
                        logger.debug("Error closing tool %s: %s", type(tool).__name__, exc)
        self._runner = None
        self._session_service = None

    def _ensure_runner(self):
        """Lazy-initialize Runner with InMemorySessionService."""
        if self._runner is not None:
            return

        from google.adk.runners import Runner
        from google.adk.sessions.in_memory_session_service import (
            InMemorySessionService,
        )

        self._session_service = InMemorySessionService()
        self._runner = Runner(
            app_name=self.APP_NAME,
            agent=self.agent,
            session_service=self._session_service,
        )

    def run(self, user_input: str) -> AgentTrace:
        """Run agent once and return trace (sync)."""
        return run_coroutine_sync(lambda: self.arun(user_input))

    async def arun(self, user_input: str) -> AgentTrace:
        """Run agent once and return trace (async)."""
        self._ensure_runner()

        session = await self._session_service.create_session(
            app_name=self.APP_NAME,
            user_id=self.USER_ID,
            session_id=str(uuid.uuid4()),
        )
        session_id = session.id

        content = _make_user_content(user_input)
        print(f"[floeval-debug] FloTorchRunner.arun start session={session_id}", flush=True)

        try:
            async for _ in self._runner.run_async(
                user_id=self.USER_ID,
                session_id=session_id,
                new_message=content,
            ):
                pass
        except Exception:
            print("[floeval-debug] FloTorchRunner.run_async raised:", flush=True)
            traceback.print_exc()
            raise

        session = await self._session_service.get_session(
            app_name=self.APP_NAME,
            user_id=self.USER_ID,
            session_id=session_id,
        )
        if session is None or not session.events:
            print(f"[floeval-debug] FloTorchRunner.arun no session events session={session_id}", flush=True)
            return AgentTrace.from_simple_response(user_input, "")

        messages = process_session_events(session.events)
        print(
            f"[floeval-debug] FloTorchRunner.arun done events={len(session.events)} trace_msgs={len(messages)}",
            flush=True,
        )
        return AgentTrace.from_messages(messages)

    def run_on_dataset(self, partial_samples: list[PartialAgentSample]) -> list[AgentSample]:
        """Run agent on each partial sample and return full samples (sync)."""
        full = []
        for partial in partial_samples:
            text = _to_display_str(partial.user_input)
            trace = self.run(text)
            full.append(AgentSample.from_partial(partial, trace))
        return full

    async def run_on_dataset_async(
        self, partial_samples: list[PartialAgentSample]
    ) -> list[AgentSample]:
        """Async batch run (one event loop for all samples)."""
        full = []
        for partial in partial_samples:
            text = _to_display_str(partial.user_input)
            trace = await self.arun(text)
            full.append(AgentSample.from_partial(partial, trace))
        return full
