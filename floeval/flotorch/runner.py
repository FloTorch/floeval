"""FloTorch ADK runner for Mode 4 agent evaluation.

If you see "Error on session runner task: Attempted to exit cancel scope in a
different task than it was entered in" — this comes from google-adk when the
agent uses MCP tools. See MODE4_FLOW_ANALYSIS.md for flow and root cause.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING

from floeval.config.schemas.io.agent_dataset import (
    AgentSample,
    AgentTrace,
    PartialAgentSample,
    _to_display_str,
)
from floeval.flotorch.adk.utils.adk_utils import process_session_events

if TYPE_CHECKING:
    from google.adk.agents import BaseAgent
    from google.genai import types


def _make_user_content(text: str) -> "types.Content":
    """Create Content for user message."""
    from google.genai import types

    return types.Content(
        role="user",
        parts=[types.Part.from_text(text=text)],
    )


class FloTorchRunner:
    """Run ADK agent with InMemorySessionService and extract traces.

    Use with AgentEvaluation as agent_runner for Mode 4.
    """

    APP_NAME = "floeval_agent_eval"
    USER_ID = "eval_user"

    def __init__(self, agent: "BaseAgent"):
        """Initialize runner with ADK agent (e.g. LlmAgent)."""
        self.agent = agent
        self._session_service = None
        self._runner = None

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
        return asyncio.run(self.arun(user_input))

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

        async for _ in self._runner.run_async(
            user_id=self.USER_ID,
            session_id=session_id,
            new_message=content,
        ):
            pass

        session = await self._session_service.get_session(
            app_name=self.APP_NAME,
            user_id=self.USER_ID,
            session_id=session_id,
        )
        if session is None or not session.events:
            return AgentTrace.from_simple_response(user_input, "")

        messages = process_session_events(session.events)
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
        """Run agent on each partial sample (async, same event loop — avoids ADK cleanup errors)."""
        full = []
        for partial in partial_samples:
            text = _to_display_str(partial.user_input)
            trace = await self.arun(text)
            full.append(AgentSample.from_partial(partial, trace))
        return full
