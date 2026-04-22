"""Run an ADK agent and build AgentTrace from session events (Mode 4)."""

from __future__ import annotations

import logging
import time
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
from floeval.utils.agent_trace.trace_context import (
    collect_session_token_usage,
    start_session_token_tracking,
)
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
        logger.debug("FloTorchRunner.arun start session=%s", session_id)

        start_session_token_tracking()
        start_time = time.time()
        try:
            async for _ in self._runner.run_async(
                user_id=self.USER_ID,
                session_id=session_id,
                new_message=content,
            ):
                pass
        except Exception:
            traceback.print_exc()
            raise
        end_time = time.time()

        session = await self._session_service.get_session(
            app_name=self.APP_NAME,
            user_id=self.USER_ID,
            session_id=session_id,
        )
        if session is None or not session.events:
            logger.debug("FloTorchRunner.arun no session events session=%s", session_id)
            return AgentTrace.from_simple_response(user_input, "")

        messages = process_session_events(session.events)
        logger.debug(
            "FloTorchRunner.arun done events=%d trace_msgs=%d",
            len(session.events), len(messages),
        )
        trace = AgentTrace.from_messages(messages)

        # Prefer tokens captured directly from LLM responses (FlotorchADKLLM path).
        # Fall back to ADK event usage_metadata if the contextvar wasn't populated.
        token_usage = collect_session_token_usage()
        if token_usage and token_usage["total"] > 0:
            total_tokens = token_usage["total"]
            prompt_tokens = token_usage["prompt"]
            completion_tokens = token_usage["completion"]
        else:
            total_tokens = prompt_tokens = completion_tokens = 0
            for event in session.events:
                usage = getattr(event, "usage_metadata", None)
                if usage is not None:
                    total_tokens += getattr(usage, "total_token_count", 0) or 0
                    prompt_tokens += getattr(usage, "prompt_token_count", 0) or 0
                    completion_tokens += getattr(usage, "candidates_token_count", 0) or 0

        return trace.model_copy(update={
            "start_time": start_time,
            "end_time": end_time,
            "total_tokens": total_tokens or None,
            "prompt_tokens": prompt_tokens or None,
            "completion_tokens": completion_tokens or None,
        })

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
