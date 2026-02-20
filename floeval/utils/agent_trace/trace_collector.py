"""Collect traces by running user's agent."""

import asyncio
import json
import logging
from typing import Awaitable, Callable

from floeval.config.schemas.io.agent_dataset import (
    AgentSample,
    AgentTrace,
    PartialAgentSample,
    _to_display_str,
)
from floeval.utils.agent_trace.patchers.langchain_callback import (
    get_langchain_callback,
    is_langchain_available,
)
from floeval.utils.agent_trace.patchers.openai_patcher import patch_openai, unpatch_openai
from floeval.utils.agent_trace.trace_context import (
    TraceContext,
    clear_current_trace,
    set_current_trace,
)

logger = logging.getLogger(__name__)


def _coerce_to_string(result: object) -> str:
    """Convert non-str/non-AgentTrace result to string for trace logging."""
    if isinstance(result, str):
        return result
    if isinstance(result, (dict, list)):
        return json.dumps(result)
    return str(result)


class TraceCollector:
    """Run user's agent and capture traces.

    Supports sync/async agents. Works with OpenAI patcher and LangChain callback.
    """

    def __init__(
        self,
        agent_callable: Callable[[str], str | AgentTrace]
        | Callable[[str], Awaitable[str | AgentTrace]],
    ):
        self.agent = agent_callable
        self._is_async = asyncio.iscoroutinefunction(agent_callable)

    def collect(self, partial_samples: list[PartialAgentSample]) -> list[AgentSample]:
        """Sync interface."""
        if self._is_async:
            return asyncio.run(self.acollect(partial_samples))
        return self._collect_sync(partial_samples)

    async def acollect(
        self, partial_samples: list[PartialAgentSample]
    ) -> list[AgentSample]:
        """Async interface."""
        patch_openai()

        try:
            full_samples = []
            for partial in partial_samples:
                trace = await self._run_one_async(partial)
                full_samples.append(AgentSample.from_partial(partial, trace))
            return full_samples
        finally:
            unpatch_openai()

    def _collect_sync(
        self, partial_samples: list[PartialAgentSample]
    ) -> list[AgentSample]:
        """Sync collection."""
        patch_openai()

        try:
            full_samples = []
            for partial in partial_samples:
                trace = self._run_one_sync(partial)
                full_samples.append(AgentSample.from_partial(partial, trace))
            return full_samples
        finally:
            unpatch_openai()

    def _invoke_agent_sync(self, user_input: str) -> str | AgentTrace:
        """Invoke agent with optional LangChain callback."""
        if is_langchain_available() and hasattr(self.agent, "invoke"):
            try:
                callback = get_langchain_callback()
                return self.agent.invoke(
                    user_input,
                    config={"callbacks": [callback]},
                )
            except Exception as e:
                logger.debug("LangChain invoke with callback failed: %s", e)
        return self.agent(user_input)

    async def _invoke_agent_async(self, user_input: str) -> str | AgentTrace:
        """Invoke agent with optional LangChain callback."""
        if is_langchain_available() and hasattr(self.agent, "ainvoke"):
            try:
                callback = get_langchain_callback()
                return await self.agent.ainvoke(
                    user_input,
                    config={"callbacks": [callback]},
                )
            except Exception as e:
                logger.debug("LangChain ainvoke with callback failed: %s", e)
        result = self.agent(user_input)
        if asyncio.iscoroutine(result):
            return await result
        return result

    def _run_one_sync(self, partial: PartialAgentSample) -> AgentTrace:
        """Run agent once (sync)."""
        user_input_str = _to_display_str(partial.user_input)
        trace_ctx = TraceContext(user_input=user_input_str)
        set_current_trace(trace_ctx)

        try:
            result = self._invoke_agent_sync(user_input_str)
            if isinstance(result, AgentTrace):
                return result
            if isinstance(result, str) and not trace_ctx.final_response:
                trace_ctx.log_ai_turn(result)
            elif not trace_ctx.final_response:
                trace_ctx.log_ai_turn(_coerce_to_string(result))
            return trace_ctx.to_trace()
        except Exception as e:
            logger.exception("Agent invocation failed for sample: %s", e)
            trace_ctx.final_response = f"[Error: {e}]"
            trace_ctx.metadata["error"] = str(e)
            return trace_ctx.to_trace()
        finally:
            clear_current_trace()

    async def _run_one_async(self, partial: PartialAgentSample) -> AgentTrace:
        """Run agent once (async)."""
        user_input_str = _to_display_str(partial.user_input)
        trace_ctx = TraceContext(user_input=user_input_str)
        set_current_trace(trace_ctx)

        try:
            result = await self._invoke_agent_async(user_input_str)
            if isinstance(result, AgentTrace):
                return result
            if isinstance(result, str) and not trace_ctx.final_response:
                trace_ctx.log_ai_turn(result)
            elif not trace_ctx.final_response:
                trace_ctx.log_ai_turn(_coerce_to_string(result))
            return trace_ctx.to_trace()
        except Exception as e:
            logger.exception("Agent invocation failed for sample: %s", e)
            trace_ctx.final_response = f"[Error: {e}]"
            trace_ctx.metadata["error"] = str(e)
            return trace_ctx.to_trace()
        finally:
            clear_current_trace()
