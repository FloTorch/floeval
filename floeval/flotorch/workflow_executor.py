"""Run a DAG of agents in-process and collect traces."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from floeval.config.schemas.io.agent_dataset import AgentTrace, WorkflowExecution
from floeval.config.schemas.io.llm import OpenAIProviderConfig
from floeval.flotorch.dag import (
    DAG,
    Node,
    NodeType,
    aggregate_parent_results,
    initial_ready,
)
from floeval.utils.agent_trace.trace_context import (
    collect_session_token_usage,
    start_session_token_tracking,
)

logger = logging.getLogger(__name__)


def _resources_to_dict(node: Node) -> dict[str, dict[str, Any] | None]:
    """Convert node.resources (ResourceRef) to dict format for memory services."""
    out: dict[str, dict[str, Any] | None] = {}
    for key, ref in node.resources.items():
        if ref is None:
            out[key] = None
        else:
            out[key] = {
                "id": ref.id,
                "kind": ref.kind,
                "callable_name": ref.callable_name,
            }
    return out


class WorkflowExecutor:
    """Execute agent workflow DAG without Temporal.

    For each node in topological order:
    - Build agent from config
    - Execute with aggregated upstream results
    - Capture trace
    - Pass result to successors
    """

    def __init__(
        self,
        dag: DAG,
        llm_config: OpenAIProviderConfig,
        app_name: str = "floeval-workflow",
        user_id: str = "workflow-user",
        run_headers: dict[str, str] | None = None,
    ):
        self.dag = dag
        self.llm_config = llm_config
        self.app_name = app_name
        self.user_id = user_id
        self.base_url = llm_config.base_url
        self.api_key = llm_config.api_key
        self.run_headers = dict(run_headers or {})

    async def execute(self, wf_input: dict[str, Any] | str) -> dict[str, Any]:
        """Execute DAG for single input, return aggregated traces.

        Args:
            wf_input: Workflow input (user question/task). Str or dict with "text" key.

        Returns:
            {
                "results": {node_id: result_dict},
                "agent_traces": [AgentTrace for each agent],
                "final_output": final agent's output text,
                "session_id": session ID used
            }
        """
        from google.adk.runners import Runner
        from google.adk.sessions.in_memory_session_service import InMemorySessionService
        from google.genai import types

        from floeval.flotorch.adk.agent import FlotorchADKAgent
        from floeval.flotorch.adk.utils.adk_utils import process_session_events

        try:
            from floeval.flotorch.adk.memory import (
                FlotorchADKVectorMemoryService,
                FlotorchMemoryService,
            )
        except ImportError:
            FlotorchMemoryService = None
            FlotorchADKVectorMemoryService = None

        node_map = self.dag.node_map()
        parents_map = self.dag.parents_map()
        succ_map = self.dag.succ_map()

        session_id = self.dag.session_id or f"eval-{id(self)}"

        session_service = InMemorySessionService()
        await session_service.create_session(
            app_name=self.app_name,
            user_id=self.user_id,
            session_id=session_id,
        )

        results: dict[str, dict[str, Any]] = {}
        agent_traces: list[AgentTrace] = []
        completion_order: list[str] = []
        prompts: dict[str, str] = {}
        running: dict[str, asyncio.Task] = {}
        completion_q: asyncio.Queue[tuple[str, dict[str, Any], AgentTrace | None]] = (
            asyncio.Queue()
        )

        def _prompt_from_prev(result_from_prev: Any) -> str:
            if isinstance(result_from_prev, dict):
                return (
                    json.dumps(result_from_prev.get("text", ""))
                    if result_from_prev.get("text") is not None
                    else json.dumps(result_from_prev)
                )
            return str(result_from_prev) if result_from_prev else ""

        async def _run_agent_node(nid: str, prompt: str) -> tuple[dict[str, Any], AgentTrace]:
            node = node_map[nid]
            memory_service = None

            if FlotorchMemoryService and FlotorchADKVectorMemoryService:
                resources_dict = _resources_to_dict(node)
                memory_ref = resources_dict.get("memory")
                if memory_ref and memory_ref.get("callable_name"):
                    memory_service = FlotorchMemoryService(
                        name=memory_ref["callable_name"],
                        api_key=self.api_key,
                        base_url=self.base_url,
                        default_headers=self.run_headers,
                    )
                else:
                    vs_ref = resources_dict.get("vector_storage")
                    if vs_ref and vs_ref.get("callable_name"):
                        memory_service = FlotorchADKVectorMemoryService(
                            api_key=self.api_key,
                            base_url=self.base_url,
                            vectorstore_id=vs_ref["callable_name"],
                            default_headers=self.run_headers,
                        )

            flt_agent = FlotorchADKAgent(
                agent_name=node.callable_name or nid,
                base_url=self.base_url,
                api_key=self.api_key,
                enable_memory=True if memory_service else False,
                default_headers=self.run_headers,
            )
            agent = flt_agent.get_agent()

            if memory_service:
                runner = Runner(
                    agent=agent,
                    app_name=self.app_name,
                    session_service=session_service,
                    memory_service=memory_service,
                )
            else:
                runner = Runner(
                    agent=agent,
                    app_name=self.app_name,
                    session_service=session_service,
                )

            content = types.Content(
                role="user",
                parts=[types.Part(text=prompt)],
            )

            final_text = ""
            start_session_token_tracking()
            node_start_time = time.time()
            session_before = await session_service.get_session(
                app_name=self.app_name,
                user_id=self.user_id,
                session_id=session_id,
            )
            event_count_before = len(session_before.events) if session_before and session_before.events else 0
            async for _ in runner.run_async(
                user_id=self.user_id,
                session_id=session_id,
                new_message=content,
            ):
                pass
            node_end_time = time.time()

            # Prefer contextvar tokens captured by FlotorchADKLLM (per-agent, no cross-contamination).
            # Fall back to ADK event usage_metadata for standard Gemini/OpenAI ADK LLMs.
            token_usage = collect_session_token_usage()
            if token_usage and token_usage["total"] > 0:
                total_tokens = token_usage["total"]
                prompt_tokens = token_usage["prompt"]
                completion_tokens = token_usage["completion"]
            else:
                total_tokens = prompt_tokens = completion_tokens = 0

            session = await session_service.get_session(
                app_name=self.app_name,
                user_id=self.user_id,
                session_id=session_id,
            )

            if memory_service and session:
                await memory_service.add_session_to_memory(session)

            if session and session.events:
                new_events = session.events[event_count_before:]
                messages = process_session_events(new_events)
                trace = AgentTrace.from_messages(messages)
                for m in reversed(messages):
                    if m.get("role") == "assistant" and m.get("content"):
                        final_text = m["content"]
                        break
                if not final_text:
                    final_text = trace.final_response

                # Only use ADK usage_metadata if FlotorchADKLLM contextvar didn't populate tokens.
                if total_tokens == 0:
                    for event in new_events:
                        usage = getattr(event, "usage_metadata", None)
                        if usage is not None:
                            total_tokens += getattr(usage, "total_token_count", 0) or 0
                            prompt_tokens += getattr(usage, "prompt_token_count", 0) or 0
                            completion_tokens += getattr(usage, "candidates_token_count", 0) or 0

                trace = trace.model_copy(update={
                    "start_time": node_start_time,
                    "end_time": node_end_time,
                    "total_tokens": total_tokens or None,
                    "prompt_tokens": prompt_tokens or None,
                    "completion_tokens": completion_tokens or None,
                })
            else:
                trace = AgentTrace.from_simple_response(prompt, "")
                trace = trace.model_copy(update={
                    "start_time": node_start_time,
                    "end_time": node_end_time,
                    "total_tokens": total_tokens or None,
                    "prompt_tokens": prompt_tokens or None,
                    "completion_tokens": completion_tokens or None,
                })

            result = {
                "status": "SUCCESS",
                "text": final_text,
                "node_id": nid,
                "agent_name": node.callable_name,
            }
            return result, trace

        async def launch(nid: str) -> None:
            node = node_map[nid]

            if node.type != NodeType.AGENT:
                await completion_q.put((nid, {"status": "SUCCESS", "node_id": nid, "result": {}}, None))
                return

            agent_parent_ids = self.dag.agent_parent_ids(nid)
            upstream_map = {
                pid: results[pid]
                for pid in agent_parent_ids
                if pid in results and results[pid].get("status") == "SUCCESS"
            }

            if not upstream_map:
                result_from_prev: Any = wf_input
            else:
                result_from_prev = aggregate_parent_results(upstream_map)

            prompt = _prompt_from_prev(result_from_prev)
            prompts[nid] = prompt

            async def _runner() -> None:
                try:
                    result, trace = await _run_agent_node(nid, prompt)
                    await completion_q.put((nid, result, trace))
                except Exception as exc:
                    logger.exception("Agent %s failed: %s", nid, exc)
                    await completion_q.put(
                        (
                            nid,
                            {
                                "status": "ERROR",
                                "text": f"Error: {str(exc)}",
                                "node_id": nid,
                            },
                            None,
                        )
                    )

            running[nid] = asyncio.create_task(_runner())

        ready: list[str] = list(initial_ready(parents_map))
        for nid in ready:
            await launch(nid)
            ready.remove(nid)

        while ready or running or not completion_q.empty():
            for nid in list(ready):
                parents = parents_map.get(nid, [])
                if parents and not all(
                    results.get(p, {}).get("status") == "SUCCESS" for p in parents
                ):
                    continue
                await launch(nid)
                ready.remove(nid)

            while not completion_q.empty():
                done_nid, done_res, done_trace = await completion_q.get()
                running.pop(done_nid, None)
                results[done_nid] = done_res

                if done_trace:
                    completion_order.append(done_nid)
                    agent_traces.append(done_trace)

                if done_res.get("status") == "VALIDATION_FAILED":
                    ready.clear()
                    running.clear()
                    break

                for succ in succ_map.get(done_nid, []):
                    parents = parents_map.get(succ, [])
                    if parents and all(
                        results.get(p, {}).get("status") == "SUCCESS" for p in parents
                    ):
                        if succ not in running and succ not in ready:
                            ready.append(succ)

            if not running and not ready:
                break

            if running and not ready:
                done_nid, done_res, done_trace = await completion_q.get()
                running.pop(done_nid, None)
                results[done_nid] = done_res

                if done_trace:
                    completion_order.append(done_nid)
                    agent_traces.append(done_trace)

                if done_res.get("status") == "VALIDATION_FAILED":
                    ready.clear()
                    running.clear()
                    break

                for succ in succ_map.get(done_nid, []):
                    parents = parents_map.get(succ, [])
                    if parents and all(
                        results.get(p, {}).get("status") == "SUCCESS" for p in parents
                    ):
                        if succ not in running and succ not in ready:
                            ready.append(succ)

        final_output = ""
        for nid in reversed(completion_order):
            res = results.get(nid, {})
            if res.get("status") == "SUCCESS":
                final_output = res.get("text", "")
                break

        agent_summaries: list[dict[str, Any]] = []
        for i, (nid, trace) in enumerate(zip(completion_order, agent_traces)):
            res = results.get(nid, {})
            node = node_map.get(nid)
            agent_name = res.get("agent_name") or (node.callable_name if node else nid)
            inp = prompts.get(nid, "")
            out = res.get("text", "")
            tool_calls = [{"name": tc.name, "args": tc.args} for tc in trace.tool_calls_made]
            turn_count = trace.turn_count
            agent_summaries.append({
                "agent_name": agent_name,
                "node_id": nid,
                "input": inp,
                "output": out,
                "tool_calls": tool_calls,
                "turn_count": turn_count,
                "tool_call_count": len(tool_calls),
            })

        return {
            "results": results,
            "agent_traces": agent_traces,
            "agent_summaries": agent_summaries,
            "final_output": final_output,
            "session_id": session_id,
        }

    async def execute_and_build(
        self,
        wf_input: dict[str, Any] | str,
        reference_outcome: Any | None = None,
        reference_tool_calls: list[Any] | None = None,
    ) -> WorkflowExecution:
        """Execute workflow DAG and return a WorkflowExecution for WorkflowEvaluation.

        This is the preferred method when integrating with WorkflowEvaluation.
        The existing execute() method is unchanged and still returns a raw dict.

        Args:
            wf_input: Workflow input (str or dict with "text" key).
            reference_outcome: Optional expected outcome for reference-based metrics.

        Returns:
            WorkflowExecution: Structured result ready for WorkflowEvaluation.
        """
        raw = await self.execute(wf_input)

        agent_traces = raw.get("agent_traces", [])
        agent_summaries: list[dict[str, Any]] = raw.get("agent_summaries", [])
        agent_names = [s.get("agent_name", f"agent_{i}") for i, s in enumerate(agent_summaries)]

        # Enrich each trace with its agent_name using model_copy (AgentTrace is frozen)
        enriched_traces = []
        for i, trace in enumerate(agent_traces):
            name = agent_names[i] if i < len(agent_names) else f"agent_{i}"
            enriched_traces.append(trace.model_copy(update={"agent_name": name}))

        user_input_str = (
            wf_input
            if isinstance(wf_input, str)
            else wf_input.get("text", str(wf_input))
        )

        return WorkflowExecution(
            user_input=user_input_str,
            final_output=raw.get("final_output", ""),
            agent_traces=enriched_traces,
            agent_names=agent_names,
            node_results=raw.get("results", {}),
            reference_outcome=reference_outcome,
            reference_tool_calls=reference_tool_calls,
            session_id=raw.get("session_id"),
            dag_graph_id=self.dag.graph_id,
            metadata={"agent_summaries": agent_summaries},
        )
