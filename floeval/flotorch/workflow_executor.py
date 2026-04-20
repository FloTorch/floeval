"""Run a DAG of agents in-process and collect traces."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from floeval.config.schemas.io.agent_dataset import AIMessage, AgentTrace
from floeval.config.schemas.io.llm import OpenAIProviderConfig
from floeval.flotorch.dag import (
    DAG,
    Node,
    NodeType,
    aggregate_parent_results,
    initial_ready,
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
            async for _ in runner.run_async(
                user_id=self.user_id,
                session_id=session_id,
                new_message=content,
            ):
                pass

            session = await session_service.get_session(
                app_name=self.app_name,
                user_id=self.user_id,
                session_id=session_id,
            )

            if memory_service and session:
                await memory_service.add_session_to_memory(session)

            if session and session.events:
                messages = process_session_events(session.events)
                trace = AgentTrace.from_messages(messages)
                for m in reversed(messages):
                    if m.get("role") == "assistant" and m.get("content"):
                        final_text = m["content"]
                        break
                if not final_text:
                    final_text = trace.final_response
            else:
                trace = AgentTrace.from_simple_response(prompt, "")

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
        for node in reversed(self.dag.nodes):
            if node.type == NodeType.AGENT:
                res = results.get(node.id, {})
                if res.get("status") == "SUCCESS":
                    final_output = res.get("text", "")
                    break

        agent_summaries: list[dict[str, Any]] = []
        for i, nid in enumerate(completion_order):
            res = results.get(nid, {})
            trace = agent_traces[i] if i < len(agent_traces) else None
            node = node_map.get(nid)
            agent_name = res.get("agent_name") or (node.callable_name if node else nid)
            inp = prompts.get(nid, "")
            out = res.get("text", "")
            if trace is None:
                tool_calls: list[dict[str, Any]] = []
                turn_count = 0
            elif i == 0:
                tool_calls = [{"name": tc.name, "args": tc.args} for tc in trace.tool_calls_made]
                turn_count = trace.turn_count
            else:
                prev_trace = agent_traces[i - 1]
                prev_len = len(prev_trace.messages)
                curr_msgs = trace.messages[prev_len:]
                tool_calls = [
                    {"name": tc.name, "args": tc.args}
                    for m in curr_msgs
                    if isinstance(m, AIMessage)
                    for tc in m.tool_calls
                ]
                turn_count = sum(1 for m in curr_msgs if isinstance(m, AIMessage))
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
