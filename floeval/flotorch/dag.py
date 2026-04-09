"""DAG models and utilities for agent workflow execution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Any

logger = logging.getLogger(__name__)


class NodeType(str, Enum):
    """Node types in agent workflow DAG."""

    START = "START"
    AGENT = "AGENT"
    END = "END"


@dataclass
class ResourceRef:
    """Reference to an external resource (Memory/Vectorstore)."""

    id: str
    kind: str  # "MEMORY" | "VECTOR_STORAGE"
    callable_name: str | None = None


@dataclass
class Node:
    """Single node in the DAG."""

    id: str
    type: NodeType
    label: str | None = None
    description: str | None = None
    callable_name: str | None = None
    resources: dict[str, ResourceRef | None] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class Edge:
    """Edge connecting two nodes."""

    from_id: str
    to_id: str


@dataclass
class DAG:
    """Directed Acyclic Graph representing agent workflow."""

    version: str
    graph_id: str
    session_id: str
    workflow_name: str | None = None
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    @staticmethod
    def _resources_list_to_dict(resources_list: Any) -> dict[str, ResourceRef | None]:
        """Convert resources array to dict keyed by type.

        Input formats:
        - List: [{"type": "MEMORY", "id": "...", "callableName": "..."}, ...]
        - Dict: {"memory": {...}, "vector_storage": {...}}
        """
        out: dict[str, ResourceRef | None] = {}

        if isinstance(resources_list, dict):
            for key, val in resources_list.items():
                if val is None:
                    out[key] = None
                    continue
                rid = val.get("id")
                kind = val.get("kind") or val.get("type")
                cname = val.get("callableName") or val.get("callable_name")
                out[key] = (
                    ResourceRef(id=rid, kind=kind, callable_name=cname)
                    if (rid and kind)
                    else None
                )
            return out

        for idx, r in enumerate(resources_list or []):
            rtype = (r.get("type") or r.get("kind") or "").strip()
            key = rtype.lower() if rtype else f"res_{idx}"
            rid = r.get("id")
            cname = r.get("callableName") or r.get("callable_name")
            kind_upper = rtype.upper() if rtype else rtype
            out[key] = (
                ResourceRef(id=rid, kind=kind_upper, callable_name=cname)
                if (rid and rtype)
                else None
            )
        return out

    @staticmethod
    def from_builder_json(raw: dict[str, Any]) -> DAG:
        """Parse DAG from workflow builder JSON (config + nodes + edges)."""
        cfg = raw.get("config")
        if not cfg:
            raise ValueError("Missing 'config' in DAG JSON")

        workflow_id = cfg.get("uid")
        if not workflow_id:
            raise ValueError("Missing uid in config")

        workflow_name = cfg.get("name")
        invocation_id = raw.get("invocationId") or cfg.get("invocationId")

        nodes: list[Node] = []
        for n in cfg.get("nodes", []):
            nid = n.get("id")
            ntype = n.get("type")
            cname = n.get("callableName")
            if not nid or not ntype:
                continue

            resources_dict = DAG._resources_list_to_dict(n.get("resources", []))
            lbl = cname if ntype == NodeType.AGENT.value else n.get("label")

            node = Node(
                id=nid,
                type=NodeType(ntype),
                label=lbl,
                description=n.get("description"),
                callable_name=cname,
                resources=resources_dict,
                raw=n,
            )
            nodes.append(node)

        edges: list[Edge] = []
        for e in cfg.get("edges", []):
            src = e.get("sourceNodeId")
            tgt = e.get("targetNodeId")
            if src and tgt:
                edges.append(Edge(from_id=src, to_id=tgt))

        dag = DAG(
            version=cfg.get("version") or "latest",
            graph_id=workflow_id,
            session_id=invocation_id or workflow_id,
            workflow_name=workflow_name,
            nodes=nodes,
            edges=edges,
        )
        dag._validate_basic()
        return dag

    def _validate_basic(self) -> None:
        """Validate DAG structure."""
        start = [n for n in self.nodes if n.type == NodeType.START]
        end = [n for n in self.nodes if n.type == NodeType.END]
        if len(start) != 1 or len(end) != 1:
            raise ValueError(
                f"DAG must have exactly one START and one END "
                f"(got start={len(start)}, end={len(end)})"
            )

        node_ids = {n.id for n in self.nodes}
        for e in self.edges:
            if e.from_id not in node_ids or e.to_id not in node_ids:
                raise ValueError(f"Edge references unknown node: {e.from_id} -> {e.to_id}")

    def node_map(self) -> dict[str, Node]:
        """Get node ID -> Node mapping."""
        return {n.id: n for n in self.nodes}

    def parents_map(self) -> dict[str, list[str]]:
        """Get node ID -> parent IDs mapping."""
        pm: dict[str, list[str]] = {n.id: [] for n in self.nodes}
        for e in self.edges:
            pm[e.to_id].append(e.from_id)
        return pm

    def succ_map(self) -> dict[str, list[str]]:
        """Get node ID -> successor IDs mapping."""
        sm: dict[str, list[str]] = {n.id: [] for n in self.nodes}
        for e in self.edges:
            sm[e.from_id].append(e.to_id)
        return sm

    def agent_parent_ids(self, nid: str) -> list[str]:
        """Get AGENT parent node IDs for given node."""
        pm = self.parents_map()
        nmap = self.node_map()
        return [pid for pid in pm.get(nid, []) if nmap[pid].type == NodeType.AGENT]


def initial_ready(parents_map: dict[str, list[str]]) -> list[str]:
    """Get nodes with no incoming edges (ready to execute)."""
    return [nid for nid, ps in parents_map.items() if len(ps) == 0]


def aggregate_parent_results(upstream_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Merge results from multiple upstream agents.

    If single upstream: return that result directly.
    If multiple: merge all results into single dict.
    """
    if len(upstream_map) == 1:
        return list(upstream_map.values())[0]

    merged_result: dict[str, Any] = {}
    for node_id, result in upstream_map.items():
        try:
            text_content = result.get("text")
            if not text_content:
                continue

            try:
                parsed_content = json.loads(text_content)
                if isinstance(parsed_content, dict):
                    merged_result.update(parsed_content)
                else:
                    merged_result[f"{node_id}_result"] = parsed_content
            except (json.JSONDecodeError, TypeError, ValueError):
                merged_result[f"{node_id}_text"] = text_content

        except (KeyError, AttributeError, TypeError):
            merged_result[f"{node_id}_error"] = "malformed_result"

    return merged_result
