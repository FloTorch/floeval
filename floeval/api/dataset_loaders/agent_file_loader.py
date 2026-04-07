"""Dataset loader for agent evaluation with robust error handling."""

import json
import logging
from pathlib import Path
from typing import Any

from floeval.config.schemas.io.agent_dataset import (
    AgentDataset,
    AgentSample,
    AgentTrace,
    AIMessage,
    HumanMessage,
    PartialAgentSample,
    ToolCall,
    ToolMessage,
)

logger = logging.getLogger(__name__)


class DatasetLoadError(Exception):
    """Raised when dataset loading fails."""


class AgentDatasetLoader:
    """Load agent datasets from files."""

    @staticmethod
    def from_file(path: str | Path) -> AgentDataset:
        """Load dataset from JSON or JSONL file.

        Raises:
            FileNotFoundError: File does not exist.
            DatasetLoadError: File format invalid or parsing failed.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        try:
            if path.suffix == ".jsonl":
                return AgentDatasetLoader._load_jsonl(path)
            if path.suffix == ".json":
                return AgentDatasetLoader._load_json(path)
            raise DatasetLoadError(
                f"Unsupported format: {path.suffix}. Use .json or .jsonl"
            )
        except DatasetLoadError:
            raise
        except Exception as e:
            raise DatasetLoadError(f"Failed to load dataset: {e}") from e

    @staticmethod
    def _trace_from_dict(trace_data: dict) -> AgentTrace:
        """Parse trace dict to AgentTrace."""
        messages = []
        for msg_data in trace_data.get("messages", []):
            role = msg_data.get("role", "")
            if role == "assistant":
                role = "ai"
            if role == "human":
                messages.append(HumanMessage(content=msg_data.get("content", "")))
            elif role == "ai":
                tool_calls = [
                    ToolCall(**tc) for tc in msg_data.get("tool_calls", [])
                ]
                messages.append(
                    AIMessage(
                        content=msg_data.get("content", ""),
                        tool_calls=tool_calls,
                    )
                )
            elif role == "tool":
                messages.append(
                    ToolMessage(
                        content=msg_data.get("content", ""),
                        tool_name=msg_data.get("tool_name", ""),
                        tool_call_id=msg_data.get("tool_call_id"),
                    )
                )
            else:
                raise ValueError(f"Unknown role: {role}")
        return AgentTrace(
            messages=messages,
            final_response=trace_data.get("final_response", ""),
            metadata=trace_data.get("metadata", {}),
        )

    @staticmethod
    def _parse_reference_tool_calls(data: dict) -> list[ToolCall] | None:
        """Parse reference tool calls from data dict."""
        raw = data.get("reference_tool_calls")
        if not raw:
            return None
        return [ToolCall(**tc) for tc in raw]

    @staticmethod
    def _get_user_input(data: dict) -> str | None:
        return data.get("user_input") or data.get("question")

    @staticmethod
    def _get_reference_outcome(data: dict) -> str | dict | None:
        return data.get("reference_outcome") or data.get("answer")

    @staticmethod
    def _load_jsonl(path: Path) -> AgentDataset:
        """Load JSONL file."""
        samples = []

        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    sample = AgentDatasetLoader._parse_sample(data)
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    raise DatasetLoadError(
                        f"Invalid JSON on line {line_num}: {e}"
                    ) from e
                except Exception as e:
                    raise DatasetLoadError(
                        f"Error parsing sample on line {line_num}: {e}"
                    ) from e

        if not samples:
            raise DatasetLoadError(f"No valid samples found in {path}")

        return AgentDataset(samples=samples)

    @staticmethod
    def _load_json(path: Path) -> AgentDataset:
        """Load JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise DatasetLoadError("JSON must be an object with 'samples' array")

        if "samples" not in data:
            raise DatasetLoadError("JSON must have 'samples' key")

        if not isinstance(data["samples"], list):
            raise DatasetLoadError("'samples' must be an array")

        samples = [
            AgentDatasetLoader._parse_sample(s) for s in data["samples"]
        ]

        if not samples:
            raise DatasetLoadError("'samples' array is empty")

        return AgentDataset(samples=samples)

    @staticmethod
    def _parse_sample(data: dict) -> AgentSample | PartialAgentSample:
        """Parse dict to sample (auto-detect partial vs full).
        """
        user_input = AgentDatasetLoader._get_user_input(data)
        if not user_input:
            raise ValueError("Record missing both 'user_input' and 'question'")
        reference_outcome = AgentDatasetLoader._get_reference_outcome(data)
        ref_tool_calls = AgentDatasetLoader._parse_reference_tool_calls(data)

        def _conv_meta() -> dict[str, Any]:
            return {
                "scenario": data.get("scenario"),
                "expected_outcome": data.get("expected_outcome"),
                "user_description": data.get("user_description"),
                "chatbot_role": data.get("chatbot_role"),
                "conversation_context": data.get("conversation_context"),
            }

        if "trace" not in data:
            return PartialAgentSample(
                user_input=user_input,
                reference_outcome=reference_outcome,
                reference_tool_calls=ref_tool_calls,
                metadata=data.get("metadata", {}),
                **_conv_meta(),
            )

        trace = AgentDatasetLoader._trace_from_dict(data["trace"])
        agent_traces: list[AgentTrace] | None = None
        raw_traces = data.get("agent_traces")
        if isinstance(raw_traces, list) and raw_traces:
            agent_traces = [
                AgentDatasetLoader._trace_from_dict(t) for t in raw_traces if isinstance(t, dict)
            ]

        return AgentSample(
            user_input=user_input,
            trace=trace,
            reference_outcome=reference_outcome,
            reference_tool_calls=ref_tool_calls,
            agent_traces=agent_traces,
            metadata=data.get("metadata", {}),
            **_conv_meta(),
        )
