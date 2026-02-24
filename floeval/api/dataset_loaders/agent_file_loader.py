"""Dataset loader for agent evaluation with robust error handling."""

import json
import logging
from pathlib import Path

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
            raise DatasetLoadError(f"Unsupported format: {path.suffix}. Use .json or .jsonl")
        except DatasetLoadError:
            raise
        except Exception as e:
            raise DatasetLoadError(f"Failed to load dataset: {e}") from e

    @staticmethod
    def _parse_reference_tool_calls(data: dict) -> list[ToolCall] | None:
        """Parse reference tool calls from data dict."""
        raw = data.get("reference_tool_calls")
        if not raw:
            return None
        return [ToolCall(**tc) for tc in raw]

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
                    raise DatasetLoadError(f"Invalid JSON on line {line_num}: {e}") from e
                except Exception as e:
                    raise DatasetLoadError(f"Error parsing sample on line {line_num}: {e}") from e

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

        samples = [AgentDatasetLoader._parse_sample(s) for s in data["samples"]]

        if not samples:
            raise DatasetLoadError("'samples' array is empty")

        return AgentDataset(samples=samples)

    @staticmethod
    def _parse_sample(data: dict) -> AgentSample | PartialAgentSample:
        """Parse dict to sample (auto-detect partial vs full)."""
        if "trace" not in data:
            return PartialAgentSample(
                user_input=data["user_input"],
                reference_outcome=data.get("reference_outcome"),
                reference_tool_calls=AgentDatasetLoader._parse_reference_tool_calls(data),
                metadata=data.get("metadata", {}),
            )

        trace_data = data["trace"]
        messages = []

        for msg_data in trace_data["messages"]:
            role = msg_data.get("role", "")

            if role == "human":
                messages.append(HumanMessage(content=msg_data.get("content", "")))
            elif role == "ai":
                tool_calls = [ToolCall(**tc) for tc in msg_data.get("tool_calls", [])]
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

        trace = AgentTrace(
            messages=messages,
            final_response=trace_data.get("final_response", ""),
            metadata=trace_data.get("metadata", {}),
        )

        return AgentSample(
            user_input=data["user_input"],
            trace=trace,
            reference_outcome=data.get("reference_outcome"),
            reference_tool_calls=AgentDatasetLoader._parse_reference_tool_calls(data),
            metadata=data.get("metadata", {}),
        )
