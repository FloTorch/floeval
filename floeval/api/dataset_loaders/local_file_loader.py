import json
from pathlib import Path

from floeval.api.dataset_loaders.base import BaseDatasetLoader
from floeval.config.schemas.io.conversational_dataset import (
    ConversationalSample,
    PartialConversationalSample,
)
from floeval.config.schemas.io.dataset import PartialSample, Sample


class JSONLLoader(BaseDatasetLoader):
    @staticmethod
    def _load_data(file_path: str) -> list[dict]:
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    @classmethod
    def to_samples(cls, file_path: str) -> list[Sample]:
        data = cls._load_data(file_path)
        return [Sample(**d) for d in data]

    @classmethod
    def to_partial_samples(cls, file_path: str) -> list[PartialSample]:
        """Load data and convert to PartialDataset format (llm_response field empty)."""
        data = cls._load_data(file_path)
        partial_samples = []
        for d in data:
            # Copy dict; records missing llm_response get empty string.
            partial_dict = {**d}
            partial_samples.append(PartialSample(**partial_dict))
        return partial_samples


class JSONLoader(BaseDatasetLoader):
    @staticmethod
    def _load_data(file_path: str) -> list[dict]:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f).get("samples", [])

    @classmethod
    def to_samples(cls, file_path: str) -> list[Sample]:
        data = cls._load_data(file_path)
        return [Sample(**d) for d in data]

    @classmethod
    def to_partial_samples(cls, file_path: str) -> list[PartialSample]:
        """Load data and convert to PartialDataset format (llm_response field empty)."""
        data = cls._load_data(file_path)
        partial_samples = []
        for d in data:
            # Copy dict; records missing llm_response get empty string.
            partial_dict = {**d}
            partial_samples.append(PartialSample(**partial_dict))
        return partial_samples


class ConversationalJSONLLoader(BaseDatasetLoader):
    @staticmethod
    def _load_rows(file_path: str) -> list[dict]:
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    @classmethod
    def to_conversational_samples(cls, file_path: str) -> list[ConversationalSample]:
        return [ConversationalSample(**d) for d in cls._load_rows(file_path)]

    @classmethod
    def to_partial_conversational_samples(cls, file_path: str) -> list[PartialConversationalSample]:
        return [PartialConversationalSample(**d) for d in cls._load_rows(file_path)]


class ConversationalJSONLoader(BaseDatasetLoader):
    @staticmethod
    def _load_rows(file_path: str) -> list[dict]:
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        rows = payload.get("conversational_samples")
        if not isinstance(rows, list):
            raise ValueError("JSON must contain 'conversational_samples' array")
        return rows

    @classmethod
    def to_conversational_samples(cls, file_path: str) -> list[ConversationalSample]:
        return [ConversationalSample(**d) for d in cls._load_rows(file_path)]

    @classmethod
    def to_partial_conversational_samples(cls, file_path: str) -> list[PartialConversationalSample]:
        return [PartialConversationalSample(**d) for d in cls._load_rows(file_path)]


def get_loader_for_file(file_path: str | Path) -> type[BaseDatasetLoader]:
    """Detect file type based on extension."""
    # TODO: Add robust detection (magic numbers, content sniffing)?
    if not isinstance(file_path, (str, Path)):
        raise ValueError(f"file_path must be a string or Path, got {type(file_path)}")
    if isinstance(file_path, str):
        file_path = Path(file_path)

    # Remove the leading dot and convert to lowercase
    ext = file_path.suffix[1:].lower()
    if ext == "jsonl":
        return JSONLLoader
    elif ext == "json":
        return JSONLoader
    raise ValueError(f"Unsupported file type: {ext}")


def get_conversational_loader_for_file(file_path: str | Path) -> type[BaseDatasetLoader]:
    """JSON / JSONL loaders for ``ConversationalDataset`` files."""
    if not isinstance(file_path, (str, Path)):
        raise ValueError(f"file_path must be a string or Path, got {type(file_path)}")
    if isinstance(file_path, str):
        file_path = Path(file_path)
    ext = file_path.suffix[1:].lower()
    if ext == "jsonl":
        return ConversationalJSONLLoader
    elif ext == "json":
        return ConversationalJSONLoader
    raise ValueError(f"Unsupported file type: {ext}")
