import json
from pathlib import Path

from floeval.api.dataset_loaders.base import BaseDatasetLoader
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
    else:
        raise ValueError(f"Unsupported file type: {ext}")
