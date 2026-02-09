import json
from pathlib import Path

from floeval.api.dataset_loaders.base import BaseDatasetLoader
from floeval.config.schemas.io.dataset import Sample


class JSONLLoader(BaseDatasetLoader):

    @staticmethod
    def to_samples(file_path: str) -> list[Sample]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        return [Sample(**d) for d in data]


class JSONLoader(BaseDatasetLoader):

    @staticmethod
    def to_samples(file_path: str) -> list[Sample]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [Sample(**d) for d in data]


def get_loader_for_file(file_path: str | Path) -> type[BaseDatasetLoader]:
    """Detect file type based on extension."""
    # TODO: Do we want to add more robust detection (e.g., magic numbers, content sniffing) in the future?
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
