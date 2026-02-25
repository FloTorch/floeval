"""Dataset loaders for JSON, CSV, YAML and other formats."""

import csv
import json
from pathlib import Path
from typing import Any, cast

import yaml

from floeval.config.schemas.prompts import PromptFile


def load_json(file_path: str) -> list[dict[str, Any]]:
    """Load dataset samples from a JSON file.

    Args:
        file_path: Path to JSON file with a top-level 'samples' list.

    Returns:
        List of sample dicts.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the JSON structure is invalid.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or "samples" not in data:
        raise ValueError("JSON dataset must be a dict with a 'samples' key.")
    return cast(list[dict[str, Any]], data["samples"])


def load_csv(file_path: str) -> list[dict[str, Any]]:
    """Load dataset samples from a CSV file.

    Args:
        file_path: Path to CSV file. First row must be column headers.

    Returns:
        List of sample dicts (one per non-header row).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_prompts_file(path: str | Path) -> PromptFile:
    """Load prompts from YAML or JSON file.

    Args:
        path: Path to prompts file (YAML or JSON)

    Returns:
        PromptFile containing prompt definitions

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file format is not supported
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Prompts file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        if path.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(f)
        elif path.suffix == ".json":
            data = json.load(f)
        else:
            raise ValueError(
                f"Unsupported prompts file format: {path.suffix}. Use .yaml, .yml, or .json"
            )

    return PromptFile.model_validate(data)
