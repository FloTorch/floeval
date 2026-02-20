"""Dataset loaders for JSON, CSV, YAML and other formats."""

import json
from pathlib import Path

import yaml

from floeval.config.schemas.prompts import PromptFile


def load_json(file_path: str):
    """Load dataset from JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Loaded dataset data
    """
    pass


def load_csv(file_path: str):
    """Load dataset from CSV file.

    Args:
        file_path: Path to CSV file

    Returns:
        Loaded dataset data
    """
    pass


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
                f"Unsupported prompts file format: {path.suffix}. "
                "Use .yaml, .yml, or .json"
            )

    return PromptFile.model_validate(data)

