import json
from pathlib import Path

from floeval.config.schemas.io.dataset import Dataset


def export_dataset_to_json(dataset: Dataset, output_path: Path, indent=0) -> None:
    """Export a Dataset object to a JSON file in the PRD format."""
    output_data = {"samples": [s.model_dump() for s in dataset.samples]}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=indent)


def save_json_output(payload: dict, output_path: Path) -> None:
    """Save payload to JSON file with consistent CLI formatting."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, default=str)
    print(f"Results saved to {output_path}")


def export_dataset_to_jsonl(dataset: Dataset, output_path: Path) -> None:
    """Export a Dataset object to a JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in dataset.samples:
            json_line = json.dumps(sample.model_dump())
            f.write(json_line + "\n")
