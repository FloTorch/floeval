import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

import yaml

from floeval.cli import CLIEvaluationConfig, CLIGenerationConfig

ConfigT = TypeVar("ConfigT", CLIEvaluationConfig, CLIGenerationConfig)


def check_if_file_exists(file_path: str | Path) -> Path:
    """Utility function to check if a file exists at `file_path`.

    Args:
        file_path: the path to the file to check

    Raises:
        FileNotFoundError: If the file does not exist at the specified path.

    Returns:
        pathlib.Path: The Path object representing the file.
    """
    if isinstance(file_path, str):
        path = Path(file_path)
    else:
        path = file_path

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    return path


class ConfigLoader(ABC):
    @abstractmethod
    def load(self, file_path: str | Path) -> ConfigT:
        """Load configuration from a file and return it as a dictionary."""
        pass


class CLIConfigLoader(Generic[ConfigT], ConfigLoader):
    def __init__(
        self,
        model_class: type[ConfigT],
    ) -> None:
        super().__init__()
        self.model_class = model_class

    def _load_from_yaml(self, file_path: str | Path) -> ConfigT:
        with open(file_path, "r") as f:
            yaml_data = yaml.safe_load(f)

            assert isinstance(
                yaml_data, dict
            ), "config must be a dictionary at the top level"

            cli_config = self.model_class(**yaml_data)
            return cli_config

    def _load_from_json(self, file_path: str | Path) -> ConfigT:
        with open(file_path, "r") as f:
            json_data = json.load(f)

            assert isinstance(
                json_data, dict
            ), "config must be a dictionary at the top level"

            cli_config = self.model_class(**json_data)
            return cli_config

    def load(self, file_path: str | Path) -> ConfigT:

        if isinstance(file_path, str):
            _file_path = Path(file_path)
        else:
            _file_path = file_path

        if _file_path.suffix.lower() not in [".yaml", ".yml", ".json"]:
            raise ValueError(
                "Config file must be a YAML or JSON file with .yaml, .yml, or .json extension"
            )
        if _file_path.suffix.lower() in [".yaml", ".yml"]:
            return self._load_from_yaml(file_path)
        if _file_path.suffix.lower() == ".json":
            return self._load_from_json(file_path)

        raise ValueError("Unsupported config file format")
