import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml

from floeval.cli import CliConfig


class ConfigLoader(ABC):
    @abstractmethod
    def load(self, file_path: str) -> dict[str, Any]:
        """Load configuration from a file and return it as a dictionary."""
        pass


class EvalConfigLoader(ConfigLoader):

    def _load_from_yaml(self, file_path: str) -> dict[str, Any]:
        with open(file_path, "r") as f:
            yaml_data = yaml.safe_load(f)

            assert isinstance(
                yaml_data, dict
            ), "config must be a dictionary at the top level"

            cli_config = CliConfig(**yaml_data)
            return cli_config.model_dump()

    def _load_from_json(self, file_path: str) -> dict[str, Any]:
        with open(file_path, "r") as f:
            json_data = json.load(f)

            assert isinstance(
                json_data, dict
            ), "config must be a dictionary at the top level"

            cli_config = CliConfig(**json_data)
            return cli_config.model_dump()

    def load(self, file_path: str) -> dict[str, Any]:
        _file_path = Path(file_path)

        if _file_path.suffix.lower() not in [".yaml", ".yml", ".json"]:
            raise ValueError(
                "Config file must be a YAML file with .yaml or .yml extension"
            )
        if _file_path.suffix.lower() in [".yaml", ".yml"]:
            return self._load_from_yaml(file_path)
        if _file_path.suffix.lower() == ".json":
            return self._load_from_json(file_path)

        raise ValueError("Unsupported config file format")
