"""Root pytest configuration and shared fixtures."""

import logging
import os
from pathlib import Path
from typing import Any

import pytest
import yaml


def pytest_configure(config: pytest.Config) -> None:
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")


@pytest.fixture(scope="session")
def tests_root() -> Path:
    return Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def config_data_dir(tests_root: Path) -> Path:
    return tests_root / "config_data"


@pytest.fixture(scope="session")
def load_yaml():
    def _load(path: Path) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        return data

    return _load


def resolve_env_placeholders(obj: Any) -> Any:
    """Replace strings like ${VAR} with os.environ.get(VAR, '')."""
    if isinstance(obj, dict):
        return {k: resolve_env_placeholders(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_env_placeholders(x) for x in obj]
    if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        return os.environ.get(obj[2:-1], "")
    return obj


@pytest.fixture(scope="session")
def resolve_env():
    return resolve_env_placeholders


@pytest.fixture(scope="session", autouse=True)
def _session_logging() -> None:
    level = os.getenv("FLOEVAL_LOG_LEVEL", "WARNING").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    logging.getLogger("floeval").setLevel(level)
    logging.getLogger("deepeval").setLevel(logging.WARNING)
    logging.getLogger("ragas").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


@pytest.fixture
def requires_llm_credentials() -> None:
    if not os.environ.get("FLOEVAL_API_KEY", "").strip():
        pytest.skip("FLOEVAL_API_KEY is not set (required for LLM-backed integration tests)")
