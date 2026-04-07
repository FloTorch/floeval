"""MetricRegistry smoke tests (providers register on import)."""

# ruff: noqa: D103 — test names describe behavior

import pytest

import floeval.metric_providers  # noqa: F401 — register deepeval/ragas/builtin
from floeval.api.metrics.registry import MetricRegistry

pytestmark = pytest.mark.unit


def test_list_providers_includes_expected_backends() -> None:
    reg = MetricRegistry()
    providers = reg.list_providers()
    assert "deepeval" in providers
    assert "ragas" in providers


@pytest.mark.parametrize(
    ("provider", "metric_id"),
    [
        ("deepeval", "exact_match"),
        ("deepeval", "pattern_match"),
        ("deepeval", "faithfulness"),
        ("deepeval", "conversational_g_eval"),
        ("ragas", "answer_relevancy"),
        ("ragas", "context_recall"),
    ],
)
def test_metric_registered(provider: str, metric_id: str) -> None:
    reg = MetricRegistry()
    names = reg.list_metrics(provider)
    assert metric_id in names, f"{provider}:{metric_id} not in {names}"
