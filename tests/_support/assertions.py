"""Assertions for evaluation results."""

from typing import Any

from floeval.api.evaluation import EvaluationResult


def assert_samples_have_metric(
    results: EvaluationResult,
    metric_key: str,
    *,
    require_score: bool = True,
) -> None:
    """Every sample must include metric_key; optionally require a numeric score."""
    assert results.sample_results, "expected non-empty sample_results"
    for sr in results.sample_results:
        metric_data = sr.get("metrics", {}).get(metric_key)
        assert metric_data is not None, f"missing metric {metric_key!r} for sample {sr!r}"
        if require_score:
            assert metric_data.get("score") is not None, (
                f"score is None for {metric_key} (metadata={metric_data.get('metadata')})"
            )


def assert_metric_registered(all_metrics: dict[str, Any], provider: str, metric_id: str) -> None:
    key = f"{provider}:{metric_id}"
    assert key in all_metrics, f"expected {key!r} in registry, keys sample: {list(all_metrics)[:20]}"
