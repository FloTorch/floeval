"""Extract scores from RAGAS evaluation result DataFrames."""

from __future__ import annotations

from typing import Any


def extract_ragas_score(
    results_df: Any,
    row_index: int,
    available_columns: list[str],
    ragas_metric_instance: Any,
    metric_name: str,
    num_metrics: int,
    metric_index: int,
) -> float:
    """Get score for one metric from RAGAS DataFrame row (by column name or fallback)."""
    ragas_metric_name = getattr(ragas_metric_instance, "name", metric_name)
    metric_class_name = ragas_metric_instance.__class__.__name__
    score = None
    if ragas_metric_name in available_columns:
        score = float(results_df[ragas_metric_name].iloc[row_index])
    elif metric_name in available_columns:
        score = float(results_df[metric_name].iloc[row_index])
    elif metric_class_name in available_columns:
        score = float(results_df[metric_class_name].iloc[row_index])
    else:
        for col in available_columns:
            col_lower = col.lower()
            if (
                ragas_metric_name.lower() in col_lower
                or col_lower in ragas_metric_name.lower()
                or metric_name.lower() in col_lower
                or col_lower in metric_name.lower()
                or metric_class_name.lower() in col_lower
                or col_lower in metric_class_name.lower()
            ):
                score = float(results_df[col].iloc[row_index])
                break
    if score is None and num_metrics == 1 and available_columns:
        score = float(results_df[available_columns[-1]].iloc[row_index])
    if score is None and num_metrics > 1 and metric_index < len(available_columns):
        score = float(results_df[available_columns[metric_index]].iloc[row_index])
    if score is None:
        raise ValueError(
            f"Could not find score for metric {metric_name} (ragas name: {ragas_metric_name}, "
            f"class: {metric_class_name}) in RAGAS results. Available columns: {available_columns}"
        )
    return score
