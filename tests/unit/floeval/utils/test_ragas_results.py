"""Pure helper: extract_ragas_score."""

import pandas as pd
import pytest

from floeval.utils.ragas_results import extract_ragas_score

pytestmark = pytest.mark.unit


class _Metric:
    name = "answer_relevancy"


def test_extract_ragas_score_by_metric_name() -> None:
    df = pd.DataFrame({"answer_relevancy": [0.75, 0.25]})
    score = extract_ragas_score(
        df,
        row_index=0,
        available_columns=list(df.columns),
        ragas_metric_instance=_Metric(),
        metric_name="answer_relevancy",
        num_metrics=1,
        metric_index=0,
    )
    assert score == pytest.approx(0.75)


def test_extract_ragas_score_fallback_single_column() -> None:
    df = pd.DataFrame({"some_other": [0.9]})
    score = extract_ragas_score(
        df,
        row_index=0,
        available_columns=["some_other"],
        ragas_metric_instance=_Metric(),
        metric_name="missing",
        num_metrics=1,
        metric_index=0,
    )
    assert score == pytest.approx(0.9)
