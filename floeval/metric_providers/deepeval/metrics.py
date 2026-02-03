"""
DeepEval metric implementations
"""

from abc import abstractmethod

from deepeval.evaluate import evaluate
from deepeval.metrics import FaithfulnessMetric

from floeval.api.dataset import Sample
from floeval.api.metrics.base import BaseMetric, MetricResult
from floeval.metric_providers.deepeval.adapter import DeepEvalAdapter

__VALID_DEEPEVAL_METRICS__ = {
    "faithfulness": FaithfulnessMetric,
}


class DeepEvalMetric(BaseMetric):
    """
    Base class for DeepEval metrics.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adapter = DeepEvalAdapter(config=kwargs.get("params", {}))
        self.llm_adapter = kwargs.get("llm_adapter", None)
        # TODO: find a way to not do it when not needed.
        assert self.llm_adapter is not None, "LLM adapter must be provided for DeepEval metrics."

    @property
    def config(self):
        return self.adapter.config

    @abstractmethod
    def compute(self, *args, **kwargs) -> MetricResult:
        """
        Compute DeepEval metric score.
        """


class FaithfulnessDeepEvalMetric(DeepEvalMetric):
    """
    Faithfulness
    metric implementation using DeepEval.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, name="faithfulness", **kwargs)

    def compute(self, sample: Sample) -> MetricResult:
        """
        Compute faithfulness metric score using DeepEval.

        Returns:
            MetricResult: The result of the faithfulness metric computation
        """
        metric_instance = FaithfulnessMetric(model=self.llm_adapter, **self.config)
        test_case = self.adapter.transform_test_case(
            metric_name="faithfulness", test_case_dict=sample.inputs
        )
        result = evaluate(metrics=[metric_instance], test_cases=[test_case]).test_results[0]

        assert result.metrics_data is not None, "Expected metrics_data in the result."
        assert len(result.metrics_data) >= 1, (
            "Expected at least one metric data entry in the result."
        )

        # TODO: We need to decide how to validate multiple metrics_data entries
        try:
            assert len(result.metrics_data) == 1, (
                "Multiple metric results found; only single metric expected."
            )

            metric_data = result.metrics_data[0]
            score = metric_data.score
            passed = metric_data.success

            assert score is not None, (
                f"Expected a score value for metric: {metric_data.name}; success: {metric_data.success}"
            )

            return MetricResult(
                score=score,
                metadata={
                    "passed": passed,
                    "provider": "deepeval",
                    "metric_name": "faithfulness",
                },
            )
        except Exception as e:
            return MetricResult(
                score=0.0,
                metadata={
                    "passed": False,
                    "provider": "deepeval",
                    "metric_name": "faithfulness",
                    "error": str(e),
                },
            )
