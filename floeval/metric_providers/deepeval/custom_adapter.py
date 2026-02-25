"""DeepEval custom metric adapter."""

import logging
from typing import Any, Type

from deepeval.metrics import BaseMetric as DeepEvalBaseMetric
from deepeval.test_case import LLMTestCase

from floeval.api.dataset import Dataset, Sample
from floeval.api.metrics.base import BaseMetric, MetricResult
from floeval.config.schemas.io.llm import LLMProviderConfig
from floeval.metric_providers.deepeval.adapter import DeepEvalAdapter

logger = logging.getLogger(__name__)


class DeepEvalCustomMetricAdapter:
    """Transforms Floeval custom metrics to DeepEval native metrics."""

    def __init__(self, llm_config: LLMProviderConfig | None = None):
        """Initialize adapter with optional llm config."""
        self.llm_config = llm_config
        self._deepeval_adapter = DeepEvalAdapter(config={})

    def transform_metric(self, floeval_metric: BaseMetric) -> Type[DeepEvalBaseMetric]:
        """Transform Floeval metric to DeepEval metric class."""
        if hasattr(floeval_metric, "user_func") or hasattr(floeval_metric, "_user_func"):
            return self._transform_function_metric(floeval_metric)

        if hasattr(floeval_metric, "description") and hasattr(floeval_metric, "llm_helper"):
            return self._transform_criteria_metric(floeval_metric)

        raise ValueError(
            f"Unsupported metric type for DeepEval transformation: {type(floeval_metric)}. "
            f"Expected FunctionBasedMetric or CriteriaBasedMetric."
        )

    def _transform_function_metric(self, metric: BaseMetric) -> Type[DeepEvalBaseMetric]:
        """Transform function-based metric to DeepEval metric class."""
        metric_name = metric.name
        floeval_metric_instance = metric
        threshold = getattr(metric, "threshold", 0.5)

        class GeneratedDeepEvalMetric(DeepEvalBaseMetric):
            """
            Runtime-generated DeepEval metric from Floeval custom metric.

            Wraps Floeval metric and adapts it to DeepEval interface.
            """

            def __init__(self, **kwargs):
                """Initialize DeepEval metric with required state."""
                self.threshold = threshold
                self._floeval_metric = floeval_metric_instance

                # DeepEval state management (required!)
                self.score = None
                self.success = None
                self.reason = None
                self.error = None

            def measure(self, test_case: LLMTestCase) -> float:
                """
                Synchronous evaluation method.

                DeepEval requires this method. Must set self.score, self.success, self.reason.

                Args:
                    test_case: DeepEval LLMTestCase

                Returns:
                    float: Score between 0 and 1
                """
                try:
                    # Transform test case to Floeval sample
                    floeval_sample = self._transform_test_case(test_case)

                    # Execute Floeval metric (sync)
                    result = self._floeval_metric.evaluate(floeval_sample)

                    # Extract and set state
                    self._set_state_from_result(result)

                    return self.score if self.score is not None else 0.0

                except Exception as e:
                    self.error = str(e)
                    self.score = 0.0
                    self.success = False
                    self.reason = f"Error: {str(e)}"
                    return 0.0

            async def a_measure(self, test_case: LLMTestCase) -> float:
                """
                Asynchronous evaluation method.

                DeepEval supports async evaluation via this method.

                Args:
                    test_case: DeepEval LLMTestCase

                Returns:
                    float: Score between 0 and 1
                """
                try:
                    # Transform test case to Floeval sample
                    floeval_sample = self._transform_test_case(test_case)

                    # Execute Floeval metric (async)
                    result = await self._floeval_metric.aevaluate(floeval_sample)

                    # Extract and set state
                    self._set_state_from_result(result)

                    return self.score if self.score is not None else 0.0

                except Exception as e:
                    self.error = str(e)
                    self.score = 0.0
                    self.success = False
                    self.reason = f"Error: {str(e)}"
                    return 0.0

            def is_successful(self) -> bool:
                """
                Check if evaluation passed.

                DeepEval requires this method.

                Returns:
                    bool: True if score >= threshold
                """
                if self.error is not None:
                    return False
                if self.success is not None:
                    return self.success
                if self.score is not None:
                    return self.score >= self.threshold
                return False

            @property
            def __name__(self):
                """Metric name property (required by DeepEval)."""
                return metric_name

            def _transform_test_case(self, test_case: LLMTestCase) -> Sample:
                """
                Transform DeepEval test case to Floeval sample.

                Mapping:
                - input → user_input
                - actual_output → llm_response
                - expected_output → ground_truth
                - retrieval_context → contexts

                Args:
                    test_case: DeepEval LLMTestCase

                Returns:
                    Floeval Sample
                """
                return Sample(
                    user_input=test_case.input or "",
                    llm_response=test_case.actual_output or "",
                    contexts=test_case.retrieval_context or [],
                    ground_truth=test_case.expected_output,
                )

            def _set_state_from_result(self, result: Any) -> None:
                """
                Extract state from Floeval MetricResult and set DeepEval state.

                Args:
                    result: Floeval MetricResult or numeric value
                """
                if isinstance(result, MetricResult):
                    self.score = result.score if result.score is not None else 0.0
                    self.reason = result.metadata.get("reason") or result.metadata.get("error", "")
                elif isinstance(result, (int, float)):
                    self.score = float(result)
                    self.reason = "Score computed"
                else:
                    self.score = 0.0
                    self.reason = "Unknown result type"

                # Ensure score is in 0-1 range
                self.score = max(0.0, min(1.0, self.score))

                # Set success based on threshold
                self.success = self.score >= self.threshold

        return GeneratedDeepEvalMetric

    def _transform_criteria_metric(self, metric: BaseMetric) -> Type[DeepEvalBaseMetric]:
        """
        Transform criteria-based metric to DeepEval metric class.

        For criteria-based metrics, we generate a DeepEval metric that uses the Floeval
        criteria metric's LLM evaluation logic. This wraps the criteria metric's evaluation.

        Args:
            metric: Floeval CriteriaBasedMetric instance

        Returns:
            DeepEval metric class
        """
        metric_name = metric.name
        threshold = getattr(metric, "threshold", 0.5)
        floeval_metric_instance = metric

        class GeneratedDeepEvalCriteriaMetric(DeepEvalBaseMetric):
            """
            Runtime-generated DeepEval metric from Floeval criteria-based metric.

            Uses Floeval criteria metric's LLM-as-judge evaluation logic.
            """

            def __init__(self, **kwargs):
                """Initialize DeepEval criteria metric with required state."""
                self.threshold = threshold
                self._floeval_metric = floeval_metric_instance

                # DeepEval state management (required!)
                self.score = None
                self.success = None
                self.reason = None
                self.error = None

            def measure(self, test_case: LLMTestCase) -> float:
                """Synchronous evaluation using Floeval criteria metric."""
                try:
                    floeval_sample = self._transform_test_case(test_case)
                    result = self._floeval_metric.evaluate(floeval_sample)
                    self._set_state_from_result(result)
                    return self.score if self.score is not None else 0.0
                except Exception as e:
                    self.error = str(e)
                    self.score = 0.0
                    self.success = False
                    self.reason = f"Error: {str(e)}"
                    return 0.0

            async def a_measure(self, test_case: LLMTestCase) -> float:
                """Asynchronous evaluation using Floeval criteria metric."""
                try:
                    floeval_sample = self._transform_test_case(test_case)
                    result = await self._floeval_metric.aevaluate(floeval_sample)
                    self._set_state_from_result(result)
                    return self.score if self.score is not None else 0.0
                except Exception as e:
                    self.error = str(e)
                    self.score = 0.0
                    self.success = False
                    self.reason = f"Error: {str(e)}"
                    return 0.0

            def is_successful(self) -> bool:
                """Check if evaluation passed."""
                if self.error is not None:
                    return False
                if self.success is not None:
                    return self.success
                if self.score is not None:
                    return self.score >= self.threshold
                return False

            @property
            def __name__(self):
                """Metric name property."""
                return metric_name

            def _transform_test_case(self, test_case: LLMTestCase) -> Sample:
                """Transform DeepEval test case to Floeval sample."""
                return Sample(
                    user_input=test_case.input or "",
                    llm_response=test_case.actual_output or "",
                    contexts=test_case.retrieval_context or [],
                    ground_truth=test_case.expected_output,
                )

            def _set_state_from_result(self, result: Any) -> None:
                """Extract state from Floeval MetricResult."""
                if isinstance(result, MetricResult):
                    self.score = result.score if result.score is not None else 0.0
                    self.reason = result.metadata.get("reason") or result.metadata.get("error", "")
                elif isinstance(result, (int, float)):
                    self.score = float(result)
                    self.reason = "Score computed"
                else:
                    self.score = 0.0
                    self.reason = "Unknown result type"

                self.score = max(0.0, min(1.0, self.score))
                self.success = self.score >= self.threshold

        return GeneratedDeepEvalCriteriaMetric

    def transform_sample(self, sample: Sample) -> LLMTestCase:
        """Transform Floeval sample to DeepEval LLMTestCase."""
        sample_dict = sample.model_dump()
        return LLMTestCase(
            input=sample_dict.get("user_input", ""),
            actual_output=sample_dict.get("llm_response", ""),
            expected_output=sample_dict.get("ground_truth"),
            retrieval_context=sample_dict.get("contexts", []),
        )

    def transform_dataset(self, dataset: Dataset) -> list[LLMTestCase]:
        """Transform Floeval dataset to list of DeepEval LLMTestCase."""
        test_cases = []
        for sample in dataset.samples:
            test_case = self.transform_sample(sample)
            test_cases.append(test_case)
        return test_cases
