"""
DeepEval metric implementations
"""

import logging
from typing import Any, Dict

from deepeval.evaluate import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    FaithfulnessMetric,
)

from floeval.api.metrics.base import BaseMetric, MetricResult
from floeval.config.schemas.io.dataset import Sample
from floeval.config.schemas.io.llm import LLMProviderConfig
from floeval.metric_providers.deepeval.adapter import DeepEvalAdapter, DeepEvalLLMAdapter

logger = logging.getLogger(__name__)

try:
    import nest_asyncio

    nest_asyncio.apply()
except ImportError:
    pass


class DeepEvalMetric(BaseMetric):
    """Base class for DeepEval metrics."""

    def __init__(self, llm_config: LLMProviderConfig | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.provider = "deepeval"
        self.llm_config = llm_config

        params = kwargs.get("params", {})
        if not isinstance(params, dict):
            params = {}

        self.adapter = DeepEvalAdapter(config=params)

        self._metric_params = self._extract_metric_params(params, kwargs)

        # Initialize LLM adapter internally (like RAGAS does)
        self._llm_adapter = self._create_llm_adapter(llm_config) if llm_config else None

    def _extract_metric_params(
        self, params: Dict[str, Any], kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract parameters that should be passed to DeepEval metric constructor.

        Excludes llm_config and other Floeval-specific params.
        Includes threshold, include_reason, async_mode, strict_mode, verbose_mode, etc.
        """
        metric_params = {}

        if isinstance(params, dict):
            metric_params.update(params)

        for key in [
            "threshold",
            "include_reason",
            "async_mode",
            "strict_mode",
            "verbose_mode",
            "evaluation_template",
            "truths_extraction_limit",
            "penalize_ambiguous_claims",
        ]:
            if key in kwargs and key not in metric_params:
                metric_params[key] = kwargs[key]

        return metric_params

    def _create_llm_adapter(self, llm_config: LLMProviderConfig) -> DeepEvalLLMAdapter:
        """Create and cache LLM adapter internally.

        Args:
            llm_config: LLM configuration (optional)

        Returns:
            DeepEvalLLMAdapter instance
        """
        model_name = llm_config.chat_model
        return DeepEvalLLMAdapter(model_name=model_name, config=llm_config)

    @property
    def config(self):
        return self.adapter.config

    def _run_evaluate(self, metrics, test_cases):
        """Run DeepEval evaluate with proper event loop handling.

        DeepEval's OpenAI client uses async operations internally, so we need
        to ensure consistent event loop usage to avoid conflicts.
        """
        try:
            try:
                import nest_asyncio

                nest_asyncio.apply()
            except ImportError:
                pass

            return evaluate(metrics=metrics, test_cases=test_cases).test_results[0]
        except RuntimeError as e:
            error_msg = str(e).lower()
            if (
                "bound to a different event loop" in error_msg
                or "event loop is closed" in error_msg
            ):
                logger.debug(f"Event loop error in DeepEval evaluate: {e}")
                try:
                    import nest_asyncio

                    nest_asyncio.apply()
                    return evaluate(metrics=metrics, test_cases=test_cases).test_results[0]
                except Exception as retry_error:
                    logger.error(
                        f"Failed to retry DeepEval evaluate after event loop error: {retry_error}"
                    )
                    raise RuntimeError(
                        "Event loop conflict in DeepEval evaluation. "
                        "Install `nest_asyncio` to resolve this issue."
                    ) from e
            raise

    def _extract_metric_result(self, result, metric_name: str) -> MetricResult:
        """Extract MetricResult from DeepEval evaluation result.

        Common error handling for all DeepEval metrics.
        """
        if result.metrics_data is None:
            return MetricResult(
                score=None,
                metadata={
                    "passed": False,
                    "provider": "deepeval",
                    "metric_name": metric_name,
                    "error": "Expected metrics_data in the result, but got None.",
                },
            )

        if len(result.metrics_data) == 0:
            return MetricResult(
                score=None,
                metadata={
                    "passed": False,
                    "provider": "deepeval",
                    "metric_name": metric_name,
                    "error": "Expected at least one metric data entry in the result, but got empty list.",
                },
            )

        try:
            if len(result.metrics_data) > 1:
                return MetricResult(
                    score=None,
                    metadata={
                        "passed": False,
                        "provider": "deepeval",
                        "metric_name": metric_name,
                        "error": f"Multiple metric results found ({len(result.metrics_data)}); only single metric expected.",
                    },
                )

            metric_data = result.metrics_data[0]
            score = metric_data.score
            passed = metric_data.success

            if score is None:
                return MetricResult(
                    score=None,
                    metadata={
                        "passed": False,
                        "provider": "deepeval",
                        "metric_name": metric_name,
                        "error": f"Expected a score value for metric: {metric_data.name}; success: {metric_data.success}",
                    },
                )

            metadata = {
                "provider": "deepeval",
                "metric_name": metric_name,
            }

            threshold = self._metric_params.get("threshold")
            if threshold is not None:
                metadata["threshold"] = threshold
                passed = score >= threshold
            else:
                metadata["threshold"] = None

            metadata["passed"] = passed

            return MetricResult(
                score=score,
                metadata=metadata,
            )
        except Exception as e:
            return MetricResult(
                score=None,
                metadata={
                    "passed": False,
                    "provider": "deepeval",
                    "metric_name": metric_name,
                    "error": str(e),
                },
            )


class FaithfulnessDeepEvalMetric(DeepEvalMetric):
    """Faithfulness metric implementation using DeepEval."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, name="faithfulness", **kwargs)

    def evaluate(self, sample: Sample, **kwargs) -> MetricResult:
        """Compute faithfulness metric score using DeepEval.

        Args:
            sample: Sample to evaluate
            **kwargs: Additional arguments (unused, kept for interface consistency)

        Returns:
            MetricResult: The result of the faithfulness metric computation
        """
        # Use internal adapter (initialized in __init__)
        if self._llm_adapter is None:
            raise ValueError(
                "LLM adapter not initialized. Provide llm_config when initializing Evaluation."
            )
        # Create metric instance with internal adapter and all metric params
        metric_kwargs = self._metric_params | {"model": self._llm_adapter}
        metric_instance = FaithfulnessMetric(**metric_kwargs)
        test_case = self.adapter.transform_test_case(
            metric_name="faithfulness", test_case_dict=sample.model_dump()
        )
        result = self._run_evaluate(metrics=[metric_instance], test_cases=[test_case])
        return self._extract_metric_result(result, "faithfulness")


class AnswerRelevancyDeepEvalMetric(DeepEvalMetric):
    """Answer Relevancy metric implementation using DeepEval."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, name="answer_relevancy", **kwargs)

    def evaluate(self, sample: Sample, **kwargs) -> MetricResult:
        """Compute answer relevancy metric score using DeepEval.

        Args:
            sample: Sample to evaluate
            **kwargs: Additional arguments (unused, kept for interface consistency)

        Returns:
            MetricResult: The result of the answer relevancy metric computation
        """
        # Use internal adapter (initialized in __init__)
        if self._llm_adapter is None:
            raise ValueError(
                "LLM adapter not initialized. Provide llm_config when initializing Evaluation."
            )
        # Create metric instance with internal adapter and all metric params
        metric_kwargs = self._metric_params | {"model": self._llm_adapter}
        metric_instance = AnswerRelevancyMetric(**metric_kwargs)
        test_case = self.adapter.transform_test_case(
            metric_name="answer_relevancy",
            test_case_dict=sample.model_dump(),
        )
        result = self._run_evaluate(metrics=[metric_instance], test_cases=[test_case])
        return self._extract_metric_result(result, "answer_relevancy")


class ContextualPrecisionDeepEvalMetric(DeepEvalMetric):
    """Contextual Precision metric implementation using DeepEval."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, name="contextual_precision", **kwargs)

    def evaluate(self, sample: Sample, **kwargs) -> MetricResult:
        """Compute contextual precision metric score using DeepEval.

        Args:
            sample: Sample to evaluate
            **kwargs: Additional arguments (unused, kept for interface consistency)

        Returns:
            MetricResult: The result of the contextual precision metric computation
        """
        # Use internal adapter (initialized in __init__)
        if self._llm_adapter is None:
            raise ValueError(
                "LLM adapter not initialized. Provide llm_config when initializing Evaluation."
            )
        # Create metric instance with internal adapter and all metric params
        metric_kwargs = self._metric_params | {"model": self._llm_adapter}
        metric_instance = ContextualPrecisionMetric(**metric_kwargs)
        test_case = self.adapter.transform_test_case(
            metric_name="contextual_precision",
            test_case_dict=sample.model_dump(),
        )
        result = self._run_evaluate(metrics=[metric_instance], test_cases=[test_case])
        return self._extract_metric_result(result, "contextual_precision")
