"""
DeepEval metric implementations
"""

import asyncio
import logging
from abc import abstractmethod

from deepeval.evaluate import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

from typing import Optional, Dict, Any

from floeval.api.dataset import Sample
from floeval.api.metrics.base import BaseMetric, MetricResult
from floeval.config import GatewayConfig
from floeval.metric_providers.deepeval.adapter import DeepEvalAdapter, DeepEvalLLMAdapter

logger = logging.getLogger(__name__)

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

__VALID_DEEPEVAL_METRICS__ = {
    "faithfulness": FaithfulnessMetric,
}


class DeepEvalMetric(BaseMetric):
    """
    Base class for DeepEval metrics.
    """

    def __init__(self, gateway_config: Optional[GatewayConfig] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.provider = "deepeval"
        self.gateway_config = gateway_config
        
        params = kwargs.get("params", {})
        if not isinstance(params, dict):
            params = {}
        
        self.adapter = DeepEvalAdapter(config=params)
        
        self._metric_params = self._extract_metric_params(params, kwargs)
        
        # Initialize LLM adapter internally (like RAGAS does)
        self._llm_adapter = self._create_llm_adapter(gateway_config)
    
    def _extract_metric_params(self, params: Dict[str, Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract parameters that should be passed to DeepEval metric constructor.
        
        Excludes gateway_config and other Floeval-specific params.
        Includes threshold, include_reason, async_mode, strict_mode, verbose_mode, etc.
        """
        metric_params = {}
        
        if isinstance(params, dict):
            metric_params.update(params)
        
        for key in ["threshold", "include_reason", "async_mode", "strict_mode", "verbose_mode", 
                    "evaluation_template", "truths_extraction_limit", "penalize_ambiguous_claims"]:
            if key in kwargs and key not in metric_params:
                metric_params[key] = kwargs[key]
        
        return metric_params
    
    def _create_llm_adapter(self, gateway_config: Optional[GatewayConfig]) -> Optional[DeepEvalLLMAdapter]:
        """
        Create and cache LLM adapter internally.
        
        Args:
            gateway_config: Gateway configuration (optional)
            
        Returns:
            DeepEvalLLMAdapter instance or None if no config provided
        """
        if gateway_config:
            model_name = gateway_config.llm_model or "default"
            return DeepEvalLLMAdapter(model_name=model_name, config=gateway_config)
        return None  # Will use DeepEval defaults if available

    @property
    def config(self):
        return self.adapter.config

    def _run_evaluate(self, metrics, test_cases):
        """
        Run DeepEval evaluate with proper event loop handling.
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
            if "bound to a different event loop" in error_msg or "event loop is closed" in error_msg:
                logger.debug(f"Event loop error in DeepEval evaluate: {e}")
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                    return evaluate(metrics=metrics, test_cases=test_cases).test_results[0]
                except Exception as retry_error:
                    logger.error(f"Failed to retry DeepEval evaluate after event loop error: {retry_error}")
                    raise RuntimeError(
                        "Event loop conflict in DeepEval evaluation. "
                        "Install `nest_asyncio` to resolve this issue."
                    ) from e
            raise

    def _extract_metric_result(self, result, metric_name: str) -> MetricResult:
        """
        Extract MetricResult from DeepEval evaluation result.
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

    @abstractmethod
    def compute(self, *args, **kwargs) -> MetricResult:
        """
        Compute DeepEval metric score.
        """


class FaithfulnessDeepEvalMetric(DeepEvalMetric):
    """
    Faithfulness metric implementation using DeepEval.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, name="faithfulness", **kwargs)
        # Cache metric instance to avoid recreating on each compute() call
        self._metric_class = FaithfulnessMetric

    def compute(self, sample: Sample, **kwargs) -> MetricResult:
        """
        Compute faithfulness metric score using DeepEval.

        Args:
            sample: Sample to evaluate
            **kwargs: Additional arguments (unused, kept for interface consistency)

        Returns:
            MetricResult: The result of the faithfulness metric computation
        """
        # Use internal adapter (initialized in __init__)
        if self._llm_adapter is None:
            raise ValueError(
                "LLM adapter not initialized. Provide gateway_config when creating metric."
            )
        # Create metric instance with internal adapter and all metric params
        metric_kwargs = {"model": self._llm_adapter}
        metric_kwargs.update(self._metric_params)
        metric_instance = self._metric_class(**metric_kwargs)
        test_case = self.adapter.transform_test_case(
            metric_name="faithfulness",
            test_case_dict=sample.inputs
            | {"expected_output": (sample.ground_truth or {}).get("expected_answer", "")},
        )
        result = self._run_evaluate(metrics=[metric_instance], test_cases=[test_case])
        return self._extract_metric_result(result, "faithfulness")


class AnswerRelevancyDeepEvalMetric(DeepEvalMetric):
    """
    Answer Relevancy metric implementation using DeepEval.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, name="answer_relevancy", **kwargs)
        # Cache metric class to avoid recreating on each compute() call
        self._metric_class = AnswerRelevancyMetric

    def compute(self, sample: Sample, **kwargs) -> MetricResult:
        """
        Compute answer relevancy metric score using DeepEval.

        Args:
            sample: Sample to evaluate
            **kwargs: Additional arguments (unused, kept for interface consistency)

        Returns:
            MetricResult: The result of the answer relevancy metric computation
        """
        # Use internal adapter (initialized in __init__)
        if self._llm_adapter is None:
            raise ValueError(
                "LLM adapter not initialized. Provide gateway_config when creating metric."
            )
        # Create metric instance with internal adapter and all metric params
        metric_kwargs = {"model": self._llm_adapter}
        metric_kwargs.update(self._metric_params)
        metric_instance = self._metric_class(**metric_kwargs)
        test_case = self.adapter.transform_test_case(
            metric_name="answer_relevancy",
            test_case_dict=sample.inputs,
        )
        result = self._run_evaluate(metrics=[metric_instance], test_cases=[test_case])
        return self._extract_metric_result(result, "answer_relevancy")
