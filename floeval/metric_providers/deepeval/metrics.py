"""
DeepEval metric implementations
"""

import logging
from typing import Any, Dict

from deepeval.evaluate import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    ExactMatchMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    JsonCorrectnessMetric,
    PatternMatchMetric,
    ToxicityMetric,
)

from floeval.api.metrics.base import BaseMetric, MetricResult
from floeval.config.schemas.io.dataset import Sample
from floeval.config.schemas.io.llm import LLMProviderConfig
from floeval.metric_providers.deepeval.adapter import (
    DeepEvalAdapter,
    DeepEvalLLMAdapter,
)

logger = logging.getLogger(__name__)


class DeepEvalMetric(BaseMetric):
    """Base class for DeepEval metrics."""

    def __init__(self, llm_config: LLMProviderConfig | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.provider = "deepeval"
        self.llm_config = llm_config
        self._extra_headers: Dict[str, Any] = dict(kwargs.get("extra_headers") or {})

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
            "expected_schema",
            "pattern",
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
        return DeepEvalLLMAdapter(
            model_name=model_name,
            config=llm_config,
            extra_headers=self._extra_headers or None,
        )

    @property
    def config(self):
        return self.adapter.config

    def _run_evaluate(self, metrics, test_cases):
        """Run DeepEval evaluate synchronously.

        Safe to call from sync context. When called from async path (e.g.
        Evaluation.arun), the orchestrator runs this via run_in_executor.
        """
        return evaluate(metrics=metrics, test_cases=test_cases).test_results[0]

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
                    "error": "Expected at least one metric data entry, got empty list.",
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
                        "error": f"Multiple metric results ({len(result.metrics_data)}); only single metric expected.",
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


class ContextualRecallDeepEvalMetric(DeepEvalMetric):
    """Contextual Recall metric implementation using DeepEval."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, name="contextual_recall", **kwargs)

    def evaluate(self, sample: Sample, **kwargs) -> MetricResult:
        """Compute contextual recall metric score using DeepEval.

        Args:
            sample: Sample to evaluate
            **kwargs: Additional arguments (unused, kept for interface consistency)

        Returns:
            MetricResult: The result of the contextual recall metric computation
        """
        # Use internal adapter (initialized in __init__)
        if self._llm_adapter is None:
            raise ValueError(
                "LLM adapter not initialized. Provide llm_config when initializing Evaluation."
            )
        # Create metric instance with internal adapter and all metric params
        metric_kwargs = self._metric_params | {"model": self._llm_adapter}
        metric_instance = ContextualRecallMetric(**metric_kwargs)
        test_case = self.adapter.transform_test_case(
            metric_name="contextual_recall",
            test_case_dict=sample.model_dump(),
        )
        result = self._run_evaluate(metrics=[metric_instance], test_cases=[test_case])
        return self._extract_metric_result(result, "contextual_recall")


class ContextualRelevancyDeepEvalMetric(DeepEvalMetric):
    """Contextual Relevancy metric implementation using DeepEval."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, name="contextual_relevancy", **kwargs)

    def evaluate(self, sample: Sample, **kwargs) -> MetricResult:
        """Compute contextual relevancy metric score using DeepEval.

        Args:
            sample: Sample to evaluate
            **kwargs: Additional arguments (unused, kept for interface consistency)

        Returns:
            MetricResult: The result of the contextual relevancy metric computation
        """
        # Use internal adapter (initialized in __init__)
        if self._llm_adapter is None:
            raise ValueError(
                "LLM adapter not initialized. Provide llm_config when initializing Evaluation."
            )
        # Create metric instance with internal adapter and all metric params
        metric_kwargs = self._metric_params | {"model": self._llm_adapter}
        metric_instance = ContextualRelevancyMetric(**metric_kwargs)
        test_case = self.adapter.transform_test_case(
            metric_name="contextual_relevancy",
            test_case_dict=sample.model_dump(),
        )
        result = self._run_evaluate(metrics=[metric_instance], test_cases=[test_case])
        return self._extract_metric_result(result, "contextual_relevancy")


class HallucinationDeepEvalMetric(DeepEvalMetric):
    """Hallucination metric implementation using DeepEval."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, name="hallucination", **kwargs)

    def evaluate(self, sample: Sample, **kwargs) -> MetricResult:
        """Compare actual_output against context to detect factual contradictions."""
        if self._llm_adapter is None:
            raise ValueError(
                "LLM adapter not initialized. Provide llm_config when initializing Evaluation."
            )
        metric_kwargs = self._metric_params | {"model": self._llm_adapter}
        metric_instance = HallucinationMetric(**metric_kwargs)
        test_case = self.adapter.transform_test_case(
            metric_name="hallucination",
            test_case_dict=sample.model_dump(),
        )
        result = self._run_evaluate(metrics=[metric_instance], test_cases=[test_case])
        return self._extract_metric_result(result, "hallucination")


class ToxicityDeepEvalMetric(DeepEvalMetric):
    """Toxicity metric implementation using DeepEval."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, name="toxicity", **kwargs)

    def evaluate(self, sample: Sample, **kwargs) -> MetricResult:
        """Extract opinions from actual_output and classify each as toxic or not."""
        if self._llm_adapter is None:
            raise ValueError(
                "LLM adapter not initialized. Provide llm_config when initializing Evaluation."
            )
        metric_kwargs = self._metric_params | {"model": self._llm_adapter}
        metric_instance = ToxicityMetric(**metric_kwargs)
        test_case = self.adapter.transform_test_case(
            metric_name="toxicity",
            test_case_dict=sample.model_dump(),
        )
        result = self._run_evaluate(metrics=[metric_instance], test_cases=[test_case])
        return self._extract_metric_result(result, "toxicity")


class ExactMatchDeepEvalMetric(DeepEvalMetric):
    """Exact match metric implementation using DeepEval."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, name="exact_match", **kwargs)

    def evaluate(self, sample: Sample, **kwargs) -> MetricResult:
        """Check if actual_output exactly matches expected_output."""
        metric_instance = ExactMatchMetric(**self._metric_params)
        test_case = self.adapter.transform_test_case(
            metric_name="exact_match",
            test_case_dict=sample.model_dump(),
        )
        result = self._run_evaluate(metrics=[metric_instance], test_cases=[test_case])
        return self._extract_metric_result(result, "exact_match")


class PatternMatchDeepEvalMetric(DeepEvalMetric):
    """Pattern match metric implementation using DeepEval."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, name="pattern_match", **kwargs)

    def evaluate(self, sample: Sample, **kwargs) -> MetricResult:
        """Check if actual_output matches the configured regex pattern."""
        metric_instance = PatternMatchMetric(**self._metric_params)
        test_case = self.adapter.transform_test_case(
            metric_name="pattern_match",
            test_case_dict=sample.model_dump(),
        )
        result = self._run_evaluate(metrics=[metric_instance], test_cases=[test_case])
        return self._extract_metric_result(result, "pattern_match")


class JsonCorrectnessDeepEvalMetric(DeepEvalMetric):
    """JSON correctness metric implementation using DeepEval."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, name="json_correctness", **kwargs)

    def evaluate(self, sample: Sample, **kwargs) -> MetricResult:
        """Validate actual_output against the configured expected JSON schema."""
        metric_kwargs = dict(self._metric_params)
        if self._llm_adapter is not None:
            metric_kwargs["model"] = self._llm_adapter
        metric_instance = JsonCorrectnessMetric(**metric_kwargs)
        test_case = self.adapter.transform_test_case(
            metric_name="json_correctness",
            test_case_dict=sample.model_dump(),
        )
        result = self._run_evaluate(metrics=[metric_instance], test_cases=[test_case])
        return self._extract_metric_result(result, "json_correctness")


class TaskCompletionGEvalMetric(DeepEvalMetric):
    """DeepEval G-Eval metric for agent task completion.

    Uses DeepEval's G-Eval framework (criteria-based LLM judge) to evaluate
    whether the agent completed the assigned task.

    Registered as: deepeval:task_completion_geval

    Preferred over builtin:task_completion when: user wants DeepEval's specific
    G-Eval chain-of-thought evaluation methodology.
    Prefer builtin:task_completion when: using custom gateway LLMs.

    Requires: AgentSample with trace.final_response.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, name="task_completion_geval", **kwargs)

    def evaluate(self, sample: "AgentSample", **kwargs) -> MetricResult:  # type: ignore[override]
        try:
            from deepeval.metrics import GEval
            from deepeval.test_case import LLMTestCaseParams
        except ImportError:
            return MetricResult(
                score=None,
                metadata={
                    "error": "deepeval GEval not available in installed version",
                    "provider": "deepeval",
                },
            )

        if self._llm_adapter is None:
            raise ValueError(
                "LLM adapter not initialized. Provide llm_config when initializing Evaluation."
            )

        metric_kwargs: dict = {
            "name": "Task Completion",
            "criteria": (
                "Determine whether the AI agent successfully completed the user's task "
                "based on the final response."
            ),
            "evaluation_params": [
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            "model": self._llm_adapter,
        }
        # Forward threshold if provided
        for k in ("threshold", "strict_mode", "verbose_mode"):
            if k in self._metric_params:
                metric_kwargs[k] = self._metric_params[k]

        try:
            metric_instance = GEval(**metric_kwargs)
            test_case = self.adapter.transform_agent_sample(sample)
            result = self._run_evaluate(metrics=[metric_instance], test_cases=[test_case])
            return self._extract_metric_result(result, "task_completion_geval")
        except Exception as e:
            logger.error("TaskCompletionGEvalMetric failed: %s", e, exc_info=True)
            return MetricResult(
                score=None,
                metadata={"error": str(e), "provider": "deepeval"},
            )


class ToolCorrectnessDeepEvalMetric(DeepEvalMetric):
    """DeepEval ToolCorrectnessMetric for agent tool call evaluation.

    Checks whether the agent called the correct tools with the correct arguments
    compared to reference_tool_calls.

    Registered as: deepeval:tool_correctness

    Requires: reference_tool_calls in the AgentSample.
    Returns score=None if reference_tool_calls not provided.

    Preferred over builtin:tool_selection_accuracy when: tool argument semantic
    matching (not exact string match) is needed.
    Requires deepeval>=3.0 for ToolCorrectnessMetric.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, name="tool_correctness", **kwargs)

    def evaluate(self, sample: "AgentSample", **kwargs) -> MetricResult:  # type: ignore[override]
        if not sample.reference_tool_calls:
            return MetricResult(
                score=None,
                metadata={
                    "reason": "reference_tool_calls required for deepeval:tool_correctness",
                    "provider": "deepeval",
                },
            )

        try:
            from deepeval.metrics import ToolCorrectnessMetric  # type: ignore[import]
        except ImportError:
            return MetricResult(
                score=None,
                metadata={
                    "error": (
                        "deepeval ToolCorrectnessMetric not available. "
                        "Upgrade deepeval>=3.0 to use this metric."
                    ),
                    "provider": "deepeval",
                },
            )

        # Build DeepEval tool call objects — guard against missing ToolCall class
        try:
            from deepeval.test_case import ToolCall as DeepEvalToolCall  # type: ignore[import]

            expected_tools = [
                DeepEvalToolCall(name=tc.name, input_parameters=tc.args or {})
                for tc in sample.reference_tool_calls
            ]
            actual_tools = [
                DeepEvalToolCall(name=tc.name, input_parameters=tc.args or {})
                for tc in (sample.trace.tool_calls_made if sample.trace else [])
            ]
        except (ImportError, AttributeError):
            return MetricResult(
                score=None,
                metadata={
                    "error": (
                        "deepeval.test_case.ToolCall not available. "
                        "Upgrade deepeval>=3.0 to use this metric."
                    ),
                    "provider": "deepeval",
                },
            )

        try:
            metric_instance = ToolCorrectnessMetric(**self._metric_params)
            test_case = self.adapter.transform_agent_sample(sample)

            # Attach tool call lists to test case only if the attributes exist
            if hasattr(test_case, "tools_called"):
                test_case.tools_called = actual_tools
            if hasattr(test_case, "expected_tools"):
                test_case.expected_tools = expected_tools

            result = self._run_evaluate(metrics=[metric_instance], test_cases=[test_case])
            return self._extract_metric_result(result, "tool_correctness")
        except Exception as e:
            logger.error("ToolCorrectnessDeepEvalMetric failed: %s", e, exc_info=True)
            return MetricResult(
                score=None,
                metadata={"error": str(e), "provider": "deepeval"},
            )
