"""RAGAS metric implementations with custom LLM provider support.

This module provides RAGAS metrics (answer_relevancy, faithfulness, aspect_critic)
that can be configured with custom LLM providers.
"""

import copy
import logging
from typing import Any

from ragas.metrics import (
    NoiseSensitivity,
    answer_relevancy,
    context_entity_recall,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.metrics._aspect_critic import AspectCritic

from floeval.api.metrics.base import BaseMetric, MetricResult
from floeval.config.schemas.io.llm import LLMProviderConfig
from floeval.metric_providers.ragas.adapter import RAGASAdapter
from floeval.utils.asyncio_compat import run_coroutine_sync

logger = logging.getLogger(__name__)


class RAGASMetric(BaseMetric):
    """Base class for RAGAS metrics.

    Handles common initialization and computation patterns.
    """

    def __init__(
        self,
        ragas_metric_instance: Any,
        llm_config: LLMProviderConfig | None = None,
        adapter: RAGASAdapter | None = None,
        threshold: float | None = None,
        name: str = "ragas_metric",
        **kwargs: Any,
    ):
        super().__init__(name=name)
        self.provider = "ragas"
        self.llm_config = llm_config
        self._extra_headers: dict[str, str] = dict(kwargs.get("extra_headers") or {})

        if threshold is not None:
            self.threshold = threshold
        elif "threshold" in kwargs:
            self.threshold = kwargs.get("threshold")
        else:
            params = kwargs.get("params", {})
            self.threshold = params.get("threshold") if isinstance(params, dict) else None

        self.adapter = adapter or RAGASAdapter(
            config=llm_config, extra_headers=self._extra_headers or None
        )

        # NOTE: RAGAS exports metric instances; deepcopy to avoid shared-state mutations
        self.ragas_metric = copy.deepcopy(ragas_metric_instance)

        # RAGAS 0.4.x expects llm/embeddings to be set on the metric object itself.
        try:
            self.ragas_metric.llm = self.adapter.llm
            if hasattr(self.ragas_metric, "embeddings"):
                self.ragas_metric.embeddings = self.adapter.embeddings
            logger.debug(
                "Initialized RAGAS metric %s (llm_config=%s, base_url=%s)",
                self.name,
                bool(self.llm_config),
                self.llm_config.base_url if self.llm_config else None,
            )
        except Exception as e:
            logger.error("Failed to initialize RAGAS LLM/embeddings: %s", e)
            raise

    def _build_metadata(self, score_float: float, error: str | None = None) -> dict[str, Any]:
        """Build metadata dict with consistent structure."""
        metadata = {
            "provider": self.provider,
            "metric_name": self.name,
            "llm_config": self.llm_config is not None,
        }

        if error:
            metadata["error"] = error
            metadata["passed"] = False
        else:
            # Only compute pass/fail if threshold is provided
            if self.threshold is not None:
                metadata["threshold"] = self.threshold
                metadata["passed"] = score_float >= self.threshold

        return metadata


class RAGASAnswerRelevancy(RAGASMetric):
    """RAGAS Answer Relevancy metric with custom LLM provider support.

    Measures how relevant the generated answer is to the given question.
    Higher scores indicate more relevant answers.

    Args:
        llm_config: Optional LLM configuration for custom API endpoint.
            If None, uses default RAGAS configuration (environment variables).
        adapter: Optional RAGASAdapter instance (for reuse across metrics).
        threshold: Optional threshold for pass/fail determination.
            If None, only score is returned (no pass/fail).
        name: Metric name (default: "answer_relevancy")
    """

    def __init__(
        self,
        llm_config: LLMProviderConfig | None = None,
        adapter: RAGASAdapter | None = None,
        threshold: float | None = None,
        name: str = "answer_relevancy",
        **kwargs: Any,
    ):
        super().__init__(
            ragas_metric_instance=answer_relevancy,
            llm_config=llm_config,
            adapter=adapter,
            threshold=threshold,
            name=name,
            **kwargs,
        )

    def evaluate(self, sample: Any, **kwargs: Any) -> MetricResult:
        """Compute answer relevancy score for a sample (sync).

        Notebook-compatible sync path via coroutine runner fallback.
        Use aevaluate() inside async code for best performance.

        Args:
            sample: Floeval Sample object with inputs and ground_truth
            **kwargs: Additional arguments (unused)

        Returns:
            MetricResult with score and metadata
        """
        try:
            ragas_sample = self.adapter.transform_sample(sample)
            score = run_coroutine_sync(lambda: self.ragas_metric.single_turn_ascore(ragas_sample))
            score_float = float(score)
            return MetricResult(
                score=score_float,
                metadata=self._build_metadata(score_float),
            )
        except Exception as e:  # noqa: BLE001
            logger.error("Error computing answer relevancy: %s", e, exc_info=True)
            return MetricResult(
                score=None,
                metadata=self._build_metadata(0.0, error=str(e)),
            )

    async def aevaluate(self, sample: Any, **kwargs: Any) -> MetricResult:
        """Compute answer relevancy score for a sample (async)."""
        try:
            ragas_sample = self.adapter.transform_sample(sample)
            score = await self.ragas_metric.single_turn_ascore(ragas_sample)
            score_float = float(score)
            return MetricResult(
                score=score_float,
                metadata=self._build_metadata(score_float),
            )
        except Exception as e:  # noqa: BLE001
            logger.error("Error computing answer relevancy (async): %s", e, exc_info=True)
            return MetricResult(
                score=None,
                metadata=self._build_metadata(0.0, error=str(e)),
            )


class RAGASFaithfulness(RAGASMetric):
    """RAGAS Faithfulness metric with custom LLM provider support.

    Measures how grounded the generated answer is in the provided context.
    Higher scores indicate answers that are more faithful to the context.

    Args:
        llm_config: Optional LLM configuration for custom API endpoint.
            If None, uses default RAGAS configuration (environment variables).
        adapter: Optional RAGASAdapter instance (for reuse across metrics).
        threshold: Optional threshold for pass/fail determination.
            If None, only score is returned (no pass/fail).
        name: Metric name (default: "faithfulness")
    """

    def __init__(
        self,
        llm_config: LLMProviderConfig | None = None,
        adapter: RAGASAdapter | None = None,
        threshold: float | None = None,
        name: str = "faithfulness",
        **kwargs: Any,
    ):
        super().__init__(
            ragas_metric_instance=faithfulness,
            llm_config=llm_config,
            adapter=adapter,
            threshold=threshold,
            name=name,
            **kwargs,
        )

    def evaluate(self, sample: Any, **kwargs: Any) -> MetricResult:
        """Compute faithfulness score for a sample (sync).

        Notebook-compatible sync path via coroutine runner fallback.
        Use aevaluate() inside async code for best performance.

        Args:
            sample: Floeval Sample object with inputs and ground_truth
            **kwargs: Additional arguments (unused)

        Returns:
            MetricResult with score and metadata
        """
        try:
            ragas_sample = self.adapter.transform_sample(sample)
            score = run_coroutine_sync(lambda: self.ragas_metric.single_turn_ascore(ragas_sample))
            score_float = float(score)

            return MetricResult(
                score=score_float,
                metadata=self._build_metadata(score_float),
            )
        except Exception as e:
            logger.error("Error computing faithfulness: %s", e, exc_info=True)
            return MetricResult(
                score=None,
                metadata=self._build_metadata(0.0, error=str(e)),
            )

    async def aevaluate(self, sample: Any, **kwargs: Any) -> MetricResult:
        try:
            ragas_sample = self.adapter.transform_sample(sample)
            score = await self.ragas_metric.single_turn_ascore(ragas_sample)
            score_float = float(score)
            return MetricResult(
                score=score_float,
                metadata=self._build_metadata(score_float),
            )
        except Exception as e:
            logger.error("Error computing faithfulness (async): %s", e, exc_info=True)
            return MetricResult(
                score=None,
                metadata=self._build_metadata(0.0, error=str(e)),
            )


class RAGASContextPrecision(RAGASMetric):
    """RAGAS Context Precision metric with custom gateway support.

    Measures whether relevant retrieved contexts are ranked ahead of irrelevant ones.
    Higher scores indicate better retrieval ranking precision.

    Args:
        llm_config: Optional LLM configuration for custom API endpoint.
            If None, uses default RAGAS configuration (environment variables).
        adapter: Optional RAGASAdapter instance (for reuse across metrics).
        threshold: Optional threshold for pass/fail determination.
            If None, only score is returned (no pass/fail).
        name: Metric name (default: "context_precision")
    """

    def __init__(
        self,
        llm_config: LLMProviderConfig | None = None,
        adapter: RAGASAdapter | None = None,
        threshold: float | None = None,
        name: str = "context_precision",
        **kwargs: Any,
    ):
        super().__init__(
            ragas_metric_instance=context_precision,
            llm_config=llm_config,
            adapter=adapter,
            threshold=threshold,
            name=name,
            **kwargs,
        )

    def evaluate(self, sample: Any, **kwargs: Any) -> MetricResult:
        """Compute context precision score for a sample.

        Notebook-compatible sync path via coroutine runner fallback.

        Args:
            sample: Floeval Sample object with inputs and ground_truth
            **kwargs: Additional arguments (unused)

        Returns:
            MetricResult with score and metadata
        """
        try:
            ragas_sample = self.adapter.transform_sample(sample)
            score = run_coroutine_sync(lambda: self.ragas_metric.single_turn_ascore(ragas_sample))
            score_float = float(score)

            return MetricResult(
                score=score_float,
                metadata=self._build_metadata(score_float),
            )
        except Exception as e:
            logger.error("Error computing context precision: %s", e, exc_info=True)
            return MetricResult(
                score=None,
                metadata=self._build_metadata(0.0, error=str(e)),
            )

    async def aevaluate(self, sample: Any, **kwargs: Any) -> MetricResult:
        try:
            ragas_sample = self.adapter.transform_sample(sample)
            score = await self.ragas_metric.single_turn_ascore(ragas_sample)
            score_float = float(score)
            return MetricResult(
                score=score_float,
                metadata=self._build_metadata(score_float),
            )
        except Exception as e:
            logger.error("Error computing context precision (async): %s", e, exc_info=True)
            return MetricResult(
                score=None,
                metadata=self._build_metadata(0.0, error=str(e)),
            )


class RAGASContextRecall(RAGASMetric):
    """RAGAS Context Recall metric with custom gateway support.

    Measures how much relevant reference information is covered by retrieved contexts.
    Higher scores indicate better retrieval coverage (fewer missed relevant details).

    Args:
        llm_config: Optional LLM configuration for custom API endpoint.
            If None, uses default RAGAS configuration (environment variables).
        adapter: Optional RAGASAdapter instance (for reuse across metrics).
        threshold: Optional threshold for pass/fail determination.
            If None, only score is returned (no pass/fail).
        name: Metric name (default: "context_recall")
    """

    def __init__(
        self,
        llm_config: LLMProviderConfig | None = None,
        adapter: RAGASAdapter | None = None,
        threshold: float | None = None,
        name: str = "context_recall",
        **kwargs: Any,
    ):
        super().__init__(
            ragas_metric_instance=context_recall,
            llm_config=llm_config,
            adapter=adapter,
            threshold=threshold,
            name=name,
            **kwargs,
        )

    def evaluate(self, sample: Any, **kwargs: Any) -> MetricResult:
        """Compute context recall score for a sample.

        Notebook-compatible sync path via coroutine runner fallback.

        Args:
            sample: Floeval Sample object with inputs and ground_truth
            **kwargs: Additional arguments (unused)

        Returns:
            MetricResult with score and metadata
        """
        try:
            ragas_sample = self.adapter.transform_sample(sample)
            score = run_coroutine_sync(lambda: self.ragas_metric.single_turn_ascore(ragas_sample))
            score_float = float(score)

            return MetricResult(
                score=score_float,
                metadata=self._build_metadata(score_float),
            )
        except Exception as e:
            logger.error("Error computing context recall: %s", e, exc_info=True)
            return MetricResult(
                score=None,
                metadata=self._build_metadata(0.0, error=str(e)),
            )

    async def aevaluate(self, sample: Any, **kwargs: Any) -> MetricResult:
        try:
            ragas_sample = self.adapter.transform_sample(sample)
            score = await self.ragas_metric.single_turn_ascore(ragas_sample)
            score_float = float(score)
            return MetricResult(
                score=score_float,
                metadata=self._build_metadata(score_float),
            )
        except Exception as e:
            logger.error("Error computing context recall (async): %s", e, exc_info=True)
            return MetricResult(
                score=None,
                metadata=self._build_metadata(0.0, error=str(e)),
            )


class RAGASContextEntityRecall(RAGASMetric):
    """RAGAS Context Entity Recall metric with custom gateway support.

    Measures how well entities in the reference are covered by retrieved contexts.
    Higher scores indicate stronger entity-level retrieval coverage.

    Args:
        llm_config: Optional LLM configuration for custom API endpoint.
            If None, uses default RAGAS configuration (environment variables).
        adapter: Optional RAGASAdapter instance (for reuse across metrics).
        threshold: Optional threshold for pass/fail determination.
            If None, only score is returned (no pass/fail).
        name: Metric name (default: "context_entity_recall")
    """

    def __init__(
        self,
        llm_config: LLMProviderConfig | None = None,
        adapter: RAGASAdapter | None = None,
        threshold: float | None = None,
        name: str = "context_entity_recall",
        **kwargs: Any,
    ):
        super().__init__(
            ragas_metric_instance=context_entity_recall,
            llm_config=llm_config,
            adapter=adapter,
            threshold=threshold,
            name=name,
            **kwargs,
        )

    def evaluate(self, sample: Any, **kwargs: Any) -> MetricResult:
        """Compute context entity recall score for a sample.

        Notebook-compatible sync path via coroutine runner fallback.

        Args:
            sample: Floeval Sample object with inputs and ground_truth
            **kwargs: Additional arguments (unused)

        Returns:
            MetricResult with score and metadata
        """
        try:
            ragas_sample = self.adapter.transform_sample(sample)
            score = run_coroutine_sync(lambda: self.ragas_metric.single_turn_ascore(ragas_sample))
            score_float = float(score)

            return MetricResult(
                score=score_float,
                metadata=self._build_metadata(score_float),
            )
        except Exception as e:
            logger.error("Error computing context entity recall: %s", e, exc_info=True)
            return MetricResult(
                score=None,
                metadata=self._build_metadata(0.0, error=str(e)),
            )

    async def aevaluate(self, sample: Any, **kwargs: Any) -> MetricResult:
        try:
            ragas_sample = self.adapter.transform_sample(sample)
            score = await self.ragas_metric.single_turn_ascore(ragas_sample)
            score_float = float(score)
            return MetricResult(
                score=score_float,
                metadata=self._build_metadata(score_float),
            )
        except Exception as e:
            logger.error("Error computing context entity recall (async): %s", e, exc_info=True)
            return MetricResult(
                score=None,
                metadata=self._build_metadata(0.0, error=str(e)),
            )


class RAGASNoiseSensitivity(RAGASMetric):
    """RAGAS Noise Sensitivity metric with custom gateway support.

    Measures how often the response includes incorrect claims attributable to
    retrieved context noise. Lower scores indicate better performance.

    Args:
        llm_config: Optional LLM configuration for custom API endpoint.
            If None, uses default RAGAS configuration (environment variables).
        adapter: Optional RAGASAdapter instance (for reuse across metrics).
        threshold: Optional threshold for pass/fail determination.
            If None, only score is returned (no pass/fail).
        name: Metric name (default: "noise_sensitivity")
    """

    def __init__(
        self,
        llm_config: LLMProviderConfig | None = None,
        adapter: RAGASAdapter | None = None,
        threshold: float | None = None,
        name: str = "noise_sensitivity",
        **kwargs: Any,
    ):
        super().__init__(
            ragas_metric_instance=NoiseSensitivity(),
            llm_config=llm_config,
            adapter=adapter,
            threshold=threshold,
            name=name,
            **kwargs,
        )

    def evaluate(self, sample: Any, **kwargs: Any) -> MetricResult:
        """Compute noise sensitivity score for a sample.

        Notebook-compatible sync path via coroutine runner fallback.

        Args:
            sample: Floeval Sample object with inputs and ground_truth
            **kwargs: Additional arguments (unused)

        Returns:
            MetricResult with score and metadata
        """
        try:
            ragas_sample = self.adapter.transform_sample(sample)
            score = run_coroutine_sync(lambda: self.ragas_metric.single_turn_ascore(ragas_sample))
            score_float = float(score)

            return MetricResult(
                score=score_float,
                metadata=self._build_metadata(score_float),
            )
        except Exception as e:
            logger.error("Error computing noise sensitivity: %s", e, exc_info=True)
            return MetricResult(
                score=None,
                metadata=self._build_metadata(0.0, error=str(e)),
            )

    async def aevaluate(self, sample: Any, **kwargs: Any) -> MetricResult:
        try:
            ragas_sample = self.adapter.transform_sample(sample)
            score = await self.ragas_metric.single_turn_ascore(ragas_sample)
            score_float = float(score)
            return MetricResult(
                score=score_float,
                metadata=self._build_metadata(score_float),
            )
        except Exception as e:
            logger.error("Error computing noise sensitivity (async): %s", e, exc_info=True)
            return MetricResult(
                score=None,
                metadata=self._build_metadata(0.0, error=str(e)),
            )


class RAGASAspectCritic(RAGASMetric):
    """RAGAS Aspect Critic metric with custom gateway support.

    Evaluates whether a response satisfies a user-defined aspect definition.
    Returns a binary score (0.0 or 1.0).
    """

    def __init__(
        self,
        llm_config: LLMProviderConfig | None = None,
        adapter: RAGASAdapter | None = None,
        definition: str | None = None,
        aspect_name: str | None = None,
        strictness: int | None = None,
        threshold: float | None = None,
        name: str = "aspect_critic",
        **kwargs: Any,
    ):
        params = kwargs.get("params", {})
        resolved_definition = definition or (
            params.get("definition") if isinstance(params, dict) else None
        )
        if not resolved_definition:
            raise ValueError("RAGAS aspect_critic requires a non-empty 'definition'.")

        resolved_aspect_name = aspect_name or (
            params.get("aspect_name") if isinstance(params, dict) else None
        )
        resolved_strictness = strictness
        if resolved_strictness is None and isinstance(params, dict):
            resolved_strictness = params.get("strictness")

        ragas_kwargs: dict[str, Any] = {
            "name": resolved_aspect_name or name,
            "definition": resolved_definition,
        }
        if resolved_strictness is not None:
            ragas_kwargs["strictness"] = resolved_strictness

        super().__init__(
            ragas_metric_instance=AspectCritic(**ragas_kwargs),
            llm_config=llm_config,
            adapter=adapter,
            threshold=threshold,
            name=name,
            **kwargs,
        )
        # aspect_critic uses PydanticPrompt which requires InstructorBaseRagasLLM
        # for structured output. Override the plain LangchainLLMWrapper set by
        # the base class with the structured instructor LLM.
        self.ragas_metric.llm = self.adapter.agent_llm

    def evaluate(self, sample: Any, **kwargs: Any) -> MetricResult:
        try:
            ragas_sample = self.adapter.transform_sample(sample)
            score = run_coroutine_sync(lambda: self.ragas_metric.single_turn_ascore(ragas_sample))
            score_float = float(score)

            return MetricResult(
                score=score_float,
                metadata=self._build_metadata(score_float),
            )
        except Exception as e:  # noqa: BLE001
            logger.error("Error computing aspect_critic: %s", e, exc_info=True)
            return MetricResult(
                score=None,
                metadata=self._build_metadata(0.0, error=str(e)),
            )

    async def aevaluate(self, sample: Any, **kwargs: Any) -> MetricResult:
        try:
            ragas_sample = self.adapter.transform_sample(sample)
            score = await self.ragas_metric.single_turn_ascore(ragas_sample)
            score_float = float(score)
            return MetricResult(
                score=score_float,
                metadata=self._build_metadata(score_float),
            )
        except Exception as e:  # noqa: BLE001
            logger.error("Error computing aspect_critic (async): %s", e, exc_info=True)
            return MetricResult(
                score=None,
                metadata=self._build_metadata(0.0, error=str(e)),
            )
