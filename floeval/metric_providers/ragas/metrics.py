"""RAGAS metric implementations with custom llm providers support.

This module provides RAGAS metrics (answer_relevancy, faithfulness)
that can be configured with custom LLM providers.
"""

import asyncio
import copy
import logging
from typing import Any, Dict, Optional

from floeval.config.schemas.io.llm import LLMProviderConfig

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

from ragas.metrics import answer_relevancy, faithfulness

from floeval.api.metrics.base import BaseMetric, MetricResult
from floeval.metric_providers.ragas.adapter import RAGASAdapter

logger = logging.getLogger(__name__)

def _run_async(coro: Any) -> Any:
    try:
        running = asyncio.get_running_loop()
        if running.is_running():
            try:
                import nest_asyncio
                nest_asyncio.apply()
                return running.run_until_complete(coro)
            except ImportError:
                raise RuntimeError(
                    "RAGAS metric called inside a running event loop. "
                    "Install `nest_asyncio` or use an async execution path."
                )
    except RuntimeError:
        pass

    try:
        return asyncio.run(coro)
    except RuntimeError as e:
        error_msg = str(e).lower()
        if "event loop is closed" in error_msg or "bound to a different event loop" in error_msg:
            logger.debug(f"Suppressed event loop cleanup error: {e}")
            raise RuntimeError(
                "Event loop cleanup error occurred. "
                "This is usually harmless - the evaluation may have completed successfully."
            ) from e
        raise


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

        if threshold is not None:
            self.threshold = threshold
        elif "threshold" in kwargs:
            self.threshold = kwargs.get("threshold")
        else:
            params = kwargs.get("params", {})
            self.threshold = params.get("threshold") if isinstance(params, dict) else None

        # Use provided adapter or create new one
        self.adapter = adapter or RAGASAdapter(config=llm_config)

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
            logger.error(f"Failed to initialize RAGAS LLM/embeddings: {e}")
            raise

    def _build_metadata(self, score_float: float, error: Optional[str] = None) -> Dict[str, Any]:
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
    """RAGAS Answer Relevancy metric with custom gateway support.

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
        """Compute answer relevancy score for a sample.

        Args:
            sample: Floeval Sample object with inputs and ground_truth
            **kwargs: Additional arguments (unused)

        Returns:
            MetricResult with score and metadata
        """
        try:
            ragas_sample = self.adapter.transform_sample(sample)
            score = _run_async(self.ragas_metric.single_turn_ascore(ragas_sample))
            score_float = float(score)

            return MetricResult(
                score=score_float,
                metadata=self._build_metadata(score_float),
            )
        except Exception as e:
            logger.error(f"Error computing answer relevancy: {e}", exc_info=True)
            return MetricResult(
                score=None,
                metadata=self._build_metadata(0.0, error=str(e)),
            )


class RAGASFaithfulness(RAGASMetric):
    """RAGAS Faithfulness metric with custom gateway support.

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
        """Compute faithfulness score for a sample.

        Args:
            sample: Floeval Sample object with inputs and ground_truth
            **kwargs: Additional arguments (unused)

        Returns:
            MetricResult with score and metadata
        """
        try:
            ragas_sample = self.adapter.transform_sample(sample)
            score = _run_async(self.ragas_metric.single_turn_ascore(ragas_sample))
            score_float = float(score)

            return MetricResult(
                score=score_float,
                metadata=self._build_metadata(score_float),
            )
        except Exception as e:
            logger.error(f"Error computing faithfulness: {e}", exc_info=True)
            return MetricResult(
                score=None,
                metadata=self._build_metadata(0.0, error=str(e)),
            )
