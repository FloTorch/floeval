"""
RAGAS metric implementations with custom gateway support.

This module provides RAGAS metrics (answer_relevancy, faithfulness)
that can be configured with custom API gateways.
"""

import asyncio
import logging
import copy
from typing import Optional, Dict, Any

try:
    from ragas.metrics import answer_relevancy, faithfulness
except ImportError as e:
    raise ImportError(
        f"RAGAS not installed. Please install ragas>=0.4.3. Original error: {e}"
    )

from ...api.metrics.base import BaseMetric, MetricResult
from .adapter import (
    RAGASGatewayConfig,
    create_ragas_llm,
    create_ragas_embeddings,
    sample_to_ragas,
)

logger = logging.getLogger(__name__)

_EVAL_LOOP: Optional[asyncio.AbstractEventLoop] = None


def _run_async(coro: Any) -> Any:
    """
    Run an async coroutine from sync code without repeatedly closing event loops.

    Why:
    - `asyncio.run()` creates and closes a loop each call.
    - Some HTTP clients (httpx/anyio) can log `RuntimeError: Event loop is closed`
      during connection cleanup when loops are closed aggressively.

    Strategy:
    - If we're already inside a running loop (notebooks), use nest_asyncio if available.
    - Otherwise, create/reuse a module-level event loop and do NOT close it.
    """
    try:
        running = asyncio.get_running_loop()
        if running.is_running():
            try:
                import nest_asyncio  # type: ignore

                nest_asyncio.apply()
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "RAGAS metric called inside a running event loop. "
                    "Install `nest_asyncio` or use an async execution path."
                ) from e
            return running.run_until_complete(coro)
    except RuntimeError:
        # No running loop in this thread.
        pass

    global _EVAL_LOOP
    if _EVAL_LOOP is None or _EVAL_LOOP.is_closed():
        _EVAL_LOOP = asyncio.new_event_loop()
        asyncio.set_event_loop(_EVAL_LOOP)
    return _EVAL_LOOP.run_until_complete(coro)


class RAGASAnswerRelevancy(BaseMetric):
    """
    RAGAS Answer Relevancy metric with custom gateway support.
    
    Measures how relevant the generated answer is to the given question.
    Higher scores indicate more relevant answers.
    
    Args:
        gateway_config: Optional gateway configuration for custom API endpoint.
            If None, uses default RAGAS configuration (environment variables).
        threshold: Threshold for pass/fail determination (default: 0.7)
        name: Metric name (default: "answer_relevancy")
    """
    
    def __init__(
        self,
        gateway_config: Optional[RAGASGatewayConfig] = None,
        threshold: float = 0.7,
        name: str = "answer_relevancy",
        **kwargs: Any
    ):
        super().__init__(name=name)
        self.provider = "ragas"
        self.threshold = threshold
        self.gateway_config = gateway_config
        # NOTE: RAGAS exports metric instances; deepcopy to avoid shared-state mutations
        self.ragas_metric = copy.deepcopy(answer_relevancy)

        # Always initialize LLM/embeddings:
        # - if gateway_config is provided -> use it
        # - else -> rely on environment-driven defaults (e.g. OPENAI_API_KEY)
        try:
            cfg = gateway_config or RAGASGatewayConfig()
            self.llm = create_ragas_llm(cfg)
            self.embeddings = create_ragas_embeddings(cfg)
            # RAGAS 0.4.x expects llm/embeddings to be set on the metric object itself.
            # Do NOT pass llm/embeddings as kwargs to `single_turn_ascore`.
            self.ragas_metric.llm = self.llm
            if hasattr(self.ragas_metric, "embeddings"):
                self.ragas_metric.embeddings = self.embeddings
            logger.debug(
                "Initialized RAGAS Answer Relevancy (gateway_used=%s, base_url=%s)",
                bool(gateway_config),
                getattr(cfg, "gateway_base_url", None),
            )
        except Exception as e:
            logger.error(f"Failed to initialize RAGAS LLM/embeddings: {e}")
            raise
    
    def compute(self, sample: Any, **kwargs: Any) -> MetricResult:
        """
        Compute answer relevancy score for a sample.
        
        Args:
            sample: Floeval Sample object with inputs and ground_truth
            **kwargs: Additional arguments (unused)
            
        Returns:
            MetricResult with score and metadata
        """
        try:
            ragas_sample = sample_to_ragas(sample)

            score = _run_async(self.ragas_metric.single_turn_ascore(ragas_sample))
            
            score_float = float(score)
            passed = score_float >= self.threshold
            
            return MetricResult(
                score=score_float,
                metadata={
                    "threshold": self.threshold,
                    "passed": passed,
                    "provider": self.provider,
                    "metric_name": self.name,
                    "gateway_used": self.gateway_config is not None,
                },
            )
        except Exception as e:
            logger.error(f"Error computing answer relevancy: {e}", exc_info=True)
            # Return a failed result rather than raising
            return MetricResult(
                score=0.0,
                metadata={
                    "threshold": self.threshold,
                    "passed": False,
                    "provider": self.provider,
                    "metric_name": self.name,
                    "error": str(e),
                    "gateway_used": self.gateway_config is not None,
                },
            )


class RAGASFaithfulness(BaseMetric):
    """
    RAGAS Faithfulness metric with custom gateway support.
    
    Measures how grounded the generated answer is in the provided context.
    Higher scores indicate answers that are more faithful to the context.
    
    Args:
        gateway_config: Optional gateway configuration for custom API endpoint.
            If None, uses default RAGAS configuration (environment variables).
        threshold: Threshold for pass/fail determination (default: 0.7)
        name: Metric name (default: "faithfulness")
    """
    
    def __init__(
        self,
        gateway_config: Optional[RAGASGatewayConfig] = None,
        threshold: float = 0.7,
        name: str = "faithfulness",
        **kwargs: Any
    ):
        super().__init__(name=name)
        self.provider = "ragas"
        self.threshold = threshold
        self.gateway_config = gateway_config
        # NOTE: RAGAS exports metric instances; deepcopy to avoid shared-state mutations
        self.ragas_metric = copy.deepcopy(faithfulness)

        # Always initialize LLM/embeddings:
        # - if gateway_config is provided -> use it
        # - else -> rely on environment-driven defaults (e.g. OPENAI_API_KEY)
        try:
            cfg = gateway_config or RAGASGatewayConfig()
            self.llm = create_ragas_llm(cfg)
            self.embeddings = create_ragas_embeddings(cfg)
            # RAGAS 0.4.x expects llm/embeddings to be set on the metric object itself.
            self.ragas_metric.llm = self.llm
            if hasattr(self.ragas_metric, "embeddings"):
                self.ragas_metric.embeddings = self.embeddings
            logger.debug(
                "Initialized RAGAS Faithfulness (gateway_used=%s, base_url=%s)",
                bool(gateway_config),
                getattr(cfg, "gateway_base_url", None),
            )
        except Exception as e:
            logger.error(f"Failed to initialize RAGAS LLM/embeddings: {e}")
            raise
    
    def compute(self, sample: Any, **kwargs: Any) -> MetricResult:
        """
        Compute faithfulness score for a sample.
        
        Args:
            sample: Floeval Sample object with inputs and ground_truth
            **kwargs: Additional arguments (unused)
            
        Returns:
            MetricResult with score and metadata
        """
        try:
            ragas_sample = sample_to_ragas(sample)

            score = _run_async(self.ragas_metric.single_turn_ascore(ragas_sample))
            
            score_float = float(score)
            passed = score_float >= self.threshold
            
            return MetricResult(
                score=score_float,
                metadata={
                    "threshold": self.threshold,
                    "passed": passed,
                    "provider": self.provider,
                    "metric_name": self.name,
                    "gateway_used": self.gateway_config is not None,
                },
            )
        except Exception as e:
            logger.error(f"Error computing faithfulness: {e}", exc_info=True)
            # Return a failed result rather than raising
            return MetricResult(
                score=0.0,
                metadata={
                    "threshold": self.threshold,
                    "passed": False,
                    "provider": self.provider,
                    "metric_name": self.name,
                    "error": str(e),
                    "gateway_used": self.gateway_config is not None,
                },
            )
