"""Builtin core metrics: ExactMatch and SemanticSim.

These were previously stub implementations (compute() methods that did nothing
and were not registered). Now they are fully implemented.

Note on ambiguity: deepeval:exact_match also exists. When requesting exact_match
without a provider prefix, MetricRegistry.resolve_best() will raise an ambiguity
error. Always use the full form: builtin:exact_match or deepeval:exact_match.
"""

from __future__ import annotations

import logging

from floeval.api.metrics.base import BaseMetric, MetricResult
from floeval.api.metrics.registry import MetricRegistry

logger = logging.getLogger(__name__)


class ExactMatch(BaseMetric):
    """Exact string match between the response and the ground truth / reference.

    Works with both RAG samples (llm_response vs ground_truth) and agent
    samples (trace.final_response vs reference_outcome).
    Score: 1.0 if strings match, 0.0 otherwise.
    """

    def __init__(self, case_sensitive: bool = False, **kwargs):
        super().__init__(name="exact_match", **kwargs)
        self.provider = "builtin"
        self.case_sensitive = case_sensitive

    def evaluate(self, sample, **kwargs) -> MetricResult:
        response = getattr(sample, "llm_response", None)
        if response is None and hasattr(sample, "trace") and sample.trace:
            response = sample.trace.final_response

        ground_truth = getattr(sample, "ground_truth", None)
        if ground_truth is None:
            ground_truth = getattr(sample, "reference_outcome", None)

        if response is None or ground_truth is None:
            return MetricResult(
                score=None,
                metadata={"error": "response and ground_truth required", "provider": "builtin"},
            )

        r, g = str(response).strip(), str(ground_truth).strip()
        if not self.case_sensitive:
            r, g = r.lower(), g.lower()

        score = 1.0 if r == g else 0.0
        return MetricResult(
            score=score,
            metadata={
                "passed": score == 1.0,
                "case_sensitive": self.case_sensitive,
                "provider": "builtin",
                "metric_name": "exact_match",
            },
        )


class SemanticSim(BaseMetric):
    """Cosine similarity using sentence-transformers embeddings.

    Requires the sentence-transformers package:
        pip install sentence-transformers

    Score: 0.0–1.0 (cosine similarity, clamped to [0, 1]).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.8, **kwargs):
        super().__init__(name="semantic_sim", **kwargs)
        self.provider = "builtin"
        self.model_name = model_name
        self.threshold = threshold
        self._model = None
        self._util = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer, util  # type: ignore[import]

            self._model = SentenceTransformer(self.model_name)
            self._util = util
        except ImportError as exc:
            raise ImportError(
                "builtin:semantic_sim requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            ) from exc

    def evaluate(self, sample, **kwargs) -> MetricResult:
        response = getattr(sample, "llm_response", None)
        if response is None and hasattr(sample, "trace") and sample.trace:
            response = sample.trace.final_response

        ground_truth = getattr(sample, "ground_truth", None)
        if ground_truth is None:
            ground_truth = getattr(sample, "reference_outcome", None)

        if response is None or ground_truth is None:
            return MetricResult(
                score=None,
                metadata={"error": "response and ground_truth required", "provider": "builtin"},
            )

        try:
            self._load_model()
            emb1 = self._model.encode(str(response), convert_to_tensor=True)
            emb2 = self._model.encode(str(ground_truth), convert_to_tensor=True)
            score = float(self._util.cos_sim(emb1, emb2)[0][0])
            score = round(max(0.0, min(1.0, score)), 4)
            return MetricResult(
                score=score,
                metadata={
                    "passed": score >= self.threshold,
                    "threshold": self.threshold,
                    "model": self.model_name,
                    "provider": "builtin",
                    "metric_name": "semantic_sim",
                },
            )
        except Exception as e:
            logger.error("SemanticSim failed: %s", e, exc_info=True)
            return MetricResult(score=None, metadata={"error": str(e), "provider": "builtin"})


MetricRegistry.register("builtin", "exact_match", ExactMatch)
MetricRegistry.register("builtin", "semantic_sim", SemanticSim)
