"""RAGAS custom metric adapter."""

import logging
from typing import Type

from ragas import EvaluationDataset, SingleTurnSample
from ragas.metrics.base import MetricType, MetricWithLLM, SingleTurnMetric

from floeval.api.dataset import Dataset, Sample
from floeval.api.metrics.base import BaseMetric, MetricResult
from floeval.config.schemas.io.llm import LLMProviderConfig
from floeval.metric_providers.ragas.adapter import RAGASAdapter

logger = logging.getLogger(__name__)


class RAGASCustomMetricAdapter:
    """Transforms Floeval custom metrics to RAGAS native metrics."""

    def __init__(
        self,
        llm_config: LLMProviderConfig | None = None,
        ragas_adapter: RAGASAdapter | None = None,
    ):
        """Use ragas_adapter if provided, else create from llm_config."""
        self.llm_config = llm_config
        self._ragas_adapter = (
            ragas_adapter if ragas_adapter is not None else RAGASAdapter(config=llm_config)
        )

    @property
    def llm(self):
        """Get RAGAS LLM wrapper."""
        return self._ragas_adapter.llm

    @property
    def embeddings(self):
        """Get RAGAS embeddings wrapper."""
        return self._ragas_adapter.embeddings

    def transform_metric(self, floeval_metric: BaseMetric) -> Type[MetricWithLLM]:
        """Return RAGAS metric class for this Floeval metric."""
        if hasattr(floeval_metric, "user_func") or hasattr(floeval_metric, "_user_func"):
            return self._transform_function_metric(floeval_metric)

        if hasattr(floeval_metric, "description") and hasattr(floeval_metric, "llm_helper"):
            return self._transform_criteria_metric(floeval_metric)

        raise ValueError(
            f"Unsupported metric type for RAGAS transformation: {type(floeval_metric)}. "
            f"Expected FunctionBasedMetric or CriteriaBasedMetric."
        )

    def _transform_function_metric(self, metric: BaseMetric) -> Type[MetricWithLLM]:
        """Transform function-based metric to RAGAS metric class."""
        metric_name = metric.name
        floeval_metric_instance = metric
        threshold = getattr(metric, "threshold", 0.5)

        class GeneratedRAGASMetric(MetricWithLLM, SingleTurnMetric):
            """Runtime-generated RAGAS metric from Floeval custom metric."""

            def __init__(self, llm=None, name=metric_name):
                _required_columns = {
                    MetricType.SINGLE_TURN: {"response", "user_input", "retrieved_contexts"}
                }
                super().__init__(_required_columns=_required_columns, llm=llm, name=name)
                self._floeval_metric = floeval_metric_instance
                self._threshold = threshold

            async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks=None) -> float:
                floeval_sample = self._transform_sample(sample)
                result = await self._floeval_metric.aevaluate(floeval_sample)
                if isinstance(result, MetricResult):
                    score = result.score if result.score is not None else 0.0
                elif isinstance(result, (int, float)):
                    score = float(result)
                else:
                    score = 0.0
                return max(0.0, min(1.0, score))

            def _transform_sample(self, ragas_sample: SingleTurnSample) -> Sample:
                """Transform RAGAS sample to Floeval sample.

                Mapping:
                - user_input → user_input
                - response → llm_response
                - retrieved_contexts → contexts
                - reference → ground_truth

                Args:
                    ragas_sample: RAGAS SingleTurnSample

                Returns:
                    Floeval Sample
                """
                return Sample(
                    user_input=ragas_sample.user_input or "",
                    llm_response=ragas_sample.response or "",
                    contexts=ragas_sample.retrieved_contexts or [],
                    ground_truth=ragas_sample.reference,
                )

        return GeneratedRAGASMetric

    def _transform_criteria_metric(self, metric: BaseMetric) -> Type[MetricWithLLM]:
        """Transform criteria-based metric to RAGAS metric class.

        For criteria-based metrics, we generate a RAGAS metric that uses the LLM
        to evaluate based on the criteria description. This wraps the criteria
        metric's LLM evaluation logic.

        Args:
            metric: Floeval CriteriaBasedMetric instance

        Returns:
            RAGAS metric class
        """
        metric_name = metric.name
        description = getattr(metric, "description", "")
        evaluation_steps = getattr(metric, "evaluation_steps", [])
        threshold = getattr(metric, "threshold", 0.5)
        floeval_metric_instance = metric

        class GeneratedRAGASCriteriaMetric(MetricWithLLM, SingleTurnMetric):
            """Runtime-generated RAGAS metric from Floeval criteria-based metric."""

            def __init__(self, llm=None, name=metric_name):
                _required_columns = {MetricType.SINGLE_TURN: {"response", "user_input"}}
                super().__init__(_required_columns=_required_columns, llm=llm, name=name)
                self._floeval_metric = floeval_metric_instance
                self._description = description
                self._evaluation_steps = evaluation_steps
                self._threshold = threshold

            async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks=None) -> float:
                """RAGAS evaluation method for criteria-based metric.

                Uses Floeval criteria metric's evaluation logic.

                Args:
                    sample: RAGAS SingleTurnSample
                    callbacks: RAGAS callbacks (unused)

                Returns:
                    float: Score between 0 and 1
                """
                # Transform sample
                floeval_sample = self._transform_sample(sample)

                # Execute Floeval criteria metric (async)
                result = await self._floeval_metric.aevaluate(floeval_sample)
                if isinstance(result, MetricResult):
                    score = result.score if result.score is not None else 0.0
                elif isinstance(result, (int, float)):
                    score = float(result)
                else:
                    score = 0.0
                return max(0.0, min(1.0, score))

            def _transform_sample(self, ragas_sample: SingleTurnSample) -> Sample:
                return Sample(
                    user_input=ragas_sample.user_input or "",
                    llm_response=ragas_sample.response or "",
                    contexts=ragas_sample.retrieved_contexts or [],
                    ground_truth=ragas_sample.reference,
                )

        return GeneratedRAGASCriteriaMetric

    def transform_sample(self, sample: Sample) -> SingleTurnSample:
        """Transform Floeval sample to RAGAS SingleTurnSample."""
        return self._ragas_adapter.transform_sample(sample)

    def transform_dataset(self, dataset: Dataset) -> EvaluationDataset:
        """Transform Floeval dataset to RAGAS EvaluationDataset."""
        ragas_samples = []
        for sample in dataset.samples:
            ragas_sample = self.transform_sample(sample)
            ragas_samples.append(ragas_sample)

        return EvaluationDataset(samples=ragas_samples)
