"""Conversational (multi-turn) DeepEval metrics."""

from typing import Any, ClassVar, Literal

from deepeval.evaluate import evaluate
from deepeval.metrics import ConversationalGEval

from floeval.api.metrics.base import MetricResult
from floeval.config.schemas.io.agent_dataset import AgentSample
from floeval.config.schemas.io.conversational_dataset import ConversationalSample
from floeval.metric_providers.deepeval.conversational_test_case import (
    conversational_row_to_deepeval,
)
from floeval.metric_providers.deepeval.metrics import DeepEvalMetric


class ConversationalGEvalDeepEvalMetric(DeepEvalMetric):
    """LLM-as-judge over a full transcript using DeepEval ``ConversationalGEval``."""

    execute_via: ClassVar[str] = "deepeval"
    deepeval_test_case_kind: ClassVar[Literal["conversational"]] = "conversational"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, name="conversational_g_eval", **kwargs)

    def _extract_metric_params(
        self, params: dict[str, Any], kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        metric_params = super()._extract_metric_params(params, kwargs)
        for key in (
            "criteria",
            "evaluation_params",
            "evaluation_steps",
            "evaluation_name",
            "rubric",
        ):
            if key in kwargs and key not in metric_params:
                metric_params[key] = kwargs[key]
        return metric_params

    def create_deepeval_metric_instance(self) -> ConversationalGEval:
        """Return a DeepEval metric instance for batched ``evaluate()`` runs."""
        if self._llm_adapter is None:
            raise ValueError(
                "llm_config is required for conversational_g_eval. "
                "Pass llm_config into Evaluation()."
            )
        criteria = self._metric_params.get("criteria")
        if not criteria:
            raise ValueError(
                "conversational_g_eval requires criteria (e.g. params.criteria in YAML)."
            )
        eval_name: str = str(self._metric_params.get("evaluation_name", self.name))
        forward = {
            k: v
            for k, v in self._metric_params.items()
            if k
            in (
                "evaluation_params",
                "evaluation_steps",
                "rubric",
                "strict_mode",
                "verbose_mode",
                "evaluation_template",
                "async_mode",
                "threshold",
            )
        }
        return ConversationalGEval(
            name=eval_name,
            model=self._llm_adapter,
            criteria=criteria,
            **forward,
        )

    def evaluate(
        self, sample: ConversationalSample | AgentSample, **kwargs: Any
    ) -> MetricResult:
        """Score a single sample via DeepEval (standalone execution path)."""
        inst = self.create_deepeval_metric_instance()
        tc = conversational_row_to_deepeval(sample)
        agg = evaluate(metrics=[inst], test_cases=[tc])
        if not agg.test_results:
            return MetricResult(
                score=None,
                metadata={
                    "passed": False,
                    "provider": "deepeval",
                    "metric_name": self.name,
                    "error": "DeepEval returned no test_results.",
                },
            )
        row = agg.test_results[0]
        return self._extract_metric_result(row, self.name)
