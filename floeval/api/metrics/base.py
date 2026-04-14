"""
Base metric abstract class and result model
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Mapping


class MetricResult:
    """Result container for metric evaluation."""

    def __init__(self, score: float | None, metadata: Mapping[str, Any] | None = None):
        self.score = score
        self.metadata = dict(metadata) if metadata else {}


class BaseMetric(ABC):
    """Abstract base class for all FloEval metrics.

    Class-level attributes (override in subclasses to control routing):

    execution_scope: str
        "per_sample" (default): metric is called once per AgentSample / Sample.
        "per_workflow": metric is called once per WorkflowExecution.
                        Only valid inside WorkflowEvaluation — ignored by
                        Evaluation and AgentEvaluation.

    execute_via: str | None
        Existing attribute used by Evaluation for provider routing.
        "ragas"    → routed to ragas batch evaluate()
        "deepeval" → routed to deepeval batch evaluate()
        None       → standalone execution (default)
    """

    # Controls execution routing in WorkflowEvaluation
    execution_scope: str = "per_sample"  # "per_sample" | "per_workflow"

    # Controls how WorkflowEvaluation aggregates per-agent scores into one number:
    #   "mean" → average across agents (use for quality/rate metrics 0-1)
    #   "sum"  → total across agents  (use for counts/volume metrics)
    workflow_aggregate: str = "mean"

    def __init__(self, name: str, *args, **kwargs):
        self.name: str = name
        self.provider: str | None = None

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> MetricResult:
        """Evaluate metric synchronously."""
        pass

    async def aevaluate(self, *args, **kwargs) -> MetricResult:
        """Default: run evaluate() in executor. Override for true async."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.evaluate(*args, **kwargs))
