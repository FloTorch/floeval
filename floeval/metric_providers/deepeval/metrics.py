"""
DeepEval metric implementations
"""

from abc import abstractmethod
from dataclasses import asdict
from api.metrics.base import BaseMetric, MetricResult
from .adapter import DeepEvalAdapter
from deepeval.metrics import FaithfulnessMetric
from deepeval import evaluate

__VALID_DEEPEVAL_METRICS__ = {
    "faithfulness": FaithfulnessMetric,
}

class DeepEvalMetric(BaseMetric):
    """
    Base class for DeepEval metrics.
    """
    
    def __init__(self, name: str, *args, **kwargs):
        super().__init__(name=name)
        self.adapter = DeepEvalAdapter(name, config=kwargs['config'], test_cases=kwargs['test_cases'])
        
    @property
    def config(self):
        return self.adapter.config
    
    @property
    def test_cases(self):
        return self.adapter.test_cases
    
    @abstractmethod
    def compute(self) -> MetricResult:
        """
        Compute DeepEval metric score.
        """
        
        
class FaithfulnessDeepEvalMetric(DeepEvalMetric):
    """
    Faithfulness metric implementation using DeepEval.
    """
    
    def compute(self) -> MetricResult:
        """
        Compute faithfulness metric score using DeepEval.
        
        Returns:
            MetricResult: The result of the faithfulness metric computation
        """
        metric_instance = FaithfulnessMetric(**asdict(self.config))
        result = evaluate(metrics=[metric_instance], test_cases=self.test_cases).test_results[0]
        
        # TODO: We need to decide how to validate multiple metrics_data entries
        return MetricResult(score=result.metrics_data[0].score)

