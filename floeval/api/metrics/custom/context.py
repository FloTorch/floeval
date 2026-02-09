"""
MetricContext provides access to full sample data for custom metrics.

Used when metrics need more than simple parameters.
"""

from dataclasses import dataclass
from typing import Any

from floeval.api.dataset import Sample
from floeval.config import GatewayConfig


@dataclass
class MetricContext:
    """
    Context object for custom metric functions.
    
    Provides simplified interface to complex Sample structure.
    Encapsulates sample access logic for cleaner metric code.
    
    Usage:
        @custom_metric
        def my_metric(response: str, context: MetricContext) -> float:
            question = context.get_input("question")
            expected = context.get_ground_truth("expected_answer")
            return compute_score(response, question, expected)
    """
    
    sample: Sample
    gateway_config: GatewayConfig | None = None
    
    @property
    def inputs(self) -> dict[str, Any]:
        """Shortcut to sample.inputs."""
        return self.sample.inputs
    
    @property
    def ground_truth(self) -> dict[str, Any]:
        """Shortcut to sample.ground_truth."""
        return self.sample.ground_truth or {}
    
    def get_input(self, key: str, default: Any = None) -> Any:
        """
        Get input value with default.
        
        Args:
            key: Input key to retrieve from sample.inputs
            default: Default value if key not found
        
        Returns:
            Value from sample.inputs or default
        """
        return self.sample.inputs.get(key, default)
    
    def get_ground_truth(self, key: str, default: Any = None) -> Any:
        """
        Get ground truth value with default.
        
        Args:
            key: Ground truth key to retrieve from sample.ground_truth
            default: Default value if key not found
        
        Returns:
            Value from sample.ground_truth or default
        """
        return (self.sample.ground_truth or {}).get(key, default)
