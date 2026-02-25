"""MetricContext provides access to full sample data for custom metrics.

Used when metrics need more than simple parameters.
"""

from dataclasses import dataclass
from typing import Any

from floeval.api.dataset import Sample
from floeval.config.schemas.io.llm import LLMProviderConfig


@dataclass
class MetricContext:
    """Context object for custom metric functions.

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
    llm_config: LLMProviderConfig | None = None

    @property
    def inputs(self) -> dict[str, Any]:
        """Shortcut to sample fields as dict."""
        return {
            "user_input": self.sample.user_input,
            "llm_response": self.sample.llm_response,
            "contexts": self.sample.contexts or [],
        }

    @property
    def ground_truth(self) -> str | None:
        """Shortcut to sample.ground_truth."""
        return self.sample.ground_truth

    def get_input(self, key: str, default: Any = None) -> Any:
        """
        Get input value with default.

        Args:
            key: Input key to retrieve (user_input, llm_response, contexts)
            default: Default value if key not found

        Returns:
            Value from sample or default
        """
        if key == "user_input":
            return getattr(self.sample, "user_input", default)
        elif key == "llm_response":
            return getattr(self.sample, "llm_response", default)
        elif key == "contexts":
            return getattr(self.sample, "contexts", default) or []
        return default

    def get_ground_truth(self, key: str, default: Any = None) -> Any:
        """
        Get ground truth value with default.

        Args:
            key: Ground truth key (ignored, ground_truth is a string)
            default: Default value if ground_truth not found

        Returns:
            Value from sample.ground_truth or default
        """
        return self.sample.ground_truth if self.sample.ground_truth else default
