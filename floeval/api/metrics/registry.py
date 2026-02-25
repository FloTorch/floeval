"""Two-level metric registry: provider -> metric_name -> metric_class."""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any, Callable

from floeval.api.metrics.base import BaseMetric

MetricFactory = type[BaseMetric] | Callable[..., BaseMetric]


class MetricRegistry:
    """
    Two-level registry for metrics: provider -> metric_name -> metric_class_or_factory

    Supports both classes (e.g. RAGASAnswerRelevancy) and factory callables
    (e.g. for custom metrics). Thread-safe.
    """

    _instance: MetricRegistry | None = None
    _registry: defaultdict[str, dict[str, MetricFactory]] = defaultdict(dict)
    _lock: threading.RLock = threading.RLock()

    def __new__(cls) -> MetricRegistry:
        """Singleton pattern - return the same instance (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize instance (only runs once due to singleton)."""
        if not hasattr(self, "_initialized"):
            self._initialized = True

    @classmethod
    def register(
        cls,
        provider: str,
        metric_name: str,
        metric_class: MetricFactory,
        allow_override: bool = False,
    ) -> None:
        """
        Register a metric class or factory under a provider and metric name.

        Args:
            provider: The provider name (e.g., 'builtin', 'ragas', 'deepeval', 'custom')
            metric_name: The name of the metric
            metric_class: The metric class or factory callable (invoked with **params)
            allow_override: If True, allow overriding existing metrics (for custom metrics)

        Raises:
            ValueError: If metric exists and allow_override=False (except for custom provider)
        """
        with cls._lock:
            if metric_name in cls._registry[provider]:
                existing_class = cls._registry[provider][metric_name]
                if existing_class is metric_class:
                    return
                if provider == "custom":
                    import warnings

                    warnings.warn(
                        f"Metric '{metric_name}' already registered in 'custom' namespace. "
                        "Overriding with new definition. "
                        "To avoid this, use unique metric names.",
                        UserWarning,
                        stacklevel=2,
                    )
                elif not allow_override:
                    raise ValueError(
                        f"Metric '{provider}:{metric_name}' is already registered. "
                        "Cannot override provider metrics."
                    )
            cls._registry[provider][metric_name] = metric_class

    @classmethod
    def get_class(cls, provider: str, metric_name: str) -> MetricFactory | None:
        """Return the metric class or factory for provider and metric name, or None."""
        with cls._lock:
            if provider not in cls._registry:
                return None
            return cls._registry[provider].get(metric_name)

    def get(self, provider: str, metric_name: str, **params: Any) -> Any:
        return self.create(provider, metric_name, **params)

    @classmethod
    def create(cls, provider: str, metric_name: str, **params: Any) -> Any:
        """
        Create a metric instance by provider and metric name.

        Args:
            provider: provider name (e.g. "ragas")
            metric_name: metric id/name (e.g. "answer_relevancy")
            **params: forwarded to metric constructor

        Raises:
            KeyError if provider/metric not found.
        """
        factory = cls.get_class(provider, metric_name)
        if factory is None:
            available = cls.list_metrics(provider) if provider in cls._registry else []
            raise KeyError(f"Unknown metric: {provider}:{metric_name}. Available: {available}")
        return factory(**params)

    @classmethod
    def resolve_best(cls, metric_name: str, default_provider: str | None = None) -> str:
        """
        Resolve the best provider for a metric name.

        Logic:
        1) If only one provider has it -> return that provider
        2) If multiple providers have it -> return default_provider if it contains it
        3) Else -> raise ambiguity error
        """
        providers = [p for p, metrics in cls._registry.items() if metric_name in metrics]

        if not providers:
            raise KeyError(
                f"No provider has '{metric_name}'. Available metrics: {cls.list_all_metrics()}"
            )

        if len(providers) == 1:
            return providers[0]

        if default_provider and default_provider in providers:
            return default_provider

        raise ValueError(
            f"'{metric_name}' is ambiguous across providers: {providers}. "
            f"Use 'provider:metric_name' or set default_provider."
        )

    @classmethod
    def list_providers(cls):
        """
        List all registered providers.

        Returns:
            List of provider names
        """
        return list(cls._registry.keys())

    @classmethod
    def list_metrics(cls, provider: str):
        """
        List all metrics for a given provider.

        Args:
            provider: The provider name

        Returns:
            List of metric names for the provider
        """
        if provider not in cls._registry:
            return []
        return list(cls._registry[provider].keys())

    @classmethod
    def list_all_metrics(cls) -> list[str]:
        """List all metric names across all providers."""
        all_metrics: set[str] = set()
        for provider_metrics in cls._registry.values():
            all_metrics.update(provider_metrics.keys())
        return sorted(all_metrics)
