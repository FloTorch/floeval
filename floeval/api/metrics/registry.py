"""Two-level metric registry: provider -> metric_name -> metric_class."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from floeval.api.metrics.base import BaseMetric


class MetricRegistry:
    """
    Two-level registry for metrics: provider -> metric_name -> metric_class
    
    This class uses a singleton pattern to ensure all registrations
    are stored in a shared registry instance.
    """
    
    _instance: MetricRegistry | None = None
    _registry: defaultdict[str, dict[str, type[BaseMetric]]] = defaultdict(dict)

    def __new__(cls):
        """Singleton pattern - return the same instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize instance (only runs once due to singleton)."""
        # Use class-level _registry to ensure all instances share the same data
        if not hasattr(self, "_initialized"):
            self._initialized = True

    @classmethod
    def register(cls, provider: str, metric_name: str, metric_class: type, allow_override: bool = False):
        """
        Register a metric class under a provider and metric name.
        
        Args:
            provider: The provider name (e.g., 'builtin', 'ragas', 'deepeval', 'custom')
            metric_name: The name of the metric
            metric_class: The metric class to register
            allow_override: If True, allow overriding existing metrics (for custom metrics)
        
        Raises:
            ValueError: If metric exists and allow_override=False (except for custom provider)
        """
        # Check for duplicates
        if metric_name in cls._registry[provider]:
            existing_class = cls._registry[provider][metric_name]
            
            # If same class, ignore (re-import scenario)
            if existing_class is metric_class:
                return
            
            # Different class - handle based on provider
            if provider == "custom":
                # Allow override for custom metrics with warning
                import warnings
                warnings.warn(
                    f"Metric '{metric_name}' already registered in 'custom' namespace. "
                    f"Overriding with new definition. "
                    f"To avoid this, use unique metric names.",
                    UserWarning
                )
            elif allow_override:
                pass
            else:
                raise ValueError(
                    f"Metric '{provider}:{metric_name}' is already registered. "
                    f"Cannot override provider metrics."
                )
        
        cls._registry[provider][metric_name] = metric_class

    @classmethod
    def get_class(cls, provider: str, metric_name: str) -> type[BaseMetric] | None:
        """Return the metric class for provider and metric name, or None if not registered."""
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
        metric_cls = cls.get_class(provider, metric_name)
        if metric_cls is None:
            available = cls.list_metrics(provider) if provider in cls._registry else []
            raise KeyError(f"Unknown metric: {provider}:{metric_name}. Available: {available}")
        return metric_cls(**params)

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
        all_metrics = set()
        for provider_metrics in cls._registry.values():
            all_metrics.update(provider_metrics.keys())
        return sorted(all_metrics)

