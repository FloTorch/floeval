"""
Metric registry for managing and resolving metrics (two-level registry)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Type, Any


class MetricRegistry:
    """
    Two-level registry for metrics: provider -> metric_name -> metric_class
    
    This class uses a singleton pattern to ensure all registrations
    are stored in a shared registry instance.
    """
    
    _instance: Optional["MetricRegistry"] = None
    _registry: Dict[str, Dict[str, type]] = {}
    
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
    
    def register(self, provider: str, metric_name: str, metric_class: type, allow_override: bool = False):
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
        if provider not in MetricRegistry._registry:
            MetricRegistry._registry[provider] = {}
        
        # Check for duplicates
        if metric_name in MetricRegistry._registry[provider]:
            existing_class = MetricRegistry._registry[provider][metric_name]
            
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
                # Allow override
            elif allow_override:
                # Explicitly allowed override
                pass
            else:
                # For provider metrics, raise error
                raise ValueError(
                    f"Metric '{provider}:{metric_name}' is already registered. "
                    f"Cannot override provider metrics."
                )
        
        MetricRegistry._registry[provider][metric_name] = metric_class
    
    def get_class(self, provider: str, metric_name: str) -> Optional[Type[Any]]:
        """
        Get a metric class by provider and metric name.
        
        Args:
            provider: The provider name
            metric_name: The metric name
            
        Returns:
            The metric class if found, None otherwise
        """
        if provider not in MetricRegistry._registry:
            return None
        return MetricRegistry._registry[provider].get(metric_name)

    
    def get(self, provider: str, metric_name: str, **params: Any) -> Any:
        return self.create(provider, metric_name, **params)

    def create(self, provider: str, metric_name: str, **params: Any) -> Any:
        """
        Create a metric instance by provider and metric name.

        Args:
            provider: provider name (e.g. "ragas")
            metric_name: metric id/name (e.g. "answer_relevancy")
            **params: forwarded to metric constructor

        Raises:
            KeyError if provider/metric not found.
        """
        metric_cls = self.get_class(provider, metric_name)
        if metric_cls is None:
            available = self.list_metrics(provider) if provider in MetricRegistry._registry else []
            raise KeyError(f"Unknown metric: {provider}:{metric_name}. Available: {available}")
        return metric_cls(**params)

    def resolve_best(self, metric_name: str, default_provider: Optional[str] = None) -> str:
        """
        Resolve the best provider for a metric name.

        Logic:
        1) If only one provider has it -> return that provider
        2) If multiple providers have it -> return default_provider if it contains it
        3) Else -> raise ambiguity error
        """
        providers = [p for p, metrics in MetricRegistry._registry.items() if metric_name in metrics]

        if not providers:
            raise KeyError(
                f"No provider has '{metric_name}'. Available metrics: {self.list_all_metrics()}"
            )

        if len(providers) == 1:
            return providers[0]

        if default_provider and default_provider in providers:
            return default_provider

        raise ValueError(
            f"'{metric_name}' is ambiguous across providers: {providers}. "
            f"Use 'provider:metric_name' or set default_provider."
        )
    
    def list_providers(self):
        """
        List all registered providers.
        
        Returns:
            List of provider names
        """
        return list(MetricRegistry._registry.keys())
    
    def list_metrics(self, provider: str):
        """
        List all metrics for a given provider.
        
        Args:
            provider: The provider name
            
        Returns:
            List of metric names for the provider
        """
        if provider not in MetricRegistry._registry:
            return []
        return list(MetricRegistry._registry[provider].keys())

    def list_all_metrics(self) -> List[str]:
        """List all metric names across all providers."""
        all_metrics = set()
        for provider_metrics in MetricRegistry._registry.values():
            all_metrics.update(provider_metrics.keys())
        return sorted(all_metrics)

