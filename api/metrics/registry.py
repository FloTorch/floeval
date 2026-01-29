"""
Metric registry for managing and resolving metrics (two-level registry)
"""

from typing import Dict, Optional


class MetricRegistry:
    """
    Two-level registry for metrics: provider -> metric_name -> metric_class
    """
    
    def __init__(self):
        self._registry: Dict[str, Dict[str, type]] = {}
    
    def register(self, provider: str, metric_name: str, metric_class: type):
        """
        Register a metric class under a provider and metric name.
        
        Args:
            provider: The provider name (e.g., 'builtin', 'ragas', 'deepeval')
            metric_name: The name of the metric
            metric_class: The metric class to register
        """
        if provider not in self._registry:
            self._registry[provider] = {}
        self._registry[provider][metric_name] = metric_class
    
    def get(self, provider: str, metric_name: str) -> Optional[type]:
        """
        Get a metric class by provider and metric name.
        
        Args:
            provider: The provider name
            metric_name: The metric name
            
        Returns:
            The metric class if found, None otherwise
        """
        if provider not in self._registry:
            return None
        return self._registry[provider].get(metric_name)
    
    def list_providers(self):
        """
        List all registered providers.
        
        Returns:
            List of provider names
        """
        return list(self._registry.keys())
    
    def list_metrics(self, provider: str):
        """
        List all metrics for a given provider.
        
        Args:
            provider: The provider name
            
        Returns:
            List of metric names for the provider
        """
        if provider not in self._registry:
            return []
        return list(self._registry[provider].keys())

