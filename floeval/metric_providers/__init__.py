"""
Metric providers module - All metrics organized by provider.

This module imports all provider modules to trigger metric registration.
"""

# Import providers to trigger registration
# Each provider's __init__.py registers its metrics with the global registry
from floeval.metric_providers import deepeval, ragas

__all__ = ["ragas", "deepeval"]
