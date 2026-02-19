"""
Metric providers module - All metrics organized by provider.

This module imports all provider modules to trigger metric registration.
"""

# Import providers to trigger registration
# Each provider's __init__.py registers its metrics with the global registry
import floeval.metric_providers.builtin
import floeval.metric_providers.deepeval
import floeval.metric_providers.ragas

__all__ = ["builtin", "ragas", "deepeval"]
