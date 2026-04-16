"""Filter metric constructor kwargs so only accepted parameters are passed.

Used when injecting optional keys (e.g. ``extra_headers`` for gateway tracing):
add them to the merged dict, then drop any key the metric's ``__init__`` does
not accept (unless the signature includes ``**kwargs``).
"""

from __future__ import annotations

import inspect
from typing import Any


def filter_kwargs_for_metric_factory(merged: dict[str, Any], metric_factory: Any) -> dict[str, Any]:
    """Return a copy of ``merged`` containing only keys accepted by the metric constructor."""
    try:
        if callable(metric_factory) and not isinstance(metric_factory, type):
            sig = inspect.signature(metric_factory)
        else:
            sig = inspect.signature(metric_factory.__init__)
        accepted = set(sig.parameters) - {"self"}
        has_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if has_var_kw:
            return dict(merged)
        return {k: v for k, v in merged.items() if k in accepted}
    except (TypeError, ValueError, AttributeError):
        return dict(merged)
