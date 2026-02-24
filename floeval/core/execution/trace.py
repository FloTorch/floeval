"""
ExecutionTrace model for tracking execution
"""

from datetime import datetime
from typing import Any, Dict, Optional


class ExecutionTrace:
    """
    Model for tracking execution traces.
    """

    def __init__(self):
        self.timestamp: Optional[datetime] = None
        self.metadata: Dict[str, Any] = {}
