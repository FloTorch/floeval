"""
ExecutionTrace model for tracking execution
"""

from datetime import datetime


class ExecutionTrace:
    """
    Model for tracking execution traces.
    """

    def __init__(self):
        self.timestamp: datetime | None = None
        self.metadata: dict[str, object] = {}
