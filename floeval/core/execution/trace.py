"""
ExecutionTrace model for tracking execution
"""

from typing import Dict, Any, Optional
from datetime import datetime


class ExecutionTrace:
    """
    Model for tracking execution traces.
    """
    
    def __init__(self):
        self.timestamp: Optional[datetime] = None
        self.metadata: Dict[str, Any] = {}

