"""
Base executor abstract class
"""

from abc import ABC, abstractmethod


class BaseExecutor(ABC):
    """
    Abstract base class for execution engines.
    """
    
    def __init__(self):
        pass
    
    @abstractmethod
    def execute(self, *args, **kwargs):
        """
        Execute the evaluation task.
        """
        pass

