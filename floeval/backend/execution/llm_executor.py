"""
LLM executor for future LLM-based execution
"""

from floeval.backend.execution.executor import BaseExecutor


class LLMExecutor(BaseExecutor):
    """
    Executor for LLM-based evaluation execution.
    """
    
    def __init__(self):
        super().__init__()
    
    def execute(self, *args, **kwargs):
        """
        Execute LLM-based evaluation.
        """
        pass

