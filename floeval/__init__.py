"""Floeval — Multi-backend evaluation framework for LLM, RAG, prompt, agent, and workflow evaluation."""

from floeval.api import DatasetLoader, Evaluation, MetricRegistry
from floeval.api.agent_evaluation import AgentEvaluation, AgentEvaluationResult
from floeval.api.workflow_evaluation import WorkflowEvaluation, WorkflowEvaluationResult
from floeval.config.schemas.io.agent_dataset import (
    AgentDataset,
    AgentSample,
    AgentTrace,
    PartialAgentSample,
    WorkflowExecution,
)
from floeval.config.schemas.io.dataset import Dataset

__all__ = [
    # LLM / RAG / Prompt evaluation
    "Evaluation",
    "Dataset",
    "DatasetLoader",
    "MetricRegistry",
    # Agent evaluation
    "AgentEvaluation",
    "AgentEvaluationResult",
    # Workflow evaluation
    "WorkflowEvaluation",
    "WorkflowEvaluationResult",
    # Domain models
    "AgentDataset",
    "AgentSample",
    "AgentTrace",
    "PartialAgentSample",
    "WorkflowExecution",
]
