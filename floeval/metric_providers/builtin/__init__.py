"""Built-in metrics provider.

Importing this module registers all builtin metrics with the global MetricRegistry.
"""

from floeval.metric_providers.builtin.agent_metrics import (
    GoalAchievementMetric,       # deprecated — use task_completion
    ResponseCoherenceMetric,     # deprecated — use trajectory_faithfulness
    TaskCompletionMetric,
    TrajectoryFaithfulnessMetric,
)
from floeval.metric_providers.builtin.efficiency_metrics import (
    LatencyMetric,
    TokenUsageMetric,
    ToolCallCountMetric,
    TurnCountMetric,
)
from floeval.metric_providers.builtin.metrics import ExactMatch, SemanticSim
from floeval.metric_providers.builtin.tool_metrics import (
    ToolArgumentCorrectnessMetric,
    ToolCallSuccessRateMetric,
    ToolSelectionAccuracyMetric,
)
from floeval.metric_providers.builtin.workflow_metrics import (
    AgentHandoffQualityMetric,
    CrossAgentConsistencyMetric,
    WorkflowCompletionRateMetric,
)

__all__ = [
    # Agent metrics
    "TaskCompletionMetric",
    "TrajectoryFaithfulnessMetric",
    "GoalAchievementMetric",
    "ResponseCoherenceMetric",
    # Tool metrics
    "ToolCallSuccessRateMetric",
    "ToolSelectionAccuracyMetric",
    "ToolArgumentCorrectnessMetric",
    # Efficiency metrics
    "TurnCountMetric",
    "ToolCallCountMetric",
    "LatencyMetric",
    "TokenUsageMetric",
    # Workflow metrics
    "WorkflowCompletionRateMetric",
    "AgentHandoffQualityMetric",
    "CrossAgentConsistencyMetric",
    # Core metrics
    "ExactMatch",
    "SemanticSim",
]
