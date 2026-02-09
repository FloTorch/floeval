"""
Criteria-based custom metrics using LLM-as-judge.

Implements natural language evaluation criteria.
"""

import json
import logging
import re
from typing import Any

from floeval.api.metrics.base import BaseMetric, MetricResult
from floeval.api.metrics.registry import MetricRegistry
from floeval.config import GatewayConfig

from .llm_helper import SimpleLLMHelper

# Module-level logger
logger = logging.getLogger(__name__)

# Lazy imports (optional backend):
# - openai: Optional backend; avoid import if LLM features not used


def criteria(
    name: str,
    description: str,
    threshold: float = 0.5,
    evaluation_steps: list[str] | None = None,
    llm_model: str = "gpt-4",
    execute_via: str | None = None,
    **kwargs
) -> BaseMetric:
    """
    criteria-based custom metric using LLM-as-judge.
    
    Design Pattern: Builder Pattern
    - Fluent interface for metric creation
    - Sensible defaults
    - Optional configuration
    
    Args:
        name: Unique metric name (used for registration and identification).
        description: Natural language evaluation criteria describing what to evaluate
                   (e.g., "Rate empathy on scale 0-1", "Check if response is professional").
        threshold: Pass/fail threshold (0.0-1.0). Score >= threshold passes.
        evaluation_steps: Optional list of step-by-step evaluation instructions.
                         Helps guide LLM evaluation process.
        llm_model: LLM model identifier (e.g., "gpt-4", "flotorch/openai-gpt-4").
                  Optional; overridden by gateway_config.llm_model if provided.
        execute_via: Provider for execution ("ragas"/"deepeval"/None for standalone).
                    Currently unused but reserved for future provider-specific execution.
        **kwargs: Additional parameters passed to metric instance.
                 gateway_config is injected here by Evaluation._resolve_metrics().
    
    Returns:
        CriteriaBasedMetric: Metric instance (already registered in MetricRegistry).
    
    Raises:
        ValueError: If gateway_config is not provided when metric is used.
    
    Example:
        empathy = criteria(
            name="empathy",
            description="Rate empathy on scale 0-1",
            threshold=0.7,
            evaluation_steps=[
                "Identify empathetic phrases",
                "Assess emotional acknowledgment"
            ]
        )
    """
    
    # Create metric class
    class CriteriaBasedMetric(BaseMetric):
        """Generated criteria-based metric using LLM-as-judge."""
        
        def __init__(self, **init_kwargs):
            super().__init__(name=name, **init_kwargs)
            self.description = description
            self.threshold = threshold
            self.evaluation_steps = evaluation_steps or []
            # Get gateway_config from init_kwargs (injected by Evaluation)
            self.gateway_config = init_kwargs.get('gateway_config')
            # Store the default llm_model parameter for fallback
            self._default_llm_model = llm_model
            # Use model from gateway_config if available, otherwise use llm_model parameter
            self.llm_model = (
                self.gateway_config.llm_model if self.gateway_config and self.gateway_config.llm_model
                else llm_model
            )
            self.execute_via = execute_via
            self.provider = "custom"
            
            # Use SimpleLLMHelper for both sync (compute) and async (acompute) evaluation
            self.llm_helper = SimpleLLMHelper(
                gateway_config=self.gateway_config,
                llm_model=llm_model,
            )
        
        def _llm_prompt(self, question: str, response: str) -> str:
            """Build full prompt with system instruction for LLM-as-judge."""
            prompt = self._build_prompt(question, response)
            return (
                "You are an expert evaluator. Provide scores as requested.\n\n"
                + prompt
            )
        
        def compute(self, sample: Any, **kwargs: Any) -> MetricResult:
            """
            Synchronous evaluation using LLM-as-judge.
            
            Uses SimpleLLMHelper.generate() which runs async in an isolated thread
            (production-safe). Both compute() and acompute() work.
            """
            if not self.gateway_config:
                error_msg = (
                    "gateway_config is required for criteria-based metrics. "
                    "Pass gateway_config to Evaluation when creating the evaluation instance."
                )
                logger.error(f"Metric {self.name}: {error_msg}")
                return MetricResult(
                    score=None,
                    metadata={
                        "error": error_msg,
                        "passed": False,
                        "threshold": self.threshold,
                    }
                )
            response = self._extract_response(sample)
            question = self._extract_question(sample)
            full_prompt = self._llm_prompt(question, response)
            try:
                logger.debug(f"Calling LLM for criteria metric {self.name}, model={self.llm_model}")
                content = self.llm_helper.generate(full_prompt, temperature=0.0)
                logger.debug(f"LLM response received for {self.name}, length={len(content)}")
                score, reason = self._parse_llm_response(content)
                return MetricResult(
                    score=score,
                    metadata={
                        "passed": score >= self.threshold if score is not None else False,
                        "threshold": self.threshold,
                        "reason": reason,
                        "llm_model": self.llm_model,
                        "criteria": self.description,
                    },
                )
            except Exception as e:
                logger.error(f"Criteria metric {self.name} failed: {e}", exc_info=True)
                return MetricResult(
                    score=None,
                    metadata={
                        "error": str(e),
                        "passed": False,
                        "threshold": self.threshold,
                    },
                )
        
        async def acompute(self, sample: Any, **kwargs: Any) -> MetricResult:
            """
            Async evaluation using LLM-as-judge.
            
            Uses SimpleLLMHelper.agenerate() directly (for Evaluation.arun() or async usage).
            """
            if not self.gateway_config:
                error_msg = (
                    "gateway_config is required for criteria-based metrics. "
                    "Pass gateway_config to Evaluation when creating the evaluation instance."
                )
                logger.error(f"Metric {self.name}: {error_msg}")
                return MetricResult(
                    score=None,
                    metadata={
                        "error": error_msg,
                        "passed": False,
                        "threshold": self.threshold,
                    }
                )
            response = self._extract_response(sample)
            question = self._extract_question(sample)
            full_prompt = self._llm_prompt(question, response)
            try:
                logger.debug(f"Calling LLM for criteria metric {self.name}, model={self.llm_model}")
                content = await self.llm_helper.agenerate(full_prompt, temperature=0.0)
                logger.debug(f"LLM response received for {self.name}, length={len(content)}")
                score, reason = self._parse_llm_response(content)
                return MetricResult(
                    score=score,
                    metadata={
                        "passed": score >= self.threshold if score is not None else False,
                        "threshold": self.threshold,
                        "reason": reason,
                        "llm_model": self.llm_model,
                        "criteria": self.description,
                    },
                )
            except Exception as e:
                logger.error(f"Criteria metric {self.name} failed: {e}", exc_info=True)
                return MetricResult(
                    score=None,
                    metadata={
                        "error": str(e),
                        "passed": False,
                        "threshold": self.threshold,
                    },
                )
        
        def _extract_response(self, sample: Any) -> str:
            """
            Extract response text from sample inputs.
            
            Tries multiple common field names: answer, response, actual_output.
            Returns empty string if none found.
            
            Args:
                sample: Sample object with inputs dict.
            
            Returns:
                str: Response text or empty string.
            """
            return (
                sample.inputs.get("answer") or 
                sample.inputs.get("response") or
                sample.inputs.get("actual_output") or
                ""
            )
        
        def _extract_question(self, sample: Any) -> str:
            """
            Extract question/input text from sample inputs.
            
            Tries multiple common field names: question, input, query.
            Returns empty string if none found.
            
            Args:
                sample: Sample object with inputs dict.
            
            Returns:
                str: Question text or empty string.
            """
            return (
                sample.inputs.get("question") or
                sample.inputs.get("input") or
                sample.inputs.get("query") or
                ""
            )
        
        def _build_prompt(self, question: str, response: str) -> str:
            """
            Build evaluation prompt for LLM-as-judge.
            
            Structure:
            1. Evaluation criteria (from description)
            2. Evaluation steps (if provided)
            3. Question (if available)
            4. Response to evaluate
            5. Output format instructions (JSON with score and reasoning)
            
            Args:
                question: Question/input text (may be empty).
                response: Response text to evaluate.
            
            Returns:
                str: Complete prompt string for LLM.
            """
            prompt = f"Evaluation Criteria:\n{self.description}\n\n"
            
            if self.evaluation_steps:
                prompt += "Evaluation Steps:\n"
                for i, step in enumerate(self.evaluation_steps, 1):
                    prompt += f"{i}. {step}\n"
                prompt += "\n"
            
            if question:
                prompt += f"Question: {question}\n"
            
            prompt += f"Response to evaluate: {response}\n\n"
            prompt += (
                "Provide your evaluation as JSON:\n"
                "{\n"
                '    "score": <number between 0 and 1>,\n'
                '    "reasoning": "<explanation>"\n'
                "}"
            )
            
            return prompt
        
        def _parse_llm_response(self, content: str) -> tuple[float | None, str]:
            """
            Parse LLM response to extract score and reasoning.
            
            Handles multiple response formats:
            - JSON format (preferred): {"score": 0.8, "reasoning": "..."}
            - Plain number: Extracts first numeric value
            - Scale normalization: If score > 1.0, assumes 1-5 scale and normalizes
            
            Args:
                content: Raw LLM response text.
            
            Returns:
                tuple[float | None, str]: (score normalized to 0-1, reasoning text).
                                         score is None if parsing fails.
            """
            try:
                # Try to extract JSON
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    score = float(data.get("score", 0))
                    reason = data.get("reasoning", "")
                    # Normalize to 0-1 range
                    score = max(0.0, min(1.0, score))
                    return score, reason
                
                # Fallback: extract number
                numbers = re.findall(r'\d+\.?\d*', content)
                if numbers:
                    score = float(numbers[0])
                    # If score looks like 1-5 scale, normalize
                    if score > 1.0:
                        score = score / 5.0
                    return max(0.0, min(1.0, score)), content
                
                return None, f"Could not parse response: {content}"
            
            except Exception as e:
                return None, f"Parse error: {str(e)}"
    
    # Set class metadata
    CriteriaBasedMetric.__name__ = f"{name}_criteria_metric"
    CriteriaBasedMetric.__qualname__ = f"{name}_criteria_metric"
    
    # Register in custom provider
    MetricRegistry().register("custom", name, CriteriaBasedMetric, allow_override=True)
    
    # Return instance (gateway_config comes from **kwargs via Evaluation)
    return CriteriaBasedMetric(**kwargs)
