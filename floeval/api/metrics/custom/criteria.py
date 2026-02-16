"""Criteria-based custom metrics using LLM-as-judge."""

import json
import logging
import re
from typing import Any

from floeval.api.metrics.base import BaseMetric, MetricResult
from floeval.api.metrics.registry import MetricRegistry

from .llm_helper import SimpleLLMHelper

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
    Create criteria-based metric using LLM-as-judge.
    
    Args:
        name: Metric name
        description: Evaluation criteria (e.g., "Rate empathy on scale 0-1")
        threshold: Pass/fail threshold (0.0-1.0)
        evaluation_steps: Optional evaluation instructions
        llm_model: LLM model identifier (overridden by gateway_config if provided)
        execute_via: Execution provider (ragas/deepeval/None)
        **kwargs: Additional parameters (gateway_config injected by Evaluation)
    """
    
    class CriteriaBasedMetric(BaseMetric):
        """Criteria-based metric using LLM-as-judge."""
        
        def __init__(self, **init_kwargs):
            super().__init__(name=name, **init_kwargs)
            self.description = description
            self.threshold = threshold
            self.evaluation_steps = evaluation_steps or []
            self.gateway_config = init_kwargs.get('gateway_config')
            self._default_llm_model = llm_model
            self.llm_model = (
                self.gateway_config.llm_model if self.gateway_config and self.gateway_config.llm_model
                else llm_model
            )
            self.execute_via = execute_via
            self.provider = "custom"
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
        
        def evaluate(self, sample: Any, **kwargs: Any) -> MetricResult:
            """Evaluate using LLM-as-judge synchronously."""
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
        
        async def aevaluate(self, sample: Any, **kwargs: Any) -> MetricResult:
            """Evaluate using LLM-as-judge asynchronously."""
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
            return getattr(sample, "llm_response", "") or ""
        
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
            return getattr(sample, "user_input", "") or ""
        
        def _build_prompt(self, question: str, response: str) -> str:
            """Build evaluation prompt for LLM."""
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
            """Parse LLM response to extract score and reasoning."""
            try:
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    score = float(data.get("score", 0))
                    reason = data.get("reasoning", "")
                    score = max(0.0, min(1.0, score))
                    return score, reason
                
                numbers = re.findall(r'\d+\.?\d*', content)
                if numbers:
                    score = float(numbers[0])
                    if score > 1.0:
                        score = score / 5.0
                    return max(0.0, min(1.0, score)), content
                
                return None, f"Could not parse response: {content}"
            
            except Exception as e:
                return None, f"Parse error: {str(e)}"
    
    CriteriaBasedMetric.__name__ = f"{name}_criteria_metric"
    CriteriaBasedMetric.__qualname__ = f"{name}_criteria_metric"
    
    MetricRegistry().register("custom", name, CriteriaBasedMetric, allow_override=True)
    return CriteriaBasedMetric(**kwargs)
