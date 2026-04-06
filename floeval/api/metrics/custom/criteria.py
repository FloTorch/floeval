"""Criteria-based custom metrics using LLM-as-judge."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from floeval.api.metrics.base import BaseMetric, MetricResult
from floeval.api.metrics.registry import MetricRegistry
from floeval.config.schemas.io.llm import OpenAIProviderConfig

from .llm_helper import SimpleLLMHelper

logger = logging.getLogger(__name__)


class CriteriaBasedMetric(BaseMetric):
    """LLM-as-judge metric driven by a natural language criteria description.

    Sync path (evaluate): calls SimpleLLMHelper.generate() — blocking.
    Async path (aevaluate): calls SimpleLLMHelper.agenerate() — native coroutine.
    """

    def __init__(
        self,
        metric_name: str,
        description: str,
        threshold: float = 0.5,
        evaluation_steps: list[str] | None = None,
        execute_via: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=metric_name, **kwargs)
        self.description = description
        self.threshold = threshold
        self.evaluation_steps: list[str] = evaluation_steps or []
        self.execute_via = execute_via
        self.provider = "custom"
        self._llm_config: OpenAIProviderConfig | None = kwargs.get("llm_config")
        self._extra_headers: dict[str, str] = dict(kwargs.get("extra_headers") or {})
        self._llm_helper: SimpleLLMHelper | None = None

    @property
    def llm_config(self) -> OpenAIProviderConfig | None:
        return self._llm_config

    @llm_config.setter
    def llm_config(self, value: OpenAIProviderConfig | None) -> None:
        self._llm_config = value
        self._llm_helper = None

    @property
    def extra_headers(self) -> dict[str, str]:
        return self._extra_headers

    @extra_headers.setter
    def extra_headers(self, value: dict[str, str] | None) -> None:
        self._extra_headers = dict(value or {})
        self._llm_helper = None

    @property
    def llm_helper(self) -> SimpleLLMHelper:
        """Lazy-initialised SimpleLLMHelper. Raises if llm_config not set."""
        if self._llm_config is None:
            raise ValueError(
                f"Metric '{self.name}' requires llm_config. Pass provider_config to Evaluation()."
            )
        if self._llm_helper is None:
            self._llm_helper = SimpleLLMHelper(
                self._llm_config,
                chat_model=self._llm_config.chat_model,
                extra_headers=self._extra_headers or None,
            )
        return self._llm_helper

    def evaluate(self, sample: Any, **kwargs: Any) -> MetricResult:
        """Evaluate using LLM judge (synchronous — blocking)."""
        if not self._llm_config:
            return self._missing_config_result()
        prompt = self._build_full_prompt(sample)
        try:
            logger.debug(
                "Calling LLM for criteria metric %s, model=%s",
                self.name,
                self._llm_config.chat_model,
            )
            content = self.llm_helper.generate(prompt, temperature=0.0)
            logger.debug(
                "LLM response received for %s, length=%d",
                self.name,
                len(content),
            )
            return self._parse_response(content)
        except Exception as e:  # noqa: BLE001
            logger.error("CriteriaBasedMetric %s error: %s", self.name, e, exc_info=True)
            return MetricResult(score=None, metadata={"error": str(e)})

    async def aevaluate(self, sample: Any, **kwargs: Any) -> MetricResult:
        """Evaluate using LLM judge (asynchronous — native coroutine)."""
        if not self._llm_config:
            return self._missing_config_result()
        prompt = self._build_full_prompt(sample)
        try:
            logger.debug(
                "Calling LLM for criteria metric %s, model=%s",
                self.name,
                self._llm_config.chat_model,
            )
            content = await self.llm_helper.agenerate(prompt, temperature=0.0)
            logger.debug(
                "LLM response received for %s, length=%d",
                self.name,
                len(content),
            )
            return self._parse_response(content)
        except Exception as e:  # noqa: BLE001
            logger.error(
                "CriteriaBasedMetric %s async error: %s",
                self.name,
                e,
                exc_info=True,
            )
            return MetricResult(score=None, metadata={"error": str(e)})

    def _missing_config_result(self) -> MetricResult:
        """Return error result when llm_config is not set."""
        error_msg = (
            "provider_config is required for criteria-based metrics. "
            "Pass provider_config to Evaluation when creating the evaluation instance."
        )
        logger.error("Metric %s: %s", self.name, error_msg)
        return MetricResult(
            score=None,
            metadata={
                "error": error_msg,
                "passed": False,
                "threshold": self.threshold,
            },
        )

    def _build_full_prompt(self, sample: Any) -> str:
        """Build full prompt with system instruction for LLM-as-judge."""
        question = self._extract_question(sample)
        response = self._extract_response(sample)
        prompt = self._build_prompt(question, response)
        return "You are an expert evaluator. Provide scores as requested.\n\n" + prompt

    def _extract_response(self, sample: Any) -> str:
        return getattr(sample, "llm_response", "") or ""

    def _extract_question(self, sample: Any) -> str:
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

    def _parse_response(self, content: str) -> MetricResult:
        """Parse LLM response and return MetricResult."""
        try:
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                score = float(data.get("score", 0))
                reason = data.get("reasoning", "")
                score = max(0.0, min(1.0, score))
                return MetricResult(
                    score=score,
                    metadata={
                        "passed": score >= self.threshold,
                        "threshold": self.threshold,
                        "reason": reason,
                        "chat_model": (self._llm_config.chat_model if self._llm_config else None),
                        "criteria": self.description,
                    },
                )

            numbers = re.findall(r"\d+\.?\d*", content)
            if numbers:
                score = float(numbers[0])
                if score > 1.0:
                    score = score / 5.0
                score = max(0.0, min(1.0, score))
                return MetricResult(
                    score=score,
                    metadata={
                        "passed": score >= self.threshold,
                        "threshold": self.threshold,
                        "reason": content,
                        "criteria": self.description,
                    },
                )

            return MetricResult(
                score=None,
                metadata={
                    "error": f"Could not parse response: {content}",
                    "passed": False,
                    "threshold": self.threshold,
                },
            )

        except Exception as e:  # noqa: BLE001
            return MetricResult(
                score=None,
                metadata={
                    "error": f"Parse error: {str(e)}",
                    "passed": False,
                    "threshold": self.threshold,
                },
            )


def criteria(
    name: str,
    description: str,
    threshold: float = 0.5,
    evaluation_steps: list[str] | None = None,
    execute_via: str | None = None,
    **kwargs: Any,
) -> CriteriaBasedMetric:
    """Factory: create and register a criteria-based LLM-as-judge metric.

    Args:
        name: Metric name (used as registry key).
        description: Natural language evaluation criteria.
        threshold: Pass/fail threshold (0.0-1.0). Default 0.5.
        evaluation_steps: Optional step-by-step evaluation instructions.
        execute_via: Execution backend ('ragas', 'deepeval', or None).
        **kwargs: Forwarded to CriteriaBasedMetric (e.g. llm_config).

    Returns:
        CriteriaBasedMetric instance, registered under 'custom' provider.
    """

    def factory(**init_kwargs: Any) -> CriteriaBasedMetric:
        return CriteriaBasedMetric(
            metric_name=name,
            description=description,
            threshold=threshold,
            evaluation_steps=evaluation_steps,
            execute_via=execute_via,
            **init_kwargs,
        )

    MetricRegistry().register("custom", name, factory, allow_override=True)
    return CriteriaBasedMetric(
        metric_name=name,
        description=description,
        threshold=threshold,
        evaluation_steps=evaluation_steps,
        execute_via=execute_via,
        **kwargs,
    )
