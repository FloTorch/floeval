"""DeepEval client wrapper and adapter."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from langchain_core.language_models import LanguageModelInput
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from floeval.config.schemas.deepeval import (
    AnswerRelevancyTestCase,
    ContextualPrecisionTestCase,
    ContextualRecallTestCase,
    ContextualRelevancyTestCase,
    ExactMatchTestCase,
    FaithfulnessTestCase,
    HallucinationTestCase,
    JsonCorrectnessTestCase,
    PatternMatchTestCase,
    ToxicityTestCase,
)
from floeval.config.schemas.io.llm import LLMProviderConfig, _normalize_openai_base_url


@dataclass(frozen=True)
class _LLMTestCaseSpec:
    """How to validate a dict and build a DeepEval ``LLMTestCase`` for one metric."""

    schema: type[BaseModel]
    normalize_none_contexts: bool
    build: Callable[[BaseModel], LLMTestCase]


def _validate_subtype(tc: BaseModel, expected: type[Any], metric_name: str) -> None:
    if not isinstance(tc, expected):
        msg = f"Expected {expected.__name__} after validation for {metric_name}, got {type(tc)}"
        raise TypeError(msg)


def _faithfulness(tc: BaseModel) -> LLMTestCase:
    _validate_subtype(tc, FaithfulnessTestCase, "faithfulness")
    c = cast(FaithfulnessTestCase, tc)
    return LLMTestCase(
        input=c.user_input,
        expected_output=c.ground_truth,
        retrieval_context=c.contexts,
        actual_output=c.llm_response,
    )


def _answer_relevancy(tc: BaseModel) -> LLMTestCase:
    _validate_subtype(tc, AnswerRelevancyTestCase, "answer_relevancy")
    c = cast(AnswerRelevancyTestCase, tc)
    return LLMTestCase(
        input=c.user_input,
        actual_output=c.llm_response,
    )


def _contextual_precision(tc: BaseModel) -> LLMTestCase:
    _validate_subtype(tc, ContextualPrecisionTestCase, "contextual_precision")
    c = cast(ContextualPrecisionTestCase, tc)
    return LLMTestCase(
        input=c.user_input,
        actual_output=c.llm_response,
        expected_output=c.ground_truth,
        retrieval_context=c.contexts,
    )


def _contextual_recall(tc: BaseModel) -> LLMTestCase:
    _validate_subtype(tc, ContextualRecallTestCase, "contextual_recall")
    c = cast(ContextualRecallTestCase, tc)
    return LLMTestCase(
        input=c.user_input,
        actual_output=c.llm_response,
        expected_output=c.ground_truth,
        retrieval_context=c.contexts,
    )


def _contextual_relevancy(tc: BaseModel) -> LLMTestCase:
    _validate_subtype(tc, ContextualRelevancyTestCase, "contextual_relevancy")
    c = cast(ContextualRelevancyTestCase, tc)
    return LLMTestCase(
        input=c.user_input,
        actual_output=c.llm_response,
        retrieval_context=c.contexts,
    )


def _hallucination(tc: BaseModel) -> LLMTestCase:
    _validate_subtype(tc, HallucinationTestCase, "hallucination")
    c = cast(HallucinationTestCase, tc)
    return LLMTestCase(
        input=c.user_input,
        actual_output=c.llm_response,
        context=c.contexts,
    )


def _toxicity(tc: BaseModel) -> LLMTestCase:
    _validate_subtype(tc, ToxicityTestCase, "toxicity")
    c = cast(ToxicityTestCase, tc)
    return LLMTestCase(
        input=c.user_input,
        actual_output=c.llm_response,
    )


def _exact_match(tc: BaseModel) -> LLMTestCase:
    _validate_subtype(tc, ExactMatchTestCase, "exact_match")
    c = cast(ExactMatchTestCase, tc)
    return LLMTestCase(
        input=c.user_input,
        actual_output=c.llm_response,
        expected_output=c.ground_truth,
    )


def _pattern_match(tc: BaseModel) -> LLMTestCase:
    _validate_subtype(tc, PatternMatchTestCase, "pattern_match")
    c = cast(PatternMatchTestCase, tc)
    return LLMTestCase(
        input=c.user_input,
        actual_output=c.llm_response,
    )


def _json_correctness(tc: BaseModel) -> LLMTestCase:
    _validate_subtype(tc, JsonCorrectnessTestCase, "json_correctness")
    c = cast(JsonCorrectnessTestCase, tc)
    return LLMTestCase(
        input=c.user_input,
        actual_output=c.llm_response,
    )


_LLM_TEST_CASE_SPECS: dict[str, _LLMTestCaseSpec] = {
    "faithfulness": _LLMTestCaseSpec(FaithfulnessTestCase, True, _faithfulness),
    "answer_relevancy": _LLMTestCaseSpec(AnswerRelevancyTestCase, False, _answer_relevancy),
    "contextual_precision": _LLMTestCaseSpec(
        ContextualPrecisionTestCase, True, _contextual_precision
    ),
    "contextual_recall": _LLMTestCaseSpec(ContextualRecallTestCase, True, _contextual_recall),
    "contextual_relevancy": _LLMTestCaseSpec(
        ContextualRelevancyTestCase, True, _contextual_relevancy
    ),
    "hallucination": _LLMTestCaseSpec(HallucinationTestCase, True, _hallucination),
    "toxicity": _LLMTestCaseSpec(ToxicityTestCase, False, _toxicity),
    "exact_match": _LLMTestCaseSpec(ExactMatchTestCase, False, _exact_match),
    "pattern_match": _LLMTestCaseSpec(PatternMatchTestCase, False, _pattern_match),
    "json_correctness": _LLMTestCaseSpec(JsonCorrectnessTestCase, False, _json_correctness),
}

# Backwards-compatible name for imports that referenced the old dict.
__VALID_TEST_CASE_SCHEMAS__ = {name: spec.schema for name, spec in _LLM_TEST_CASE_SPECS.items()}


# custom llm implementation for DeepEval
class DeepEvalLLMAdapter(DeepEvalBaseLLM):
    """OpenAI-compatible chat model exposed as DeepEval's base LLM."""

    def __init__(self, model_name: str, config: LLMProviderConfig):
        self._model_name = model_name
        self.config = config
        self._llm_instance = None
        _llm_instance = self.init_model()
        self._llm_instance = _llm_instance

    def init_model(self):
        """Load and cache ChatOpenAI instance with llm config."""
        if self._llm_instance is not None:
            return self._llm_instance

        kwargs: dict[str, Any] = {}

        if self.config:
            chat_model = getattr(self.config, "chat_model", None)
            if chat_model:
                kwargs["model"] = chat_model
            # Set temperature/max_tokens only if provided; else use provider defaults
            temperature = getattr(self.config, "temperature", None)
            if temperature is not None:
                kwargs["temperature"] = temperature
            max_tokens = getattr(self.config, "max_tokens", None)
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens

            api_key = getattr(self.config, "api_key", None)
            if api_key:
                kwargs["api_key"] = api_key

            base_url = getattr(self.config, "base_url", None)
            if base_url:
                # Normalize llm URL to OpenAI-compatible format (same as RAGAS)
                kwargs["base_url"] = _normalize_openai_base_url(base_url)

        self._llm_instance = ChatOpenAI(**kwargs)
        return self._llm_instance

    def load_model(self, *args, **kwargs):
        """Rebuild the chat model instance and return it."""
        self._llm_instance = self.init_model()
        return self._llm_instance

    def generate(self, prompt: LanguageModelInput) -> str:
        """Run a synchronous completion and return text content."""
        chat_model = self.init_model()
        response = chat_model.invoke(prompt)
        # TODO: handle different response types (chat/completion, dict, Sequence[str])
        return response.content

    async def a_generate(self, prompt: LanguageModelInput) -> str:
        """Run an async completion and return text content."""
        chat_model = self.init_model()
        response = await chat_model.ainvoke(prompt)
        return response.content

    def get_model_name(self) -> str:
        """Return the configured model label for DeepEval."""
        return self._model_name


class DeepEvalAdapter:
    """Adapter for DeepEval client integration.

    Adapts input and output formats as needed by deepeval library.
    """

    def __init__(self, config: Mapping[str, Any]):
        self._inp_config = config

        # TODO: Should we validate self._inp_config against a schema?
        self.config = self._inp_config

    @staticmethod
    def _with_safe_contexts(
        test_case_dict: Mapping[str, str | list[str] | None],
    ) -> dict[str, str | list[str] | None]:
        """Return a shallow copy with context fields normalized for DeepEval schemas.

        Some datasets provide `contexts=None`. Context-dependent DeepEval test-case
        schemas require a list, so normalize None -> [] before pydantic validation.
        """
        normalized = dict(test_case_dict)
        if normalized.get("contexts") is None:
            normalized["contexts"] = []
        return normalized

    def transform_test_case(
        self, metric_name: str, test_case_dict: Mapping[str, str | list[str] | None]
    ) -> LLMTestCase:
        """Convert a test case dictionary to DeepEval LLMTestCase format.

        Supports canonical Floeval keys (question, answer, contexts, expected_answer)
        and maps them to DeepEval's expected keys (user_input, actual_output, etc.).

        Args:
            metric_name: Name of the metric (faithfulness, answer_relevancy)
            test_case_dict: Input test case mapping (may use canonical or DeepEval keys)

        Returns:
            LLMTestCase: Adapted test case instance
        """
        spec = _LLM_TEST_CASE_SPECS.get(metric_name)
        if spec is None:
            raise ValueError(f"Unsupported metric for test case transformation: {metric_name}")

        raw: dict[str, str | list[str] | None]
        if spec.normalize_none_contexts:
            raw = self._with_safe_contexts(test_case_dict)
        else:
            raw = dict(test_case_dict)

        validated = spec.schema.model_validate(raw)
        return spec.build(validated)
