"""DeepEval client wrapper/adapter"""

from collections.abc import Mapping
from typing import Any

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from langchain_core.language_models import LanguageModelInput
from langchain_openai import ChatOpenAI

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

__VALID_TEST_CASE_SCHEMAS__ = {
    "faithfulness": FaithfulnessTestCase,
    "answer_relevancy": AnswerRelevancyTestCase,
    "contextual_precision": ContextualPrecisionTestCase,
    "contextual_recall": ContextualRecallTestCase,
    "contextual_relevancy": ContextualRelevancyTestCase,
    "hallucination": HallucinationTestCase,
    "toxicity": ToxicityTestCase,
    "exact_match": ExactMatchTestCase,
    "pattern_match": PatternMatchTestCase,
    "json_correctness": JsonCorrectnessTestCase,
}


# custom llm implementation for DeepEval
class DeepEvalLLMAdapter(DeepEvalBaseLLM):
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
        self._llm_instance = self.init_model()
        return self._llm_instance

    def generate(self, prompt: LanguageModelInput) -> str:
        chat_model = self.init_model()
        response = chat_model.invoke(prompt)
        # TODO: handle different response types (chat/completion, dict, Sequence[str])
        return response.content

    async def a_generate(self, prompt: LanguageModelInput) -> str:
        chat_model = self.init_model()
        response = await chat_model.ainvoke(prompt)
        return response.content

    def get_model_name(self) -> str:
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
        metric_test_case_schema = __VALID_TEST_CASE_SCHEMAS__[metric_name]

        if metric_name == "faithfulness":
            test_case = metric_test_case_schema.model_validate(
                self._with_safe_contexts(test_case_dict)
            )
            if not isinstance(test_case, FaithfulnessTestCase):
                raise TypeError(
                    f"Expected FaithfulnessTestCase after validation, got {type(test_case)}"
                )
            return LLMTestCase(
                input=test_case.user_input,
                expected_output=test_case.ground_truth,
                retrieval_context=test_case.contexts,
                actual_output=test_case.llm_response,
            )
        elif metric_name == "answer_relevancy":
            test_case = metric_test_case_schema.model_validate(test_case_dict)
            if not isinstance(test_case, AnswerRelevancyTestCase):
                raise TypeError(
                    f"Expected AnswerRelevancyTestCase after validation, got {type(test_case)}"
                )

            return LLMTestCase(
                input=test_case.user_input,
                actual_output=test_case.llm_response,
            )
        elif metric_name == "contextual_precision":
            test_case = metric_test_case_schema.model_validate(
                self._with_safe_contexts(test_case_dict)
            )
            if not isinstance(test_case, ContextualPrecisionTestCase):
                raise TypeError(
                    f"Expected ContextualPrecisionTestCase after validation, got {type(test_case)}"
                )

            return LLMTestCase(
                input=test_case.user_input,
                actual_output=test_case.llm_response,
                expected_output=test_case.ground_truth,
                retrieval_context=test_case.contexts,
            )
        elif metric_name == "contextual_recall":
            test_case = metric_test_case_schema.model_validate(
                self._with_safe_contexts(test_case_dict)
            )
            if not isinstance(test_case, ContextualRecallTestCase):
                raise TypeError(
                    f"Expected ContextualRecallTestCase after validation, got {type(test_case)}"
                )

            return LLMTestCase(
                input=test_case.user_input,
                actual_output=test_case.llm_response,
                expected_output=test_case.ground_truth,
                retrieval_context=test_case.contexts,
            )
        elif metric_name == "contextual_relevancy":
            test_case = metric_test_case_schema.model_validate(
                self._with_safe_contexts(test_case_dict)
            )
            if not isinstance(test_case, ContextualRelevancyTestCase):
                raise TypeError(
                    f"Expected ContextualRelevancyTestCase after validation, got {type(test_case)}"
                )

            return LLMTestCase(
                input=test_case.user_input,
                actual_output=test_case.llm_response,
                retrieval_context=test_case.contexts,
            )
        elif metric_name == "hallucination":
            test_case = metric_test_case_schema.model_validate(
                self._with_safe_contexts(test_case_dict)
            )
            if not isinstance(test_case, HallucinationTestCase):
                raise TypeError(
                    f"Expected HallucinationTestCase after validation, got {type(test_case)}"
                )

            return LLMTestCase(
                input=test_case.user_input,
                actual_output=test_case.llm_response,
                context=test_case.contexts,
            )
        elif metric_name == "toxicity":
            test_case = metric_test_case_schema.model_validate(test_case_dict)
            if not isinstance(test_case, ToxicityTestCase):
                raise TypeError(
                    f"Expected ToxicityTestCase after validation, got {type(test_case)}"
                )

            return LLMTestCase(
                input=test_case.user_input,
                actual_output=test_case.llm_response,
            )
        elif metric_name == "exact_match":
            test_case = metric_test_case_schema.model_validate(test_case_dict)
            if not isinstance(test_case, ExactMatchTestCase):
                raise TypeError(
                    f"Expected ExactMatchTestCase after validation, got {type(test_case)}"
                )

            return LLMTestCase(
                input=test_case.user_input,
                actual_output=test_case.llm_response,
                expected_output=test_case.ground_truth,
            )
        elif metric_name == "pattern_match":
            test_case = metric_test_case_schema.model_validate(test_case_dict)
            if not isinstance(test_case, PatternMatchTestCase):
                raise TypeError(
                    f"Expected PatternMatchTestCase after validation, got {type(test_case)}"
                )

            return LLMTestCase(
                input=test_case.user_input,
                actual_output=test_case.llm_response,
            )
        elif metric_name == "json_correctness":
            test_case = metric_test_case_schema.model_validate(test_case_dict)
            if not isinstance(test_case, JsonCorrectnessTestCase):
                raise TypeError(
                    f"Expected JsonCorrectnessTestCase after validation, got {type(test_case)}"
                )

            return LLMTestCase(
                input=test_case.user_input,
                actual_output=test_case.llm_response,
            )

        else:
            raise ValueError(
                f"Unsupported metric for test case transformation: {metric_name}"
            )
