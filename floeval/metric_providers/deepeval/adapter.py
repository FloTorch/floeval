"""DeepEval client wrapper/adapter"""

from collections.abc import Mapping
from typing import Any

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from langchain_core.language_models import LanguageModelInput
from langchain_openai import ChatOpenAI

from floeval.config.schemas.deepeval import AnswerRelevancyTestCase, FaithfulnessTestCase
from floeval.config.schemas.io.llm import LLMProviderConfig
from floeval.utils.gateway import normalize_openai_api_base

__VALID_TEST_CASE_SCHEMAS__ = {
    "faithfulness": FaithfulnessTestCase,
    "answer_relevancy": AnswerRelevancyTestCase,
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
            # Only set temperature/max_tokens if explicitly provided (let provider use defaults otherwise)
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
                kwargs["base_url"] = normalize_openai_api_base(base_url)

        self._llm_instance = ChatOpenAI(**kwargs)
        return self._llm_instance

    def load_model(self, *args, **kwargs):
        self._llm_instance = self.init_model()
        return self._llm_instance

    def generate(self, prompt: LanguageModelInput) -> str:
        chat_model = self.init_model()
        response = chat_model.invoke(prompt)
        # TODO: handle different response types (chat/completion), dict[str, Any], Sequence[str] etc.
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
            test_case = metric_test_case_schema.model_validate(test_case_dict)
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

        else:
            raise ValueError(f"Unsupported metric for test case transformation: {metric_name}")
