"""
DeepEval client wrapper/adapter
"""

from collections.abc import Mapping
from typing import Any

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from langchain_core.language_models import LanguageModelInput
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from floeval.config.schemas.deepeval import FaithfulnessTestCase

__VALID_TEST_CASE_SCHEMAS__ = {
    "faithfulness": FaithfulnessTestCase,
}


class DeepEvalGatewayConfig(BaseModel):
    """
    Configuration for RAGAS custom gateway integration.

    Attributes:
        gateway_base_url: Base URL for the custom API gateway
        api_key: API key for authentication
        llm_model: Model identifier for LLM calls
        embedding_model: Model identifier for embedding calls
        headers: Additional headers to include in requests
    """

    # All fields are optional so we can support:
    # - explicit custom gateway configuration (provide base_url/api_key/models)
    # - environment-driven defaults (OPENAI_API_KEY, etc.) when fields are omitted
    gateway_base_url: str | None = Field(
        default=None, description="Base URL for custom gateway (OpenAI-compatible)"
    )
    api_key: str | None = Field(default=None, description="API key for authentication")
    llm_model: str | None = Field(default=None, description="LLM model identifier")
    embedding_model: str | None = Field(default=None, description="Embedding model identifier")

    temperature: float = Field(default=0.7, description="Temperature for LLM generation")
    max_tokens: int = Field(default=1024, description="Max tokens for LLM generation")


# custom llm implementation for DeepEval
class DeepEvalLLMAdapter(DeepEvalBaseLLM):
    def __init__(self, model_name: str, config: DeepEvalGatewayConfig):
        self._model_name = model_name
        self.config = config
        self._llm_instance = self.init_model()

    def init_model(self):
        """Load and cache ChatOpenAI instance with gateway config."""
        if self._llm_instance is not None:
            return self._llm_instance

        kwargs = {
            "model": self.config.llm_model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key

        if self.config.gateway_base_url:
            kwargs["base_url"] = self.config.gateway_base_url

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
    """
    Adapter for DeepEval client integration.
    Adapts input and output formats as needed by deepeval library.
    """

    def __init__(self, config: Mapping[str, Any]):
        self._inp_config = config

        # TODO: Should we validate self._inp_config against a schema?
        self.config = self._inp_config

    def transform_test_case(
        self, metric_name: str, test_case_dict: Mapping[str, str | list[str] | None]
    ) -> LLMTestCase:
        """
        Convert a test case dictionary to DeepEval LLMTestCase format.

        Args:
            test_case_dict: Input test case mapping

        Returns:
            LLMTestCase: Adapted test case instance
        """
        metric_test_case_schema = __VALID_TEST_CASE_SCHEMAS__[metric_name]
        faithfulness_test_case = metric_test_case_schema.model_validate(test_case_dict)

        return LLMTestCase(
            input=faithfulness_test_case.question,
            actual_output=faithfulness_test_case.answer,
            retrieval_context=faithfulness_test_case.contexts,
        )
