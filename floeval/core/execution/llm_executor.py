import logging
from typing import Literal

# import openai
from langchain_openai import ChatOpenAI
from openai._types import Omit, SequenceNotStr

from floeval.config.schemas.io.llm import OpenAIProviderConfig
from floeval.core.execution.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI compatible llm provider implementation for response/embedding generation."""

    def __init__(self, config_name: str, **kwargs):
        super().__init__()
        # name: to reference this provider config during dataset synthesis or metric evaluation
        self.name = config_name
        self.provider_config = OpenAIProviderConfig(**kwargs)
        self._llm_client = self._initialize_llm_client()

    def _initialize_llm_client(self):
        """Initialize the LLM client based on the provider configuration."""
        # return openai.Client(
        #     base_url=self.provider_config.provider_base_url,
        #     api_key=self.provider_config.api_key,
        # )
        return ChatOpenAI(
            base_url=self.provider_config.provider_base_url,
            model=self.provider_config.chat_model,
            api_key=self.provider_config.api_key,
            **(self.provider_config.extra_kwargs or {}),
        )

    def generate(
        self,
        prompt: str,
        **kwargs,
    ) -> str:
        """Generate a response from the LLM based on the given prompt and system prompt.

        Args:
            prompt: The user prompt to send to the LLM.
            kwargs: Additional keyword arguments

        Returns:
            response: The generated response from the LLM.
        """
        # to explicitly omit the instructions field if no system prompt is provided,
        # we can use the Omit type from the OpenAI _types
        _instruction: str | Omit = Omit()

        if self.provider_config.system_prompt and isinstance(
            self.provider_config.system_prompt, str
        ):
            _instruction = self.provider_config.system_prompt + "\n\n"
        _output_text = self._llm_client.invoke(prompt).content
        if not isinstance(_output_text, str):
            logger.warning(
                f"Expected output_text to be a string, but got {type(_output_text)}"
            )
            return _output_text
        return _output_text

    def generate_embedding(
        self, input: str | SequenceNotStr[str], **kwargs
    ) -> list[float]:
        """Generate an embedding from the LLM based on the given input.

        Args:
            input: content to be embedded by the LLM. Can be a single string or an array of strings.
            **kwargs: Additional keyword arguments

        Returns:
            The generated embedding(s) as a list of floats, or None if embedding generation failed.
        """
        _encoding_format: Literal["float"] = "float"
        if (
            not self.provider_config.embedding_model
            or not self.provider_config.embedding_endpoint
        ):
            raise ValueError(
                "Embedding model and endpoint must be specified in the provider configuration"
                " to generate embeddings."
            )
        try:
            response = self._llm_client.embeddings.create(
                model=self.provider_config.embedding_model,
                input=input,
                encoding_format=_encoding_format,
                **kwargs,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise Exception(f"Error generating embedding: {e}") from e
