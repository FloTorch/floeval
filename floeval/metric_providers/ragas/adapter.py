"""
RAGAS adapter for custom LLM integration.

Key points:
- Uses unified LLMProviderConfig for consistency across providers.
- Provides RAGASAdapter class similar to DeepEvalAdapter for consistency.
"""

from typing import Any, Dict, Sequence

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

from floeval.config.schemas.io.dataset import Sample
from floeval.config.schemas.io.llm import LLMProviderConfig
from floeval.utils.gateway import normalize_openai_api_base


def create_ragas_llm(config: LLMProviderConfig | None = None) -> LangchainLLMWrapper:
    """Create RAGAS LLM wrapper configured with custom llm configuration.

    Args:
        config: LLMProviderConfig configuration (optional, uses env defaults if None)

    Returns:
        LangchainLLMWrapper instance configured with custom llm configuration
    """
    llm_args: Dict[str, Any] = {}
    if config and config.base_url:
        llm_args["openai_api_base"] = normalize_openai_api_base(config.base_url)
    if config and config.api_key:
        llm_args["openai_api_key"] = config.api_key
    if config and config.chat_model:
        llm_args["model"] = config.chat_model
    llm = ChatOpenAI(**llm_args)
    return LangchainLLMWrapper(llm)


def create_ragas_embeddings(
    config: LLMProviderConfig | None = None,
) -> LangchainEmbeddingsWrapper:
    """Create RAGAS embeddings wrapper configured with custom llm configuration.

    Args:
        config: LLMProviderConfig configuration (optional, uses env defaults if None)

    Returns:
        LangchainEmbeddingsWrapper instance configured with custom llm configuration
    """
    embedding_args: Dict[str, Any] = {
        "check_embedding_ctx_length": False,
    }
    if config and config.base_url:
        embedding_args["openai_api_base"] = normalize_openai_api_base(config.base_url)
    if config and config.api_key:
        embedding_args["openai_api_key"] = config.api_key
    if config and config.embedding_model:
        embedding_args["model"] = config.embedding_model
    embeddings = OpenAIEmbeddings(**embedding_args)
    return LangchainEmbeddingsWrapper(embeddings=embeddings)


class RAGASAdapter:
    """Adapter for RAGAS client integration.

    Adapts input and output formats as needed by RAGAS library.
    Similar to DeepEvalAdapter for consistency across providers.
    """

    def __init__(self, config: LLMProviderConfig | None = None):
        """Initialize RAGAS adapter with llm configuration.

        Args:
            config: Optional LLMProviderConfig configuration. If None, uses environment defaults.
        """
        self.config = config
        self._llm: LangchainLLMWrapper | None = None
        self._embeddings: LangchainEmbeddingsWrapper | None = None

    @property
    def llm(self) -> LangchainLLMWrapper:
        """Get or create RAGAS LLM wrapper (cached)."""
        if self._llm is None:
            self._llm = create_ragas_llm(self.config)
        return self._llm

    @property
    def embeddings(self) -> LangchainEmbeddingsWrapper:
        """Get or create RAGAS embeddings wrapper (cached)."""
        if self._embeddings is None:
            self._embeddings = create_ragas_embeddings(self.config)
        return self._embeddings

    def transform_sample(
        self, sample: Sample | dict[str, str | Sequence[Any]]
    ) -> SingleTurnSample:
        """
        Convert Floeval Sample to RAGAS SingleTurnSample format.

        Supports both Pydantic Sample models and dict-like objects.

        Args:
            sample: Floeval Sample object with inputs and ground_truth

        Returns:
            SingleTurnSample for RAGAS evaluation

        Raises:
            ValueError: If sample doesn't have required fields
        """

        # Handle Pydantic models
        if isinstance(sample, Sample):
            sample_data = sample.model_dump()
        # Handle dict-like objects
        elif isinstance(sample, dict):
            sample_data = sample
        else:
            raise ValueError(f"Unsupported sample type: {type(sample)}. Expected Sample or dict.")

        # Extract fields with defaults
        user_input = sample_data.get("user_input", "")
        contexts = sample_data.get("contexts", [])
        llm_response = sample_data.get("llm_response", "")
        ground_truth = sample_data.get("ground_truth", "")

        # TODO: Use model attributes instead of hardcoded keys? (.model_validate() for the validation and transformation logic)

        return SingleTurnSample(
            user_input=user_input,
            retrieved_contexts=contexts if isinstance(contexts, list) else [contexts],
            response=llm_response,
            reference=ground_truth,
        )
