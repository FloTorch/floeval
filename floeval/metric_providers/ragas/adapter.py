"""
RAGAS adapter for custom gateway integration.

Key points:
- Uses unified GatewayConfig for consistency across providers.
- Provides RAGASAdapter class similar to DeepEvalAdapter for consistency.
"""

from collections.abc import Mapping
from typing import Any, Dict, Optional

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
except ImportError as e:
    raise ImportError(
        f"RAGAS dependencies not installed. Please install: "
        f"ragas>=0.4.3, langchain-openai. Original error: {e}"
    )

from floeval.api.dataset import Sample
from floeval.config import GatewayConfig


def normalize_openai_api_base(url: str) -> str:
    """
    Normalize a gateway URL into an OpenAI-compatible API base.
    """
    raw = url.strip()
    if not raw:
        raise ValueError("gateway_base_url cannot be empty.")

    if not raw.startswith(("http://", "https://")):
        raw = f"https://{raw}"

    # remove trailing slash
    raw = raw.rstrip("/")

    # if a full endpoint is provided, strip it back to the base
    for suffix in ("/chat/completions", "/embeddings"):
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)]
            raw = raw.rstrip("/")

    # If caller already provided a base like /openai/v1 or /v1, keep it.
    if raw.endswith("/openai/v1") or raw.endswith("/v1"):
        return raw

    # Default for FloTorch gateways
    return f"{raw}/openai/v1"


def create_ragas_llm(config: Optional[GatewayConfig] = None) -> LangchainLLMWrapper:
    """
    Create RAGAS LLM wrapper configured with custom gateway.
    
    Args:
        config: Gateway configuration (optional, uses env defaults if None)
        
    Returns:
        LangchainLLMWrapper instance configured with custom gateway
    """
    llm_args: Dict[str, Any] = {}
    if config and config.gateway_base_url:
        llm_args["openai_api_base"] = normalize_openai_api_base(config.gateway_base_url)
    if config and config.api_key:
        llm_args["openai_api_key"] = config.api_key
    if config and config.llm_model:
        llm_args["model"] = config.llm_model
    llm = ChatOpenAI(**llm_args)
    return LangchainLLMWrapper(llm)


def create_ragas_embeddings(config: Optional[GatewayConfig] = None) -> LangchainEmbeddingsWrapper:
    """
    Create RAGAS embeddings wrapper configured with custom gateway.
    
    Args:
        config: Gateway configuration (optional, uses env defaults if None)
        
    Returns:
        LangchainEmbeddingsWrapper instance configured with custom gateway
    """
    embedding_args: Dict[str, Any] = {
        "check_embedding_ctx_length": False,
    }
    if config and config.gateway_base_url:
        embedding_args["openai_api_base"] = normalize_openai_api_base(config.gateway_base_url)
    if config and config.api_key:
        embedding_args["openai_api_key"] = config.api_key
    if config and config.embedding_model:
        embedding_args["model"] = config.embedding_model
    embeddings = OpenAIEmbeddings(**embedding_args)
    return LangchainEmbeddingsWrapper(embeddings=embeddings)


class RAGASAdapter:
    """
    Adapter for RAGAS client integration.
    Adapts input and output formats as needed by RAGAS library.
    Similar to DeepEvalAdapter for consistency across providers.
    """

    def __init__(self, config: Optional[GatewayConfig] = None):
        """
        Initialize RAGAS adapter with gateway configuration.
        
        Args:
            config: Optional gateway configuration. If None, uses environment defaults.
        """
        self.config = config
        self._llm: Optional[LangchainLLMWrapper] = None
        self._embeddings: Optional[LangchainEmbeddingsWrapper] = None

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

    def transform_sample(self, sample: Sample) -> SingleTurnSample:
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
        if hasattr(sample, "model_dump"):
            sample_data = sample.model_dump()
        elif hasattr(sample, "dict"):
            sample_data = sample.dict()
        # Handle dict-like objects
        elif isinstance(sample, dict):
            sample_data = sample
        # Handle objects with attributes
        elif hasattr(sample, "inputs") and hasattr(sample, "ground_truth"):
            sample_data = {
                "inputs": getattr(sample, "inputs", {}),
                "ground_truth": getattr(sample, "ground_truth", ""),
            }
        else:
            raise ValueError(
                f"Sample must be a dict, Pydantic model, or object with "
                f"'inputs' and 'ground_truth' attributes. Got: {type(sample)}"
            )

        inputs = sample_data.get("inputs", {})

        # Extract fields with defaults
        user_input = inputs.get("user_input", "")
        contexts = inputs.get("contexts", [])
        llm_response = inputs.get("llm_response", "")
        ground_truth = sample_data.get("ground_truth", "")

        return SingleTurnSample(
            user_input=user_input,
            retrieved_contexts=contexts if isinstance(contexts, list) else [contexts],
            response=llm_response,
            reference=ground_truth,
        )
