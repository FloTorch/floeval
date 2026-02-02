"""
RAGAS adapter for custom gateway integration.

Key points:
- Users provide a single `gateway_base_url` + `api_key` + model ids.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas import SingleTurnSample
except ImportError as e:
    raise ImportError(
        f"RAGAS dependencies not installed. Please install: "
        f"ragas>=0.4.3, langchain-openai. Original error: {e}"
    )


class RAGASGatewayConfig(BaseModel):
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
    gateway_base_url: Optional[str] = Field(
        default=None, description="Base URL for custom gateway (OpenAI-compatible)"
    )
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    llm_model: Optional[str] = Field(default=None, description="LLM model identifier")
    embedding_model: Optional[str] = Field(default=None, description="Embedding model identifier")


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


def create_ragas_llm(config: RAGASGatewayConfig) -> LangchainLLMWrapper:
    """
    Create RAGAS LLM wrapper configured with custom gateway.
    
    Args:
        config: Gateway configuration
        
    Returns:
        LangchainLLMWrapper instance configured with custom gateway
    """
    llm_args: Dict[str, Any] = {}
    if config.gateway_base_url:
        llm_args["openai_api_base"] = normalize_openai_api_base(config.gateway_base_url)
    if config.api_key:
        llm_args["openai_api_key"] = config.api_key
    if config.llm_model:
        llm_args["model"] = config.llm_model
    llm = ChatOpenAI(**llm_args)
    return LangchainLLMWrapper(llm)


def create_ragas_embeddings(config: RAGASGatewayConfig) -> LangchainEmbeddingsWrapper:
    """
    Create RAGAS embeddings wrapper configured with custom gateway.
    
    Args:
        config: Gateway configuration
        
    Returns:
        LangchainEmbeddingsWrapper instance configured with custom gateway
    """
    embedding_args: Dict[str, Any] = {
        "check_embedding_ctx_length": False,
    }
    if config.gateway_base_url:
        embedding_args["openai_api_base"] = normalize_openai_api_base(config.gateway_base_url)
    if config.api_key:
        embedding_args["openai_api_key"] = config.api_key
    if config.embedding_model:
        embedding_args["model"] = config.embedding_model
    embeddings = OpenAIEmbeddings(**embedding_args)
    return LangchainEmbeddingsWrapper(embeddings=embeddings)


def sample_to_ragas(sample: Any) -> SingleTurnSample:
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
            "ground_truth": getattr(sample, "ground_truth", {}),
        }
    else:
        raise ValueError(
            f"Sample must be a dict, Pydantic model, or object with "
            f"'inputs' and 'ground_truth' attributes. Got: {type(sample)}"
        )
    
    inputs = sample_data.get("inputs", {})
    ground_truth = sample_data.get("ground_truth", {}) or {}
    
    # Extract fields with defaults
    question = inputs.get("question", "")
    contexts = inputs.get("contexts", [])
    answer = inputs.get("answer", "")
    expected_answer = ground_truth.get("expected_answer") if ground_truth else None
    
    return SingleTurnSample(
        user_input=question,
        retrieved_contexts=contexts if isinstance(contexts, list) else [contexts],
        response=answer,
        reference=expected_answer,
    )
