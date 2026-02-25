"""
RAGAS adapter for custom LLM integration.

Key points:
- Uses unified LLMProviderConfig for consistency across providers.
- Provides RAGASAdapter class similar to DeepEvalAdapter for consistency.
- Agent metrics (agent_goal_accuracy) use LangChain-based structured LLM by default,
  which works with any OpenAI-compatible API (no response_format required).
"""

import json
import re
from typing import Any, Dict, Sequence, Type, TypeVar

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel
from ragas import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.messages import AIMessage, HumanMessage, ToolCall as RAGASToolCall, ToolMessage

from floeval.config.schemas.io.agent_dataset import (
    AgentSample,
    AIMessage as FloevalAIMessage,
    HumanMessage as FloevalHumanMessage,
    ToolMessage as FloevalToolMessage,
)
from floeval.config.schemas.io.dataset import Sample
from floeval.config.schemas.io.llm import LLMProviderConfig, _normalize_openai_base_url

T = TypeVar("T", bound=BaseModel)


class LangChainStructuredLLM(InstructorBaseRagasLLM):
    """LangChain-based LLM implementing InstructorBaseRagasLLM interface.

    Uses ChatOpenAI with plain completion (no response_format). RAGAS prompts
    already ask for JSON output; we parse the response and validate into the
    Pydantic model. Works with any OpenAI-compatible API that does not support
    response_format.
    """

    def __init__(self, config: LLMProviderConfig | None = None):
        llm_args: Dict[str, Any] = {"temperature": 0.01, "model": "gpt-4o-mini"}
        if config:
            if config.base_url:
                llm_args["openai_api_base"] = _normalize_openai_base_url(config.base_url)
            if config.api_key:
                llm_args["openai_api_key"] = config.api_key
            if config.chat_model:
                llm_args["model"] = config.chat_model
        self._llm = ChatOpenAI(**llm_args)

    def generate(self, prompt: str, response_model: Type[T]) -> T:
        """Sync generate - runs async in loop."""
        import asyncio

        return asyncio.run(self.agenerate(prompt, response_model))

    async def agenerate(self, prompt: str, response_model: Type[T]) -> T:
        """Generate structured output via ChatOpenAI + JSON parse."""
        msg = await self._llm.ainvoke(prompt)
        text = msg.content if hasattr(msg, "content") else str(msg)
        parsed = _extract_json(text)
        return response_model.model_validate(parsed)


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response (handles markdown code blocks)."""
    if not text or not text.strip():
        raise ValueError("Empty LLM response")
    text = text.strip()
    # Try ```json ... ``` first
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return json.loads(match.group(1).strip())
    # Try raw JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find {...} in text
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"Could not extract JSON from response: {text[:200]}...")


def create_ragas_llm(config: LLMProviderConfig | None = None) -> LangchainLLMWrapper:
    """Create RAGAS LLM wrapper configured with custom llm configuration.

    Args:
        config: LLMProviderConfig configuration (optional, uses env defaults if None)

    Returns:
        LangchainLLMWrapper instance configured with custom llm configuration
    """
    llm_args: Dict[str, Any] = {}
    if config and config.base_url:
        llm_args["openai_api_base"] = _normalize_openai_base_url(config.base_url)
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
        embedding_args["openai_api_base"] = _normalize_openai_base_url(config.base_url)
    if config and config.api_key:
        embedding_args["openai_api_key"] = config.api_key
    if config and config.embedding_model:
        embedding_args["model"] = config.embedding_model
    embeddings = OpenAIEmbeddings(**embedding_args)
    return LangchainEmbeddingsWrapper(embeddings=embeddings)


def create_ragas_instructor_llm(
    config: LLMProviderConfig | None = None,
):
    """Create LLM for RAGAS agent metrics (agent_goal_accuracy).

    Uses LangChain ChatOpenAI + JSON parsing (no response_format), so it works
    with any OpenAI-compatible API. Pass the same llm_config
    as the rest of evaluation - no separate RAGAS config needed.
    """
    return LangChainStructuredLLM(config)


def transform_agent_sample_to_ragas_messages(
    sample: AgentSample,
) -> list[HumanMessage | AIMessage | ToolMessage]:
    """Convert AgentSample trace messages to RAGAS message format."""
    result: list[HumanMessage | AIMessage | ToolMessage] = []
    for msg in sample.trace.messages:
        if isinstance(msg, FloevalHumanMessage):
            result.append(HumanMessage(content=msg.content))
        elif isinstance(msg, FloevalAIMessage):
            tool_calls = [RAGASToolCall(name=tc.name, args=tc.args) for tc in msg.tool_calls]
            result.append(AIMessage(content=msg.content, tool_calls=tool_calls))
        elif isinstance(msg, FloevalToolMessage):
            result.append(
                ToolMessage(
                    content=msg.content,
                    tool_call_id=msg.tool_call_id or "",
                )
            )
    return result


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
        self._agent_llm = None

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

    @property
    def agent_llm(self):
        """LLM for agent metrics (agent_goal_accuracy, tool_call_accuracy).
        Uses LangChain + JSON parse; works with any OpenAI-compatible API.
        """
        if self._agent_llm is None:
            self._agent_llm = create_ragas_instructor_llm(self.config)
        return self._agent_llm

    def transform_sample(self, sample: Sample | dict[str, str | Sequence[Any]]) -> SingleTurnSample:
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

        # TODO: Use model attributes instead of hardcoded keys?

        return SingleTurnSample(
            user_input=user_input,
            retrieved_contexts=contexts if isinstance(contexts, list) else [contexts],
            response=llm_response,
            reference=ground_truth,
        )
