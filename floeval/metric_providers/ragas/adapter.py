"""RAGAS adapter and conversion helpers."""

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
from floeval.config.schemas.io.conversational_dataset import ConversationalSample
from floeval.config.schemas.io.dataset import Sample
from floeval.config.schemas.io.llm import LLMProviderConfig, _normalize_openai_base_url
from floeval.metric_providers.ragas.multiturn_adapter import (
    conversational_sample_to_ragas_multiturn,
)
from floeval.utils.asyncio_compat import run_coroutine_sync

T = TypeVar("T", bound=BaseModel)


class LangChainStructuredLLM(InstructorBaseRagasLLM):
    """ChatOpenAI wrapper that validates JSON output into Pydantic models."""

    def __init__(
        self,
        config: LLMProviderConfig | None = None,
        extra_headers: Dict[str, str] | None = None,
    ):
        llm_args: Dict[str, Any] = {"temperature": 0.01, "model": "gpt-4o-mini"}
        if config:
            if config.base_url:
                llm_args["openai_api_base"] = _normalize_openai_base_url(config.base_url)
            if config.api_key:
                llm_args["openai_api_key"] = config.api_key
            if config.chat_model:
                llm_args["model"] = config.chat_model
        if extra_headers:
            llm_args["default_headers"] = extra_headers
        self._llm = ChatOpenAI(**llm_args)

    def generate(self, prompt: str, response_model: Type[T]) -> T:
        """Synchronous adapter over `agenerate`."""
        return run_coroutine_sync(lambda: self.agenerate(prompt, response_model))

    async def agenerate(self, prompt: str, response_model: Type[T]) -> T:
        """Generate structured output via ChatOpenAI with schema enforcement."""
        structured_llm = self._llm.with_structured_output(response_model)
        return await structured_llm.ainvoke(prompt)  # type: ignore[return-value]


def create_ragas_llm(
    config: LLMProviderConfig | None = None,
    extra_headers: Dict[str, str] | None = None,
) -> LangchainLLMWrapper:
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
    if extra_headers:
        llm_args["default_headers"] = extra_headers
    llm = ChatOpenAI(**llm_args)
    return LangchainLLMWrapper(llm)


def create_ragas_embeddings(
    config: LLMProviderConfig | None = None,
    extra_headers: Dict[str, str] | None = None,
):
    """Build the default RAGAS embeddings wrapper from provider config."""
    embedding_args: Dict[str, Any] = {
        "check_embedding_ctx_length": False,
    }
    if config and config.base_url:
        embedding_args["openai_api_base"] = _normalize_openai_base_url(config.base_url)
    if config and config.api_key:
        embedding_args["openai_api_key"] = config.api_key
    if config and config.embedding_model:
        embedding_args["model"] = config.embedding_model
    if extra_headers:
        embedding_args["default_headers"] = extra_headers
    embeddings = OpenAIEmbeddings(**embedding_args)
    return LangchainEmbeddingsWrapper(embeddings=embeddings)


def create_ragas_instructor_llm(
    config: LLMProviderConfig | None = None,
    extra_headers: Dict[str, str] | None = None,
):
    """Create LLM for RAGAS agent metrics (agent_goal_accuracy).

    Uses LangChain ChatOpenAI + JSON parsing (no response_format), so it works
    with any OpenAI-compatible API. Pass the same llm_config
    as the rest of evaluation - no separate RAGAS config needed.
    """
    return LangChainStructuredLLM(config, extra_headers=extra_headers)


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


def transform_sample_to_ragas_messages(
    sample: AgentSample | ConversationalSample,
) -> list[HumanMessage | AIMessage | ToolMessage]:
    """Convert either ``AgentSample`` or ``ConversationalSample`` to RAGAS messages."""
    if isinstance(sample, AgentSample):
        return transform_agent_sample_to_ragas_messages(sample)
    multiturn = conversational_sample_to_ragas_multiturn(sample)
    return multiturn.user_input


class RAGASAdapter:
    """Adapter between FloEval sample types and RAGAS objects."""

    def __init__(
        self,
        config: LLMProviderConfig | None = None,
        extra_headers: Dict[str, str] | None = None,
    ):
        """Initialize RAGAS adapter with llm configuration.

        Args:
            config: Optional LLMProviderConfig configuration. If None, uses environment defaults.
            extra_headers: Optional headers forwarded on every LLM/embedding API request
                (e.g. gateway run-context headers for log correlation).
        """
        self.config = config
        self._extra_headers: Dict[str, str] = dict(extra_headers or {})
        self._llm = None
        self._embeddings = None
        self._agent_llm = None

    @property
    def llm(self):
        """Get or create RAGAS LLM wrapper (cached)."""
        if self._llm is None:
            self._llm = create_ragas_llm(self.config, extra_headers=self._extra_headers or None)
        return self._llm

    @property
    def embeddings(self):
        """Get or create RAGAS embeddings wrapper (cached)."""
        if self._embeddings is None:
            self._embeddings = create_ragas_embeddings(
                self.config, extra_headers=self._extra_headers or None
            )
        return self._embeddings

    @property
    def agent_llm(self):
        """Structured LLM used by agent metrics."""
        if self._agent_llm is None:
            self._agent_llm = create_ragas_instructor_llm(
                self.config, extra_headers=self._extra_headers or None
            )
        return self._agent_llm

    def transform_sample(self, sample: Sample | dict[str, str | Sequence[Any]]) -> SingleTurnSample:
        """Convert a FloEval sample into `SingleTurnSample`."""
        if isinstance(sample, dict):
            # Validate incoming dicts via Pydantic so downstream conversion uses a typed model.
            sample = Sample.model_validate(sample)
        elif not isinstance(sample, Sample):
            raise ValueError(f"Unsupported sample type: {type(sample)}. Expected Sample or dict.")

        user_input = sample.user_input
        contexts = sample.contexts or []
        llm_response = sample.llm_response
        ground_truth = sample.ground_truth or ""

        return SingleTurnSample(
            user_input=user_input,
            retrieved_contexts=contexts if isinstance(contexts, list) else [contexts],
            response=llm_response,
            reference=ground_truth,
        )
