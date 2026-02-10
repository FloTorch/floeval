"""
LLM Helper for custom metrics.

Provides simple interface for LLM calls in custom metrics.
Supports both sync (generate) and async (agenerate) usage.
Sync API runs async code in an isolated thread via ThreadPoolExecutor (production-safe).
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import openai

from floeval.config import GatewayConfig
from floeval.utils.gateway import normalize_openai_api_base

logger = logging.getLogger(__name__)



class SimpleLLMHelper:
    """
    Simple LLM wrapper for custom metrics.
    
    Supports both sync and async usage transparently.
    - generate(): Sync API; runs async call in isolated thread (ThreadPoolExecutor).
      User can write sync metrics and call llm.generate() - it just works.
    - agenerate(): Async API for async metrics or Evaluation.arun().
    
    Each thread has its own event loop (complete isolation, no main-thread event loop creation).
    """
    
    def __init__(self, gateway_config: GatewayConfig | None = None, llm_model: str | None = None):
        """
        Initialize LLM helper from gateway configuration.
        
        Client initialization is lazy (deferred until first use) to allow
        gateway_config to be injected later by Evaluation.
        
        Args:
            gateway_config: Gateway configuration containing API endpoint and credentials.
                           Required for actual LLM calls. Can be None initially, but must be
                           set before calling generate() or agenerate().
            llm_model: LLM model identifier (e.g., "gpt-4", "flotorch/openai-gpt-4").
                       Optional; uses gateway_config.llm_model if available, otherwise defaults
                       to "gpt-4". Overridden by gateway_config.llm_model when gateway_config
                       is provided.
        
        Raises:
            ValueError: If gateway_config is None when generate() or agenerate() is called.
        """
        self.gateway_config = gateway_config
        self._llm_model_param = llm_model
        
        # Set model
        self.model = llm_model or (gateway_config.llm_model if gateway_config else "gpt-4")
        
        # Initialize client lazily (only when needed)
        self.client = None
        
        # ThreadPoolExecutor for sync generate(): one worker, own event loop per call (isolation)
        self._executor = ThreadPoolExecutor(max_workers=1)
    
    def _init_client(self):
        """
        Initialize OpenAI client from gateway config (lazy initialization).
        
        Called automatically on first use. Ensures gateway_config is available
        and creates AsyncOpenAI client with normalized gateway URL.
        
        Raises:
            ValueError: If gateway_config is None or missing required fields.
        """
        if self.client is not None:
            return  # Already initialized

        if not self.gateway_config:
            raise ValueError(
                "gateway_config is required for LLM-based custom metrics. "
                "Pass gateway_config to Evaluation when creating the evaluation instance."
            )
        
        # Update model from gateway_config if available
        if self.gateway_config.llm_model:
            self.model = self.gateway_config.llm_model
        elif self._llm_model_param:
            self.model = self._llm_model_param

        base_url = normalize_openai_api_base(self.gateway_config.gateway_base_url)
        
        self.client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=self.gateway_config.api_key
        )
        
        logger.debug(f"Initialized LLM client for model: {self.model}, base_url: {base_url}")
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None
    ) -> str:
        """
        Generate text from prompt (sync interface).
        
        Works in sync context by running the async call in an isolated thread
        via ThreadPoolExecutor. Each thread has its own event loop.
        User can write sync custom metrics and call llm.generate() - it just works.
        
        Args:
            prompt: Question or instruction to send to LLM.
            temperature: Randomness control (0.0 = deterministic, 1.0 = creative).
            max_tokens: Maximum response length. None uses model default.
        
        Returns:
            str: Raw text response from LLM. User is responsible for parsing.
        
        Raises:
            ValueError: If gateway_config is not set.
            TimeoutError: If gateway times out (504 error).
            RuntimeError: If gateway returns server error (500 error).
        
        Example:
            @custom_metric
            def helpfulness(response: str, llm) -> float:
                answer = llm.generate(f"Rate helpfulness 0-1: {response}")
                return float(answer.strip())
        """
        future = self._executor.submit(
            self._run_async_in_thread,
            prompt,
            temperature,
            max_tokens,
        )
        return future.result()
    
    def _run_async_in_thread(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """
        Run async generation in a new thread with its own event loop.
        
        bridge for sync -> async. ThreadPoolExecutor provides
        complete isolation; no main-thread event loop creation.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # Create client in this thread so it is bound to this loop (avoids "Event loop is closed" on cleanup)
        self.client = None
        try:
            self._init_client()
            return loop.run_until_complete(
                self.agenerate(prompt, temperature=temperature, max_tokens=max_tokens)
            )
        finally:
            if self.client is not None:
                try:
                    # Run async client cleanup while the loop is still open.
                    # Avoids "Event loop is closed" when httpx AsyncClient runs aclose().
                    loop.run_until_complete(self.client.close())
                except Exception as e:
                    logger.debug("LLM client close: %s", e)
            loop.close()
            self.client = None
    
    async def agenerate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None
    ) -> str:
        """
        Generate text from prompt (asynchronous interface).
        
        This is the only allowed way to call LLM from custom metrics.
        Use this method inside async custom metric functions or when Evaluation.arun() is available.
        
        Args:
            prompt: Question or instruction to send to LLM. Should be clear and
                   specific about desired output format (e.g., "Return only a number 0-1").
            temperature: Randomness control (0.0 = deterministic, 1.0 = creative).
                       Default 0.0 for consistent scoring/metrics.
            max_tokens: Maximum response length. None uses model default.
                       Lower values reduce latency and cost.
        
        Returns:
            str: Raw text response from LLM. User is responsible for parsing
                 (e.g., float(answer.strip()) for scores, json.loads() for JSON).
        
        Raises:
            ValueError: If gateway_config is not set.
            TimeoutError: If gateway times out (504 error).
            RuntimeError: If gateway returns server error (500 error).
        
        Examples:
            # In async custom metric:
            @custom_metric
            async def helpfulness(response: str, llm) -> float:
                answer = await llm.agenerate(
                    f"Rate helpfulness 0-1: {response}"
                )
                return float(answer.strip())
            
            # Yes/no question
            @custom_metric
            async def is_professional(response: str, llm) -> float:
                answer = await llm.agenerate(
                    f"Is this professional? yes/no: {response}"
                )
                return 1.0 if "yes" in answer.lower() else 0.0
        """
        # Ensure client is initialized (lazy initialization)
        self._init_client()
        
        try:
            # Build request parameters
            request_params = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature
            }
            # Only include max_tokens if specified
            if max_tokens is not None:
                request_params["max_tokens"] = max_tokens
            
            logger.debug(f"Calling LLM with model={self.model}, temperature={temperature}, max_tokens={max_tokens}")
            response = await self.client.chat.completions.create(**request_params)
            
            content = response.choices[0].message.content
            logger.debug(f"LLM response received, length={len(content)}")
            return content
            
        except Exception as e:
            # Provide better error messages for common issues
            error_code = getattr(e, 'status_code', None) or getattr(e, 'code', None)
            error_msg = str(e)
            
            logger.error(f"LLM call failed: {error_msg}", exc_info=True)
            
            if error_code == 504 or '504' in error_msg or 'timeout' in error_msg.lower():
                raise TimeoutError(
                    f"Gateway timeout (504) while calling LLM. "
                    f"This usually means the request took too long. "
                    f"Try reducing max_tokens or simplifying the prompt. "
                    f"Original error: {error_msg}"
                ) from e
            elif error_code == 500 or '500' in error_msg:
                raise RuntimeError(
                    f"Gateway server error (500) while calling LLM. "
                    f"This may be a temporary issue. Please try again later. "
                    f"Original error: {error_msg}"
                ) from e
            else:
                # Re-raise other errors as-is
                raise
