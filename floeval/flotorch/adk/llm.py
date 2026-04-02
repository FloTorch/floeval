"""ADK-compatible LLM wrapper using FloTorch SDK LLM."""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Dict, List, Type

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import BaseModel

from floeval.flotorch.adk.utils.adk_utils import (
    build_messages_from_request,
    parse_llm_response_with_tools,
    tools_to_openai_format,
)
from floeval.flotorch.sdk.llm import FlotorchLLM

logger = logging.getLogger(__name__)


def _convert_pydantic_to_json_schema(model_class: Type[BaseModel]) -> Dict[str, Any]:
    """Convert Pydantic model to JSON schema for response_format."""
    schema_dict = model_class.model_json_schema()
    return {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "schema": {**schema_dict, "additionalProperties": False},
                "name": schema_dict.get("title", model_class.__name__),
                "strict": True,
            },
        },
    }


class FlotorchADKLLM(BaseLlm):
    """ADK-compatible LLM using FloTorch HTTP chat completions."""

    def __init__(
        self,
        model_id: str,
        api_key: str,
        base_url: str,
        chat_endpoint: str = "chat/completions",
        default_headers: Dict[str, str] | None = None,
    ):
        super().__init__(model=model_id, api_key=api_key, base_url=base_url)
        self._llm = FlotorchLLM(
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
            chat_endpoint=chat_endpoint,
            default_headers=default_headers,
        )

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        """Generate content from LLM request."""
        messages = build_messages_from_request(llm_request)
        tools = None
        if hasattr(llm_request, "tools_dict") and llm_request.tools_dict:
            tools = tools_to_openai_format(llm_request.tools_dict.values())

        response_format = None
        if hasattr(llm_request, "config") and getattr(llm_request.config, "response_schema", None):
            response_format = _convert_pydantic_to_json_schema(
                llm_request.config.response_schema
            ).get("response_format")

        try:
            response = await self._llm.ainvoke(
                messages=messages,
                tools=tools,
                response_format=response_format,
                extra_body={},
            )
        except Exception as llm_error:
            logger.error("FlotorchADKLLM.generate_content_async failed: %s", llm_error)
            yield LlmResponse(
                content=types.Content(
                    role="assistant",
                    parts=[
                        types.Part(
                            text="I'm experiencing some technical difficulties. Please try again."
                        )
                    ],
                )
            )
            return

        response_data = response.metadata.get("raw_response", {})
        parsed_parts = parse_llm_response_with_tools(response_data)

        if parsed_parts:
            parts: List[types.Part] = []
            function_calls_found = False
            for part_data in parsed_parts:
                if part_data.get("type") == "function_call":
                    if not function_calls_found:
                        try:
                            part = types.Part.from_function_call(
                                name=part_data["name"], args=part_data["args"]
                            )
                            if part.function_call and "id" in part_data:
                                part.function_call.id = part_data["id"]
                            parts.append(part)
                            function_calls_found = True
                        except Exception as e:
                            logger.warning("Failed to create function call part: %s", e)
                            parts.append(
                                types.Part.from_text(
                                    text="I'll help you with that. Let me check..."
                                )
                            )
                            break
                elif part_data.get("type") == "text":
                    parts.append(types.Part.from_text(text=part_data["content"]))

            if parts:
                yield LlmResponse(content=types.Content(role="assistant", parts=parts))
                return

        text_content = (
            response.content
            if hasattr(response, "content") and response.content
            else "I'm here to help you."
        )
        yield LlmResponse(
            content=types.Content(role="assistant", parts=[types.Part(text=text_content)])
        )
