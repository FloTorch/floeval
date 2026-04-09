"""FloTorch gateway LLM behind ADK's BaseLlm (chat/completions + tools)."""

from __future__ import annotations

import logging
import traceback
from typing import Any, AsyncGenerator, Dict, List, Type

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import BaseModel, PrivateAttr

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
    """ADK-compatible LLM wrapper using FloTorch gateway LLM."""

    _llm: FlotorchLLM = PrivateAttr()

    def __init__(
        self,
        model_id: str,
        api_key: str,
        base_url: str,
        chat_endpoint: str = "chat/completions",
    ):
        super().__init__(model=model_id, api_key=api_key, base_url=base_url)
        self._llm = FlotorchLLM(
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
            chat_endpoint=chat_endpoint,
        )

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        messages = build_messages_from_request(llm_request)
        n_tools = len(llm_request.tools_dict) if getattr(llm_request, "tools_dict", None) else 0
        print(
            f"[floeval-debug] FlotorchADKLLM generate_content_async msgs={len(messages)} "
            f"url={self._llm._url} tools={n_tools}",
            flush=True,
        )

        tools = None
        if hasattr(llm_request, "tools_dict") and llm_request.tools_dict:
            tools = tools_to_openai_format(llm_request.tools_dict.values())

        response_format = None
        if hasattr(llm_request, "config") and getattr(llm_request.config, "response_schema", None):
            response = _convert_pydantic_to_json_schema(llm_request.config.response_schema)
            response_format = response["response_format"]

        try:
            response = await self._llm.ainvoke(
                messages=messages,
                tools=tools,
                response_format=response_format,
                extra_body={},
            )

            try:
                response_data = response.metadata.get('raw_response', {})
                parsed_parts = parse_llm_response_with_tools(response_data)

                if parsed_parts:
                    parts = []

                    for part_data in parsed_parts:
                        if part_data["type"] == "function_call":
                            try:
                                part = types.Part.from_function_call(name=part_data["name"], args=part_data["args"])
                                if part.function_call and "id" in part_data:
                                    part.function_call.id = part_data["id"]
                                parts.append(part)
                            except Exception as e:
                                print("[floeval-debug] Failed to build function_call Part:", flush=True)
                                traceback.print_exc()
                                logger.warning("Failed to create function call part for '%s': %s", part_data.get("name"), e)
                                if not parts:
                                    parts.append(types.Part.from_text(text="I'll help you with that. Let me check..."))
                                break
                        elif part_data["type"] == "text":
                            parts.append(types.Part.from_text(text=part_data["content"]))

                    if parts:
                        yield LlmResponse(content=types.Content(role="assistant", parts=parts))
                        return

                text_content = response.content if hasattr(response, 'content') and response.content else "I'm here to help you."
                yield LlmResponse(content=types.Content(role="assistant", parts=[types.Part(text=text_content)]))

            except Exception as parse_error:
                print("[floeval-debug] FlotorchADKLLM parse_response exception:", flush=True)
                traceback.print_exc()
                logger.error("FlotorchADKLLM parse_response failed: %s", parse_error)
                text_content = response.content if hasattr(response, 'content') and response.content else "I apologize, but I encountered an issue. How can I help you?"
                yield LlmResponse(content=types.Content(role="assistant", parts=[types.Part(text=text_content)]))

        except Exception as llm_error:
            error_str = str(llm_error)
            print("[floeval-debug] FlotorchADKLLM ainvoke / outer exception:", flush=True)
            traceback.print_exc()

            # Gemini may 502 on tool replay without thought_signature (gateway drops it).
            # Retry once without tools so the model can answer from tool output in context.
            if "thought_signature" in error_str and tools and messages:
                print("[floeval-debug] FlotorchADKLLM attempting thought_signature recovery retry", flush=True)
                try:
                    recovery_messages = messages + [
                        {"role": "assistant", "content": "I encountered a technical issue. Let me answer based on the information I already retrieved."},
                        {"role": "user", "content": "Please provide the answer based on the information you have already retrieved."},
                    ]
                    recovery_response = await self._llm.ainvoke(
                        messages=recovery_messages,
                        tools=None,
                        response_format=response_format,
                        extra_body={},
                    )
                    recovery_data = recovery_response.metadata.get('raw_response', {})
                    recovery_parts = parse_llm_response_with_tools(recovery_data)
                    if recovery_parts:
                        parts = [
                            types.Part.from_text(text=p["content"])
                            for p in recovery_parts if p["type"] == "text" and p.get("content")
                        ]
                        if parts:
                            yield LlmResponse(content=types.Content(role="assistant", parts=parts))
                            return
                except Exception as retry_err:
                    print("[floeval-debug] thought_signature retry failed:", flush=True)
                    traceback.print_exc()
                    logger.warning("thought_signature retry also failed: %s", retry_err)

            logger.error("FlotorchADKLLM.generate_content_async failed: %s", llm_error)
            error_text = "I'm experiencing some technical difficulties. Please try again."
            yield LlmResponse(content=types.Content(role="assistant", parts=[types.Part(text=error_text)]))
