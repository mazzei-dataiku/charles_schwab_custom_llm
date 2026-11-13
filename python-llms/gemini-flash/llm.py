# This file is the implementation of custom LLM gemini-flash

import logging

from dataiku.llm.python import BaseLLM

from vertexai_shared import auth as vertex_auth
from vertexai_shared import gemini as vertex_gemini
from vertexai_shared.streaming import stream_single_chunk, stream_tool_calls


class MyLLM(BaseLLM):
    def __init__(self):
        return

    def set_config(self, config: dict, plugin_config: dict) -> None:
        # Validate shared plugin settings
        for required_setting in ["project_id", "location"]:
            if required_setting not in plugin_config or not plugin_config[required_setting]:
                raise TypeError(f"Missing plugin setting: {required_setting}")

        def _get_with_default(key: str, default_value):
            value = config.get(key, default_value) if isinstance(config, dict) else default_value
            return default_value if value is None else value

        max_output_tokens = _get_with_default("max_output_tokens", 1024)
        temperature = _get_with_default("temperature", 0.2)
        thought_signature_compat_mode = _get_with_default(
            "thought_signature_compat_mode", False
        )

        if not isinstance(max_output_tokens, int) or max_output_tokens <= 0:
            raise TypeError(
                "Invalid model setting: max_output_tokens must be a positive integer"
            )
        if not isinstance(temperature, (int, float)) or temperature < 0:
            raise TypeError("Invalid model setting: temperature must be >= 0")
        if not isinstance(thought_signature_compat_mode, bool):
            raise TypeError(
                "Invalid model setting: thought_signature_compat_mode must be a boolean"
            )

        # Gemini publisher models should default to global routing.
        # Precedence: model setting `location` (default global) overrides plugin `location`.
        gemini_location = _get_with_default("location", "global")
        if not isinstance(gemini_location, str) or not gemini_location.strip():
            raise TypeError("Invalid model setting: location must be a non-empty string")
        gemini_location = gemini_location.strip()

        self._plugin_config = dict(plugin_config)
        self._plugin_config["location"] = gemini_location
        self._config = dict(config or {})
        self._config["max_output_tokens"] = int(max_output_tokens)
        self._config["temperature"] = float(temperature)
        self._config["model_id"] = (self._config.get("model_id") or "gemini-3.8-flash").strip()
        self._config["thought_signature_compat_mode"] = bool(
            thought_signature_compat_mode
        )

        self._credentials = vertex_auth.build_credentials(self._plugin_config)

    def _build_payload(self, query, settings, *, include_tool_config: bool):
        messages = query.get("messages") or []

        payload = vertex_gemini.dataiku_messages_to_gemini(
            messages=messages,
            thought_signature_compat_mode=self._config.get(
                "thought_signature_compat_mode", False
            ),
        )

        tools = settings.get("tools") if isinstance(settings, dict) else None
        if include_tool_config and tools is not None:
            tool_config = vertex_gemini.dataiku_tools_to_gemini(tools=tools)
            if tool_config:
                payload["tools"] = tool_config

        tool_choice = settings.get("toolChoice") if isinstance(settings, dict) else None
        if include_tool_config and tool_choice is not None:
            gemini_tool_config = vertex_gemini.dataiku_tool_choice_to_gemini(
                tool_choice=tool_choice,
                declared_tools=tools,
            )
            if gemini_tool_config:
                payload["toolConfig"] = gemini_tool_config

        payload["generationConfig"] = vertex_gemini.build_generation_config(
            max_output_tokens=self._config["max_output_tokens"],
            temperature=self._config["temperature"],
        )

        return payload

    def _invoke(self, payload: dict):
        # Track invocation progress for safe diagnostics.
        self._last_invoke_reached = True
        self._last_invoke_completed = False
        url = vertex_gemini.build_generate_content_url(
            project_id=self._plugin_config["project_id"],
            location=self._plugin_config["location"],
            model_id=self._config["model_id"],
        )
        bearer = vertex_gemini.get_bearer_token(credentials=self._credentials)
        out = vertex_gemini.invoke_generate_content(url=url, bearer_token=bearer, payload=payload)
        self._last_invoke_completed = True
        return out

    def process(self, query, settings, trace):
        # Non-streaming behavior: return text only.
        # Tool calling via non-streaming is not claimed; if the model returns function
        # calls, we still expose them in the returned dict under toolCalls for
        # best-effort compatibility.
        payload = self._build_payload(query, settings, include_tool_config=True)
        response_json = self._invoke(payload)

        text, tool_calls = vertex_gemini.parse_or_raise_empty(response_json)
        usage = vertex_gemini.extract_usage_metadata(response_json)
        finish_reason = vertex_gemini.extract_finish_reason(
            response_json, has_tool_calls=bool(tool_calls)
        )
        return {
            "text": text or "",
            "toolCalls": tool_calls,
            "promptTokens": usage["promptTokens"],
            "completionTokens": usage["completionTokens"],
            "finishReason": finish_reason,
        }

    def process_stream(self, query, settings, trace):
        logger = logging.getLogger(__name__)

        messages = query.get("messages") or []
        tools = settings.get("tools") if isinstance(settings, dict) else None
        tool_count = len(tools) if isinstance(tools, list) else 0

        logger.info(
            "GeminiFlash.process_stream: enter messages=%s tools=%s",
            len(messages),
            tool_count,
        )
        # Streaming-first tool calling: if the model returns tool calls,
        # emit them as a dedicated Dataiku toolCalls chunk.
        try:
            payload = self._build_payload(query, settings, include_tool_config=True)
            logger.info("GeminiFlash.process_stream: payload_built")
        except Exception:
            logger.exception("GeminiFlash.process_stream: payload_build_failed")
            raise

        # Second-iteration tool execution is handled by dataiku_messages_to_gemini(),
        # which now converts assistant toolCalls and tool toolOutputs into native
        # Gemini functionCall/functionResponse parts.

        try:
            logger.info("GeminiFlash.process_stream: invoking_vertex")
            response_json = self._invoke(payload)
            logger.info("GeminiFlash.process_stream: vertex_completed")

            diag = vertex_gemini.build_empty_response_diagnostics(response_json)
            logger.info(
                "GeminiFlash.process_stream: vertex_diag http_status=%s candidates=%s finish_reasons=%s block_reason=%s text_parts=%s function_call_parts=%s",
                diag.get("http_status"),
                diag.get("candidates"),
                diag.get("finish_reasons"),
                diag.get("prompt_feedback_block_reason"),
                diag.get("text_parts"),
                diag.get("function_call_parts"),
            )

            text, tool_calls = vertex_gemini.parse_or_raise_empty(response_json)
            logger.info(
                "GeminiFlash.process_stream: parsed text_len=%s tool_calls=%s",
                len(text or ""),
                len(tool_calls or []),
            )
        except Exception:
            logger.exception("GeminiFlash.process_stream: vertex_or_parse_failed")
            raise

        if tool_calls:
            usage = vertex_gemini.extract_usage_metadata(response_json)
            finish_reason = vertex_gemini.extract_finish_reason(
                response_json, has_tool_calls=True
            )
            footer = {
                "promptTokens": usage["promptTokens"],
                "completionTokens": usage["completionTokens"],
                "finishReason": finish_reason,
            }
            yield from stream_tool_calls(tool_calls=tool_calls, footer=footer)
            return

        usage = vertex_gemini.extract_usage_metadata(response_json)
        finish_reason = vertex_gemini.extract_finish_reason(
            response_json, has_tool_calls=False
        )
        footer = {
            "promptTokens": usage["promptTokens"],
            "completionTokens": usage["completionTokens"],
            "finishReason": finish_reason,
        }
        yield from stream_single_chunk(text=text or "", footer=footer)
