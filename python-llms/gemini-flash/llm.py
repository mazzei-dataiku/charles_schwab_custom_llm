# This file is the implementation of custom LLM gemini-flash

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

        if not isinstance(max_output_tokens, int) or max_output_tokens <= 0:
            raise TypeError(
                "Invalid model setting: max_output_tokens must be a positive integer"
            )
        if not isinstance(temperature, (int, float)) or temperature < 0:
            raise TypeError("Invalid model setting: temperature must be >= 0")

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

        self._credentials = vertex_auth.build_credentials(self._plugin_config)

    def _build_payload(self, query, settings, *, include_tool_config: bool):
        messages = query.get("messages") or []

        payload = vertex_gemini.dataiku_messages_to_gemini(messages=messages)

        tools = settings.get("tools") if isinstance(settings, dict) else None
        if include_tool_config and tools is not None:
            tool_config = vertex_gemini.dataiku_tools_to_gemini(tools=tools)
            if tool_config:
                payload["tools"] = tool_config

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
        return {"text": text or "", "toolCalls": tool_calls}

    def process_stream(self, query, settings, trace):
        # Streaming-first tool calling: if the model returns tool calls,
        # emit them as a dedicated Dataiku toolCalls chunk.
        payload = self._build_payload(query, settings, include_tool_config=True)

        # If the incoming query contains tool outputs, include them as a functionResponse
        # content. Note: native function-call roundtrip semantics are not fully verified.
        if any((m.get("role") == "tool") for m in (query.get("messages") or [])):
            # Invoke once (without tool outputs) to discover tool call IDs/names if needed.
            # This keeps behavior explicit rather than guessing call association.
            first_response = self._invoke(payload)
            _, tool_calls = vertex_gemini.parse_or_raise_empty(first_response)
            if tool_calls:
                # Surface tool calls and stop; agent will call again with toolOutputs.
                yield from stream_tool_calls(tool_calls=tool_calls, footer={})
                return

            # No tool calls; proceed by adding tool outputs to the payload.
            tool_contents = vertex_gemini.dataiku_tool_outputs_to_gemini(
                query_messages=query.get("messages") or [], prior_tool_calls=[]
            )
            if tool_contents:
                payload["contents"].extend(tool_contents)

        response_json = self._invoke(payload)
        text, tool_calls = vertex_gemini.parse_or_raise_empty(response_json)

        if tool_calls:
            yield from stream_tool_calls(tool_calls=tool_calls, footer={})
            return

        yield from stream_single_chunk(text=text or "", footer={})
