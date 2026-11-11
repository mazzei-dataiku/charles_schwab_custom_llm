# This file is the implementation of custom LLM gemma-4
from dataiku.llm.python import BaseLLM

from vertexai_shared import auth as vertex_auth
from vertexai_shared import client as vertex_client
from vertexai_shared import responses as vertex_responses
from vertexai_shared.streaming import stream_single_chunk, stream_tool_calls
from vertexai_shared.tools import dataiku_tools_to_openai


class MyLLM(BaseLLM):
    def __init__(self):
        return

    def set_config(self, config: dict, plugin_config: dict) -> None:
        # Validate shared plugin settings
        for required_setting in ["project_id", "location"]:
            if required_setting not in plugin_config or not plugin_config[required_setting]:
                raise TypeError(f"Missing plugin setting: {required_setting}")

        # Validate model-specific settings
        if not config or not config.get("endpoint_id"):
            raise TypeError("Missing model setting: endpoint_id")

        endpoint_type = (config.get("endpoint_type") or "PRIVATE_PSC").strip().upper()
        if endpoint_type not in {"PRIVATE_PSC", "DEDICATED"}:
            raise TypeError(
                "Invalid model setting: endpoint_type must be one of PRIVATE_PSC or DEDICATED"
            )

        if endpoint_type == "PRIVATE_PSC" and not plugin_config.get("psc_address"):
            raise TypeError(
                "Missing plugin setting: psc_address is required for endpoint_type=PRIVATE_PSC"
            )

        model_variant = (config.get("model_variant") or "INSTRUCTION_TUNED").strip().upper()
        if model_variant not in {"BASE", "INSTRUCTION_TUNED"}:
            raise TypeError(
                "Invalid model setting: model_variant must be one of BASE or INSTRUCTION_TUNED"
            )

        credentials = vertex_auth.build_credentials(plugin_config)

        def _get_with_default(key: str, default_value):
            value = config.get(key, default_value)
            return default_value if value is None else value

        # Conservative defaults for Gemma 4 initial testing
        max_tokens = _get_with_default("max_tokens", 128)
        temperature = _get_with_default("temperature", 0.2)
        topp = _get_with_default("topp", 0.95)

        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise TypeError("Invalid model setting: max_tokens must be a positive integer")
        if not isinstance(temperature, (int, float)) or temperature < 0:
            raise TypeError("Invalid model setting: temperature must be >= 0")
        if not isinstance(topp, (int, float)) or not (0 < topp <= 1):
            raise TypeError("Invalid model setting: topp must be in (0, 1]")

        self._plugin_config = plugin_config
        self._config = dict(config)
        self._config["endpoint_type"] = endpoint_type
        self._config["model_variant"] = model_variant
        self._config["max_tokens"] = int(max_tokens)
        self._config["temperature"] = float(temperature)
        self._config["topp"] = float(topp)

        vertex_client.init_aiplatform(
            project_id=self._plugin_config["project_id"],
            location=self._plugin_config["location"],
            credentials=credentials,
        )
        self._endpoint = vertex_client.build_endpoint(
            endpoint_id=self._config["endpoint_id"],
            endpoint_type=self._config["endpoint_type"],
        )

    def build_prompt(self, query) -> str:
        msgs = query.get("messages") or []
        parts = []
        for m in msgs:
            content = m.get("content", "")
            parts.append(content if isinstance(content, str) else str(content))
        return "\n\n".join([p.strip() for p in parts if p is not None and str(p).strip()])

    def _validate_role(self, role: str) -> str:
        role = (role or "").strip()
        if role not in {"system", "user", "assistant"}:
            raise TypeError(f"Unsupported message role: {role}")
        return role

    def build_base_instances(self, prompt: str):
        return [{"prompt": prompt, "max_tokens": self._config["max_tokens"]}]

    def build_chat_instances(self, query):
        messages = []
        for message in query.get("messages") or []:
            role = self._validate_role(message.get("role"))
            content = message.get("content")
            if not isinstance(content, str):
                raise TypeError("Only text message content is supported")
            messages.append(
                {
                    "role": role,
                    "content": [{"type": "text", "text": content}],
                }
            )

        return [
            {
                "@requestFormat": "chatCompletions",
                "messages": messages,
                "max_tokens": self._config["max_tokens"],
            }
        ]

    def build_chat_instances_with_tools(self, query, settings):
        instances = self.build_chat_instances(query)
        tools = settings.get("tools") if isinstance(settings, dict) else None
        if tools is not None:
            translated_tools = dataiku_tools_to_openai(tools)
            if translated_tools:
                instances[0]["tools"] = translated_tools
        return instances

    def build_messages_with_tool_outputs(self, query):
        """Return query messages as Vertex chat-completions messages, including tool outputs."""

        messages = []
        for message in query.get("messages") or []:
            role = message.get("role")
            if role in {"system", "user", "assistant"}:
                content = message.get("content")
                if content is None:
                    content = ""
                if not isinstance(content, str):
                    raise TypeError("Only text message content is supported")
                messages.append(
                    {
                        "role": role,
                        "content": [{"type": "text", "text": content}],
                    }
                )
            elif role == "tool":
                tool_outputs = message.get("toolOutputs") or []
                if not isinstance(tool_outputs, list):
                    raise TypeError("Invalid toolOutputs: must be a list")
                for tool_output in tool_outputs:
                    call_id = tool_output.get("callId")
                    output = tool_output.get("output")
                    if not call_id or not isinstance(call_id, str):
                        raise TypeError("Invalid toolOutputs: missing callId")
                    if output is None:
                        output = ""
                    if not isinstance(output, str):
                        raise TypeError("Invalid toolOutputs: output must be a string")
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": [{"type": "text", "text": output}],
                        }
                    )
            else:
                raise TypeError(f"Unsupported message role: {role}")

        return messages

    def build_chat_instances_from_messages(self, messages, settings):
        instances = [
            {
                "@requestFormat": "chatCompletions",
                "messages": messages,
                "max_tokens": self._config["max_tokens"],
            }
        ]
        tools = settings.get("tools") if isinstance(settings, dict) else None
        if tools is not None:
            translated_tools = dataiku_tools_to_openai(tools)
            if translated_tools:
                instances[0]["tools"] = translated_tools
        return instances

    def build_instances(self, query, prompt: str):
        variant = self._config["model_variant"]
        if variant == "BASE":
            return self.build_base_instances(prompt)
        if variant == "INSTRUCTION_TUNED":
            return self.build_chat_instances(query)
        raise TypeError(f"Unsupported model variant: {variant}")

    def process(self, query, settings, trace):
        prompt = self.build_prompt(query)
        if self._config["model_variant"] == "INSTRUCTION_TUNED":
            # Preserve existing request format and augment with tools when provided.
            # Also support passing tool outputs back to the model.
            if any((m.get("role") == "tool") for m in (query.get("messages") or [])):
                messages = self.build_messages_with_tool_outputs(query)
                instances = self.build_chat_instances_from_messages(messages, settings)
            else:
                instances = self.build_chat_instances_with_tools(query, settings)
        else:
            instances = self.build_instances(query, prompt)

        response = vertex_client.predict(
            endpoint=self._endpoint,
            endpoint_type=self._config["endpoint_type"],
            instances=instances,
            psc_address=self._plugin_config.get("psc_address"),
        )

        answer = vertex_responses.extract_answer(response.predictions)
        return {"text": answer, "prompt": prompt}

    def process_stream(self, query, settings, trace):
        # Streaming-first tool calling: if the model returns tool calls,
        # return them as streamed content chunks for Dataiku agents.
        prompt = self.build_prompt(query)

        if self._config["model_variant"] != "INSTRUCTION_TUNED":
            result = self.process(query, settings, trace)
            yield from stream_single_chunk(text=result["text"], footer={})
            return

        if any((m.get("role") == "tool") for m in (query.get("messages") or [])):
            messages = self.build_messages_with_tool_outputs(query)
            instances = self.build_chat_instances_from_messages(messages, settings)
        else:
            instances = self.build_chat_instances_with_tools(query, settings)

        response = vertex_client.predict(
            endpoint=self._endpoint,
            endpoint_type=self._config["endpoint_type"],
            instances=instances,
            psc_address=self._plugin_config.get("psc_address"),
        )

        tool_calls = vertex_responses.extract_tool_call_chunks(response.predictions)
        if tool_calls:
            yield from stream_tool_calls(tool_calls=tool_calls, footer={})
            return

        answer = vertex_responses.extract_answer(response.predictions)
        yield from stream_single_chunk(text=answer, footer={})
