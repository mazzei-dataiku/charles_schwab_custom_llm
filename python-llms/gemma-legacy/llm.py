# This file is the implementation of custom LLM gemma-legacy
from dataiku.llm.python import BaseLLM

from vertexai_shared import auth as vertex_auth
from vertexai_shared import client as vertex_client
from vertexai_shared import responses as vertex_responses


class MyLLM(BaseLLM):
    # ----------------------------------------------------------------------------------------------------
    # init
    # ----------------------------------------------------------------------------------------------------
    def __init__(self):
        return

    # ----------------------------------------------------------------------------------------------------
    # set_config
    # Save configuration settings and initialize
    # ----------------------------------------------------------------------------------------------------
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

        credentials = vertex_auth.build_credentials(plugin_config)

        def _get_with_default(key: str, default_value):
            value = config.get(key, default_value)
            return default_value if value is None else value

        max_tokens = _get_with_default("max_tokens", 1024)
        temperature = _get_with_default("temperature", 0.2)
        topp = _get_with_default("topp", 0.95)

        # Range/type validation (keep minimal and explicit)
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise TypeError("Invalid model setting: max_tokens must be a positive integer")
        if not isinstance(temperature, (int, float)) or temperature < 0:
            raise TypeError("Invalid model setting: temperature must be >= 0")
        if not isinstance(topp, (int, float)) or not (0 < topp <= 1):
            raise TypeError("Invalid model setting: topp must be in (0, 1]")

        # Save settings
        self._plugin_config = plugin_config
        self._config = dict(config)
        self._config["endpoint_type"] = endpoint_type
        self._config["max_tokens"] = int(max_tokens)
        self._config["temperature"] = float(temperature)
        self._config["topp"] = float(topp)

        # Initialize AI platform
        vertex_client.init_aiplatform(
            project_id=self._plugin_config["project_id"],
            location=self._plugin_config["location"],
            credentials=credentials,
        )

        # Initialize endpoint (model-specific)
        self._endpoint = vertex_client.build_endpoint(
            endpoint_id=self._config["endpoint_id"],
            endpoint_type=self._config["endpoint_type"],
        )

    # ----------------------------------------------------------------------------------------------------
    # build_gemma_prompt
    # Gemma has no system role, so the system prompt is prepended to the user turn
    # ----------------------------------------------------------------------------------------------------
    def build_gemma_prompt(self, system_prompt, user_prompt):
        prefix = f"{system_prompt.strip()}\n\n" if system_prompt else ""
        return (
            f"<start_of_turn>user\n{prefix}{user_prompt.strip()}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )

    # ----------------------------------------------------------------------------------------------------
    # process
    # Process a single message
    # ----------------------------------------------------------------------------------------------------
    # define aprocess instead of process for async support
    def process(self, query, settings, trace):
        prompt = self.build_gemma_prompt("", query["messages"][0]["content"])

        instances = [
            {
                "prompt": prompt,
                "max_tokens": self._config["max_tokens"],
                "temperature": self._config["temperature"],
                "top_p": self._config["topp"],
                "stop": ["<end_of_turn>"],
                "raw_response": True,
            }
        ]

        response = vertex_client.predict(
            endpoint=self._endpoint,
            endpoint_type=self._config["endpoint_type"],
            instances=instances,
            psc_address=self._plugin_config.get("psc_address"),
        )

        answer = vertex_responses.extract_answer(response.predictions)
        return {"text": answer, "prompt": prompt}

    # ----------------------------------------------------------------------------------------------------
    # process_stream
    # define aprocess_stream instead of process_stream for async support
    # ----------------------------------------------------------------------------------------------------
    def process_stream(self, query, settings, trace):
        messages = query["messages"]

        # Add code here, for LLM completion, tool execution, etc...

        yield {
            "chunk": {
                "text": "The ",
            }
        }
        yield {
            "chunk": {
                "text": "answer.",
            }
        }
        yield {
            "footer": {
                "promptTokens": 46,  # Optional
                "completionTokens": 10,  # Optional
                "estimatedCost": 0.13,  # Optional
                "toolCalls": [],
            }
        }

