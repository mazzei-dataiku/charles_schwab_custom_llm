# This file is the implementation of custom LLM gemma-4
import json

from google.cloud import aiplatform
from google.oauth2 import service_account
from dataiku.llm.python import BaseLLM


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

        auth_method = (plugin_config.get("auth_method") or "ADC").strip().upper()
        if auth_method not in {"ADC", "SERVICE_ACCOUNT"}:
            raise TypeError(
                "Invalid plugin setting: auth_method must be one of ADC or SERVICE_ACCOUNT"
            )

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

        credentials = None
        if auth_method == "SERVICE_ACCOUNT":
            raw_service_account = plugin_config.get("service_account_json")
            if not raw_service_account:
                raise TypeError(
                    "Missing plugin setting: service_account_json is required for auth_method=SERVICE_ACCOUNT"
                )

            if isinstance(raw_service_account, dict):
                service_account_info = raw_service_account
            elif isinstance(raw_service_account, str):
                try:
                    service_account_info = json.loads(raw_service_account)
                except json.JSONDecodeError as exc:
                    raise TypeError(
                        "Invalid plugin setting: service_account_json must be valid JSON"
                    ) from exc
            else:
                raise TypeError(
                    "Invalid plugin setting: service_account_json must be a JSON string"
                )

            if not isinstance(service_account_info, dict):
                raise TypeError(
                    "Invalid plugin setting: service_account_json must be a JSON object"
                )

            if not service_account_info.get("client_email") or not service_account_info.get("private_key"):
                raise TypeError(
                    "Invalid plugin setting: service_account_json is missing required fields"
                )

            credentials = service_account.Credentials.from_service_account_info(
                service_account_info,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )

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
        if credentials is None:
            aiplatform.init(
                project=self._plugin_config["project_id"],
                location=self._plugin_config["location"],
            )
        else:
            aiplatform.init(
                project=self._plugin_config["project_id"],
                location=self._plugin_config["location"],
                credentials=credentials,
            )

        # Initialize endpoint (model-specific)
        if self._config["endpoint_type"] == "PRIVATE_PSC":
            self._endpoint = aiplatform.PrivateEndpoint(endpoint_name=self._config["endpoint_id"])
        else:
            self._endpoint = aiplatform.Endpoint(endpoint_name=self._config["endpoint_id"])

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
    # extract_text
    # Extract text from prediction
    # ----------------------------------------------------------------------------------------------------
    def extract_text(self, prediction):
        # vLLM chatCompletions responses follow the OpenAI schema
        if isinstance(prediction, dict) and "choices" in prediction:
            return prediction["choices"][0]["message"]["content"]
        if isinstance(prediction, str):
            return prediction
        return json.dumps(prediction, indent=2, default=str)    
    
    # ----------------------------------------------------------------------------------------------------
    # process
    # Process a single message
    # ----------------------------------------------------------------------------------------------------
    # define aprocess instead of process for async support
    def process(self, query, settings, trace):
        # Extract message from query and build prompt
        messages = query["messages"][0]["content"]
        prompt = self.build_gemma_prompt("", query["messages"][0]["content"])

        # Build instances payload
        instances = [
            {
                "prompt": prompt,
                "max_tokens": self._config["max_tokens"],
                "temperature": self._config["temperature"],
                "top_p": self._config["topp"],
                "stop": ["<end_of_turn>"],
                "raw_response": True # Return only the generated text, not "Prompt: ... Output: ..."
            }
        ]
        
        # Query model for prediction
        if self._config["endpoint_type"] == "PRIVATE_PSC":
            response = self._endpoint.predict(
                instances=instances,
                endpoint_override=self._plugin_config["psc_address"],
            )
        else:
            response = self._endpoint.predict(
                instances=instances,
                use_dedicated_endpoint=True,
            )
        
        # Extract answer from response
        # chatCompletions may return a single object rather than a list
        predictions = response.predictions
        answer = ""
        if isinstance(predictions, dict):
            predictions = [predictions]
        for prediction in predictions:
            answer += self.extract_text(prediction)

        # Return results
        return {
            "text": answer,
            "prompt": prompt
        }

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
