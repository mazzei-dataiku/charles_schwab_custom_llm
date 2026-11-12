import json
import uuid


class GeminiAPIError(RuntimeError):
    pass


def _google_auth_transport_requests_module():
    from google.auth.transport import requests

    return requests


def _google_auth_module():
    import google.auth

    return google.auth


def _requests_module():
    import requests

    return requests


_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


def vertex_api_host_for_location(location: str) -> str:
    location = (location or "").strip()
    if not location:
        raise TypeError("Missing location")
    if location.lower() == "global":
        return "aiplatform.googleapis.com"
    return f"{location}-aiplatform.googleapis.com"


def build_generate_content_url(*, project_id: str, location: str, model_id: str) -> str:
    if not project_id:
        raise TypeError("Missing project_id")
    if not model_id:
        raise TypeError("Missing model_id")
    host = vertex_api_host_for_location(location)

    # Publisher model resource name.
    # Note: model_id should include the version suffix, e.g. "gemini-3.8-flash".
    return (
        f"https://{host}/v1/projects/{project_id}/locations/{location}"
        f"/publishers/google/models/{model_id}:generateContent"
    )


def get_bearer_token(*, credentials) -> str:
    """Return a valid OAuth2 bearer token for Vertex AI REST calls.

    Supports:
    - Explicit google-auth Credentials passed from vertexai_shared.auth.build_credentials
    - ADC discovery when credentials is None
    """

    if credentials is None:
        google_auth = _google_auth_module()
        credentials, _ = google_auth.default(scopes=[_CLOUD_PLATFORM_SCOPE])

    # Ensure the token is fresh.
    if getattr(credentials, "valid", False) and getattr(credentials, "token", None):
        return credentials.token

    auth_requests = _google_auth_transport_requests_module()
    request = auth_requests.Request()
    credentials.refresh(request)

    token = getattr(credentials, "token", None)
    if not token:
        raise GeminiAPIError("Failed to obtain access token")
    return token


def dataiku_messages_to_gemini(*, messages):
    """Translate Dataiku chat messages into Gemini generateContent payload."""

    if messages is None:
        messages = []
    if not isinstance(messages, list):
        raise TypeError("Invalid messages: must be a list")

    system_chunks = []
    contents = []

    # Track assistant tool calls for later tool output resolution.
    # Dataiku represents tool calls as OpenAI-like objects in assistant messages.
    call_id_to_function_name = {}

    def _parse_tool_calls(message_obj: dict, path: str):
        tool_calls = message_obj.get("toolCalls")
        if tool_calls is None:
            return []
        if not isinstance(tool_calls, list):
            raise TypeError(f"Invalid {path}.toolCalls: must be a list")

        parts = []
        for i, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                raise TypeError(f"Invalid {path}.toolCalls[{i}]: must be an object")
            if tool_call.get("type") != "function":
                raise TypeError(
                    f"Unsupported tool call type at {path}.toolCalls[{i}]: {tool_call.get('type')}"
                )
            call_id = tool_call.get("id")
            fn = tool_call.get("function") or {}
            name = fn.get("name")
            arguments = fn.get("arguments")
            if not call_id or not isinstance(call_id, str):
                raise TypeError(f"Invalid {path}.toolCalls[{i}].id")
            if not name or not isinstance(name, str):
                raise TypeError(f"Invalid {path}.toolCalls[{i}].function.name")
            if arguments is None:
                arguments = "{}"
            if not isinstance(arguments, str):
                raise TypeError(f"Invalid {path}.toolCalls[{i}].function.arguments")

            # Parse JSON string arguments to an object for Gemini functionCall args.
            try:
                args_obj = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as exc:
                raise TypeError(
                    f"Invalid JSON in {path}.toolCalls[{i}].function.arguments"
                ) from exc

            call_id_to_function_name[call_id] = name
            parts.append({"functionCall": {"name": name, "args": args_obj}})

        return parts
    for message in messages:
        if not isinstance(message, dict):
            raise TypeError("Invalid message: must be an object")
        role = (message.get("role") or "").strip()

        if role in {"system", "user", "assistant"}:
            content = message.get("content")
            if content is None:
                content = ""
            if not isinstance(content, str):
                raise TypeError("Only text message content is supported")

        if role == "system":
            if content.strip():
                system_chunks.append(content)
            continue

        if role == "user":
            contents.append({"role": "user", "parts": [{"text": content}]})
            continue

        if role == "assistant":
            tool_call_parts = _parse_tool_calls(message, "messages")
            parts = []
            if content.strip():
                parts.append({"text": content})
            parts.extend(tool_call_parts)
            contents.append({"role": "model", "parts": parts or [{"text": ""}]})
            continue

        if role == "tool":
            tool_outputs = message.get("toolOutputs") or []
            if not isinstance(tool_outputs, list):
                raise TypeError("Invalid toolOutputs: must be a list")

            parts = []
            for i, tool_output in enumerate(tool_outputs):
                if not isinstance(tool_output, dict):
                    raise TypeError(
                        f"Invalid toolOutputs[{i}]: each entry must be an object"
                    )
                call_id = tool_output.get("callId")
                output = tool_output.get("output")
                if not call_id or not isinstance(call_id, str):
                    raise TypeError("Invalid toolOutputs: missing callId")
                if output is None:
                    output = ""
                if not isinstance(output, str):
                    raise TypeError("Invalid toolOutputs: output must be a string")

                fn_name = call_id_to_function_name.get(call_id)
                if not fn_name:
                    raise TypeError(
                        f"Unmatched tool output callId '{call_id}': missing preceding assistant tool call"
                    )

                parts.append(
                    {
                        "functionResponse": {
                            "name": fn_name,
                            "response": {"output": output, "callId": call_id},
                        }
                    }
                )

            # Tool responses must come from the "user" role.
            contents.append({"role": "user", "parts": parts})
            continue

        raise TypeError(f"Unsupported message role: {role}")

    payload = {"contents": contents}
    if system_chunks:
        payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_chunks)}]}
    return payload


def dataiku_tools_to_gemini(*, tools):
    def _normalize_schema(schema, path: str):
        if schema is None:
            return None
        if not isinstance(schema, dict):
            raise TypeError(f"Invalid JSON schema at {path}: expected object")

        # Drop annotation-only metadata
        out = {}
        for key, value in schema.items():
            if key in {"$schema", "$id"}:
                continue
            out[key] = value

        # Validate/normalize supported keywords
        supported = {
            "type",
            "description",
            "properties",
            "required",
            "items",
            "enum",
            "additionalProperties",
            "nullable",
            "title",
            "default",
        }

        # Reject unsupported structural/validation keywords explicitly
        for key in list(out.keys()):
            if key.startswith("$") and key not in {"$ref"}:
                raise TypeError(f"Unsupported JSON schema keyword '{key}' at {path}")
            if key not in supported:
                # Allow common harmless annotations? We choose to reject to avoid silent behavior.
                raise TypeError(f"Unsupported JSON schema keyword '{key}' at {path}")

        schema_type = out.get("type")
        if schema_type is None:
            # Default to object if properties exist
            if "properties" in out:
                schema_type = "object"
                out["type"] = "object"
        if isinstance(schema_type, list):
            # Normalize [T, 'null'] -> T with nullable
            non_null = [t for t in schema_type if t != "null"]
            if len(non_null) == 1 and len(non_null) != len(schema_type):
                out["type"] = non_null[0]
                out["nullable"] = True
            else:
                raise TypeError(f"Unsupported JSON schema type union at {path}: {schema_type}")

        if out.get("type") not in {"object", "array", "string", "number", "integer", "boolean", None}:
            raise TypeError(f"Unsupported JSON schema type at {path}: {out.get('type')}")

        if out.get("type") == "object":
            props = out.get("properties") or {}
            if not isinstance(props, dict):
                raise TypeError(f"Invalid properties at {path}.properties")
            normalized_props = {}
            for prop_name, prop_schema in props.items():
                normalized_props[prop_name] = _normalize_schema(
                    prop_schema, f"{path}.properties.{prop_name}"
                )
            if normalized_props:
                out["properties"] = normalized_props

            req = out.get("required")
            if req is not None and not isinstance(req, list):
                raise TypeError(f"Invalid required at {path}.required")

            ap = out.get("additionalProperties")
            if ap is not None and not isinstance(ap, (bool, dict)):
                raise TypeError(
                    f"Invalid additionalProperties at {path}.additionalProperties"
                )
            if isinstance(ap, dict):
                out["additionalProperties"] = _normalize_schema(
                    ap, f"{path}.additionalProperties"
                )

        if out.get("type") == "array":
            items = out.get("items")
            if items is None:
                raise TypeError(f"Missing items for array schema at {path}")
            out["items"] = _normalize_schema(items, f"{path}.items")

        return out

    if tools is None:
        return None
    if not isinstance(tools, list):
        raise TypeError("Invalid tools setting: tools must be a list")

    declarations = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise TypeError("Invalid tools setting: each tool must be an object")
        if tool.get("type") != "function":
            raise TypeError(f"Unsupported tool type: {tool.get('type')}")
        function = tool.get("function")
        if not isinstance(function, dict):
            raise TypeError("Invalid tools setting: function tool is missing function")
        name = function.get("name")
        if not name or not isinstance(name, str):
            raise TypeError("Invalid tools setting: function name must be a string")

        decl = {
            "name": name,
            "description": function.get("description") or "",
        }

        parameters = function.get("parameters")
        if parameters is not None:
            if not isinstance(parameters, dict):
                raise TypeError("Invalid tools setting: parameters must be an object")
            normalized = _normalize_schema(parameters, f"tools[{name}].parameters")
            if normalized.get("type") not in {"object", None}:
                raise TypeError("Invalid tools setting: parameters.type must be 'object'")
            decl["parameters"] = normalized

        declarations.append(decl)

    if not declarations:
        return None

    return [{"functionDeclarations": declarations}]


def dataiku_tool_outputs_to_gemini(*, query_messages, prior_tool_calls):
    """Convert Dataiku tool output messages into Gemini functionResponse parts.

    Dataiku tool outputs are represented as messages with role=="tool" and a
    "toolOutputs" array with {callId, output}.

    Gemini native API typically uses functionResponse parts, but the exact
    call-id roundtrip semantics are not universally documented across all
    Vertex Gemini publisher models.

    This implementation associates responses by name when possible and includes
    the callId in the response payload for traceability.
    """

    if not query_messages:
        return []

    # Build a callId -> name mapping from prior extracted tool calls.
    call_id_to_name = {}
    for tool_call in prior_tool_calls or []:
        if not isinstance(tool_call, dict):
            continue
        call_id = tool_call.get("id")
        fn = (tool_call.get("function") or {}).get("name")
        if call_id and fn:
            call_id_to_name[call_id] = fn

    parts = []
    for message in query_messages:
        if (message.get("role") or "").strip() != "tool":
            continue
        tool_outputs = message.get("toolOutputs") or []
        if not isinstance(tool_outputs, list):
            raise TypeError("Invalid toolOutputs: must be a list")
        for tool_output in tool_outputs:
            if not isinstance(tool_output, dict):
                raise TypeError("Invalid toolOutputs: each entry must be an object")
            call_id = tool_output.get("callId")
            output = tool_output.get("output")
            if not call_id or not isinstance(call_id, str):
                raise TypeError("Invalid toolOutputs: missing callId")
            if output is None:
                output = ""
            if not isinstance(output, str):
                raise TypeError("Invalid toolOutputs: output must be a string")

            name = call_id_to_name.get(call_id)
            if not name:
                # Unverified behavior: Gemini functionResponse requires a name.
                # Fall back to including callId as name to keep association stable.
                name = call_id

            parts.append(
                {
                    "functionResponse": {
                        "name": name,
                        "response": {"output": output, "callId": call_id},
                    }
                }
            )

    if not parts:
        return []

    # Tool responses must come from the "user" role.
    return [{"role": "user", "parts": parts}]


def build_generation_config(*, max_output_tokens: int, temperature: float):
    cfg = {}
    if max_output_tokens is not None:
        cfg["maxOutputTokens"] = int(max_output_tokens)
    if temperature is not None:
        cfg["temperature"] = float(temperature)
    return cfg


def invoke_generate_content(*, url: str, bearer_token: str, payload: dict, timeout_s=60):
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Goog-Api-Client": "schwab-llm-plugin/vertex-gemini",
    }

    requests = _requests_module()
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    if getattr(resp, "ok", False):
        out = resp.json()
        if isinstance(out, dict):
            out.setdefault("_diagnostics", {})
            out["_diagnostics"]["http_status"] = getattr(resp, "status_code", None)
        return out

    text = getattr(resp, "text", "")
    status = getattr(resp, "status_code", "unknown")
    raise GeminiAPIError(f"Vertex generateContent failed ({status}): {text}")


def build_empty_response_diagnostics(response_json: dict):
    """Build safe diagnostics for unexpected/empty generateContent responses."""

    if not isinstance(response_json, dict):
        return {
            "top_level_keys": [],
            "candidates": 0,
            "finish_reasons": [],
            "prompt_feedback_block_reason": None,
            "text_parts": 0,
            "function_call_parts": 0,
        }

    candidates = response_json.get("candidates") or []
    if not isinstance(candidates, list):
        candidates = []

    finish_reasons = []
    text_parts = 0
    function_call_parts = 0

    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        fr = cand.get("finishReason")
        if fr is not None:
            finish_reasons.append(fr)
        content = cand.get("content") or {}
        parts = content.get("parts") or []
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            if part.get("text"):
                text_parts += 1
            if part.get("functionCall"):
                function_call_parts += 1

    prompt_feedback = response_json.get("promptFeedback") or {}
    block_reason = None
    if isinstance(prompt_feedback, dict):
        block_reason = prompt_feedback.get("blockReason")

    diagnostics = {
        "http_status": (response_json.get("_diagnostics") or {}).get("http_status"),
        "top_level_keys": sorted([k for k in response_json.keys() if not k.startswith("_")]),
        "candidates": len(candidates),
        "finish_reasons": finish_reasons,
        "prompt_feedback_block_reason": block_reason,
        "text_parts": text_parts,
        "function_call_parts": function_call_parts,
    }
    return diagnostics


def parse_generate_content_response(response_json: dict):
    """Parse Gemini generateContent response into (text, tool_calls).

    Tool calls are returned in Dataiku-compatible schema:
      {"id": ..., "type":"function", "function":{"name":..., "arguments":...}}

    Note: Gemini native functionCall parts do not always expose a stable call id.
    We generate one when absent.
    """

    if not isinstance(response_json, dict):
        raise TypeError("Invalid response: expected object")

    candidates = response_json.get("candidates") or []
    if not isinstance(candidates, list):
        candidates = []

    out_text = ""
    tool_calls = []

    for cand in candidates:
        content = (cand or {}).get("content") or {}
        parts = content.get("parts") or []
        if not isinstance(parts, list):
            continue

        for part in parts:
            if not isinstance(part, dict):
                continue
            if "text" in part and part.get("text"):
                out_text += part.get("text") or ""

            function_call = part.get("functionCall")
            if function_call:
                name = function_call.get("name")
                args = function_call.get("args")
                if args is None:
                    args = {}
                if not isinstance(args, (dict, list, str, int, float, bool)):
                    args = str(args)

                try:
                    arguments = args if isinstance(args, str) else json.dumps(args)
                except TypeError:
                    arguments = json.dumps(str(args))

                tool_calls.append(
                    {
                        "id": function_call.get("id") or f"call_{uuid.uuid4().hex}",
                        "type": "function",
                        "function": {"name": name, "arguments": arguments or "{}"},
                    }
                )

    return out_text, tool_calls


def parse_or_raise_empty(response_json: dict):
    """Parse response and raise a diagnostic error if it is empty."""

    text, tool_calls = parse_generate_content_response(response_json)
    if (text is not None and str(text).strip()) or tool_calls:
        return text, tool_calls

    diag = build_empty_response_diagnostics(response_json)
    raise GeminiAPIError(
        "Empty Vertex generateContent response (no text parts, no function calls). "
        f"Diagnostics: {json.dumps(diag, sort_keys=True)}"
    )
