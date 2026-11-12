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
            contents.append({"role": "model", "parts": [{"text": content}]})
            continue

        if role == "tool":
            # Tool outputs are handled by dataiku_tool_outputs_to_gemini
            raise TypeError(
                "Tool messages must be translated with dataiku_tool_outputs_to_gemini"
            )

        raise TypeError(f"Unsupported message role: {role}")

    payload = {"contents": contents}
    if system_chunks:
        payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_chunks)}]}
    return payload


def dataiku_tools_to_gemini(*, tools):
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
            # Gemini expects JSONSchema-ish objects. Validate minimally and pass through.
            if parameters.get("type") not in {"object", None}:
                raise TypeError("Invalid tools setting: parameters.type must be 'object'")
            decl["parameters"] = parameters

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
        return resp.json()

    text = getattr(resp, "text", "")
    status = getattr(resp, "status_code", "unknown")
    raise GeminiAPIError(f"Vertex generateContent failed ({status}): {text}")


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

