import json

import pytest


def test_vertex_api_host_global_and_regional():
    from vertexai_shared import gemini

    assert gemini.vertex_api_host_for_location("global") == "aiplatform.googleapis.com"
    assert (
        gemini.vertex_api_host_for_location("us-central1")
        == "us-central1-aiplatform.googleapis.com"
    )


def test_build_generate_content_url_global():
    from vertexai_shared import gemini

    url = gemini.build_generate_content_url(
        project_id="p",
        location="global",
        model_id="gemini-3.8-flash",
    )
    assert url.startswith("https://aiplatform.googleapis.com/v1/projects/p/locations/global")
    assert url.endswith("/publishers/google/models/gemini-3.8-flash:generateContent")


def test_dataiku_messages_to_gemini_system_and_history():
    from vertexai_shared import gemini

    payload = gemini.dataiku_messages_to_gemini(
        messages=[
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Call a tool"},
        ]
    )
    assert payload["systemInstruction"]["parts"][0]["text"] == "Be concise"
    assert payload["contents"][0]["role"] == "user"
    assert payload["contents"][1]["role"] == "model"
    assert payload["contents"][2]["role"] == "user"


def test_dataiku_messages_to_gemini_tool_calls_and_outputs_single():
    from vertexai_shared import gemini

    payload = gemini.dataiku_messages_to_gemini(
        messages=[
            {
                "role": "assistant",
                "content": "",
                "toolCalls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "t", "arguments": "{\"a\": 1}"},
                    }
                ],
            },
            {
                "role": "tool",
                "toolOutputs": [{"callId": "call_1", "output": "ok"}],
            },
        ]
    )

    assert payload["contents"][0]["role"] == "model"
    assert payload["contents"][0]["parts"][0]["functionCall"]["name"] == "t"
    assert payload["contents"][1]["role"] == "user"
    assert payload["contents"][1]["parts"][0]["functionResponse"]["name"] == "t"


def test_dataiku_messages_to_gemini_thought_signature_compat_disabled_by_default():
    from vertexai_shared import gemini

    payload = gemini.dataiku_messages_to_gemini(
        messages=[
            {
                "role": "assistant",
                "content": "",
                "toolCalls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "t", "arguments": "{}"},
                    }
                ],
            }
        ]
    )
    assert "thoughtSignature" not in payload["contents"][0]["parts"][0]


def test_dataiku_messages_to_gemini_thought_signature_compat_enabled_on_first_parallel_call_only():
    from vertexai_shared import gemini

    payload = gemini.dataiku_messages_to_gemini(
        messages=[
            {
                "role": "assistant",
                "content": "",
                "toolCalls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "t", "arguments": "{}"},
                    },
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {"name": "u", "arguments": "{}"},
                    },
                ],
            }
        ],
        thought_signature_compat_mode=True,
    )
    parts = payload["contents"][0]["parts"]
    assert parts[0]["thoughtSignature"] == "skip_thought_signature_validator"
    assert "thoughtSignature" not in parts[1]


def test_dataiku_messages_to_gemini_tool_calls_and_outputs_multiple_and_repeated_names():
    from vertexai_shared import gemini

    payload = gemini.dataiku_messages_to_gemini(
        messages=[
            {
                "role": "assistant",
                "content": "",
                "toolCalls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "t", "arguments": "{}"},
                    },
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {"name": "t", "arguments": "{\"q\":\"x\"}"},
                    },
                    {
                        "id": "c3",
                        "type": "function",
                        "function": {"name": "u", "arguments": "{}"},
                    },
                ],
            },
            {
                "role": "tool",
                "toolOutputs": [
                    {"callId": "c1", "output": "o1"},
                    {"callId": "c2", "output": "o2"},
                    {"callId": "c3", "output": "err: boom"},
                ],
            },
        ]
    )

    # Order preserved
    model_parts = payload["contents"][0]["parts"]
    assert [p["functionCall"]["name"] for p in model_parts] == ["t", "t", "u"]

    user_parts = payload["contents"][1]["parts"]
    assert [p["functionResponse"]["name"] for p in user_parts] == ["t", "t", "u"]


def test_dataiku_messages_to_gemini_tool_outputs_requires_matching_call_id():
    from vertexai_shared import gemini

    with pytest.raises(TypeError) as exc:
        gemini.dataiku_messages_to_gemini(
            messages=[
                {
                    "role": "assistant",
                    "content": "",
                    "toolCalls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "t", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "toolOutputs": [{"callId": "nope", "output": "x"}],
                },
            ]
        )
    assert "Unmatched tool output callId" in str(exc.value)


def test_dataiku_tools_to_gemini_declarations():
    from vertexai_shared import gemini

    tools = [
        {
            "type": "function",
            "function": {
                "name": "my_tool",
                "description": "desc",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            },
        }
    ]
    out = gemini.dataiku_tools_to_gemini(tools=tools)
    assert out == [
        {
            "functionDeclarations": [
                {
                    "name": "my_tool",
                    "description": "desc",
                    "parameters": tools[0]["function"]["parameters"],
                }
            ]
        }
    ]


def test_dataiku_tools_to_gemini_drops_schema_annotations_without_mutating_input():
    from vertexai_shared import gemini

    parameters = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "abc",
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
    }
    tools = [
        {
            "type": "function",
            "function": {"name": "t", "description": "d", "parameters": parameters},
        }
    ]
    out = gemini.dataiku_tools_to_gemini(tools=tools)
    normalized = out[0]["functionDeclarations"][0]["parameters"]
    assert "$schema" not in normalized
    assert "$id" not in normalized
    # ensure original object unchanged
    assert "$schema" in parameters and "$id" in parameters


def test_dataiku_tools_to_gemini_rejects_unsupported_keyword_with_path():
    from vertexai_shared import gemini

    tools = [
        {
            "type": "function",
            "function": {
                "name": "t",
                "description": "d",
                "parameters": {"type": "object", "minLength": 1},
            },
        }
    ]
    with pytest.raises(TypeError) as exc:
        gemini.dataiku_tools_to_gemini(tools=tools)
    assert "minLength" in str(exc.value)
    assert "tools[t].parameters" in str(exc.value)


def test_dataiku_tools_to_gemini_normalizes_nested_objects_and_arrays():
    from vertexai_shared import gemini

    tools = [
        {
            "type": "function",
            "function": {
                "name": "t",
                "description": "d",
                "parameters": {
                    "$schema": "x",
                    "type": ["object", "null"],
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {
                                "$id": "y",
                                "type": "object",
                                "properties": {"k": {"type": "string", "enum": ["a", "b"]}},
                                "required": ["k"],
                            },
                        }
                    },
                },
            },
        }
    ]
    out = gemini.dataiku_tools_to_gemini(tools=tools)
    params = out[0]["functionDeclarations"][0]["parameters"]
    assert params["type"] == "object"
    assert params["nullable"] is True
    assert params["properties"]["items"]["type"] == "array"
    assert params["properties"]["items"]["items"]["type"] == "object"


def test_parse_generate_content_response_text_and_function_call():
    from vertexai_shared import gemini

    response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "hello"},
                        {"functionCall": {"name": "t", "args": {"a": 1}}},
                    ]
                }
            }
        ]
    }
    text, tool_calls = gemini.parse_generate_content_response(response)
    assert text == "hello"
    assert tool_calls[0]["type"] == "function"
    assert tool_calls[0]["function"]["name"] == "t"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"a": 1}


def test_parse_or_raise_empty_candidates_raises_diagnostics():
    from vertexai_shared import gemini

    with pytest.raises(gemini.GeminiAPIError) as exc:
        gemini.parse_or_raise_empty({"candidates": []})
    msg = str(exc.value)
    assert "Empty Vertex generateContent response" in msg
    assert "\"candidates\": 0" in msg


def test_parse_or_raise_blocked_prompt_feedback_includes_block_reason():
    from vertexai_shared import gemini

    with pytest.raises(gemini.GeminiAPIError) as exc:
        gemini.parse_or_raise_empty(
            {
                "promptFeedback": {"blockReason": "SAFETY"},
                "candidates": [],
            }
        )
    assert "SAFETY" in str(exc.value)


def test_parse_or_raise_candidate_without_parts_raises_with_finish_reason():
    from vertexai_shared import gemini

    with pytest.raises(gemini.GeminiAPIError) as exc:
        gemini.parse_or_raise_empty({"candidates": [{"finishReason": "STOP"}]})
    assert "STOP" in str(exc.value)


def test_get_bearer_token_refreshes_custom_credentials():
    from vertexai_shared import gemini

    class Creds:
        valid = False
        token = None

        def refresh(self, request):
            self.valid = True
            self.token = "refreshed"

    assert gemini.get_bearer_token(credentials=Creds()) == "refreshed"


def test_dataiku_tool_outputs_to_gemini_associates_call_id():
    from vertexai_shared import gemini

    tool_calls = [
        {"id": "call_1", "type": "function", "function": {"name": "t", "arguments": "{}"}}
    ]
    contents = gemini.dataiku_tool_outputs_to_gemini(
        query_messages=[
            {
                "role": "tool",
                "toolOutputs": [{"callId": "call_1", "output": "ok"}],
            }
        ],
        prior_tool_calls=tool_calls,
    )
    assert contents[0]["role"] == "user"
    part = contents[0]["parts"][0]["functionResponse"]
    assert part["name"] == "t"
    assert part["response"]["output"] == "ok"


def test_dataiku_tool_outputs_without_prior_calls_uses_call_id_as_name():
    from vertexai_shared import gemini

    contents = gemini.dataiku_tool_outputs_to_gemini(
        query_messages=[
            {
                "role": "tool",
                "toolOutputs": [{"callId": "call_1", "output": "ok"}],
            }
        ],
        prior_tool_calls=[],
    )
    part = contents[0]["parts"][0]["functionResponse"]
    assert part["name"] == "call_1"


def test_invoke_generate_content_raises_on_http_error(monkeypatch):
    from vertexai_shared import gemini

    class BadResp:
        ok = False
        status_code = 401
        text = "nope"

    class Requests:
        def post(self, url, headers=None, json=None, timeout=None):
            return BadResp()

    monkeypatch.setattr(gemini, "_requests_module", lambda: Requests())

    with pytest.raises(gemini.GeminiAPIError) as exc:
        gemini.invoke_generate_content(
            url="u",
            bearer_token="t",
            payload={},
        )
    assert "Vertex generateContent failed" in str(exc.value)
