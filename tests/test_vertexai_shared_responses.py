import json

from vertexai_shared.responses import extract_answer


def test_extract_answer_string_prediction():
    assert extract_answer(["hello"]) == "hello"


def test_extract_answer_openai_style_prediction():
    prediction = {"choices": [{"message": {"content": "hi"}}]}
    assert extract_answer([prediction]) == "hi"


def test_extract_answer_openai_style_null_message_content_is_empty_string():
    prediction = {"choices": [{"message": {"content": None}}]}
    assert extract_answer([prediction]) == ""


def test_extract_answer_openai_style_null_content_with_tool_calls_is_empty_string():
    prediction = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "kb_search", "arguments": "{}"},
                        }
                    ],
                }
            }
        ]
    }
    assert extract_answer([prediction]) == ""


def test_extract_answer_multiple_concatenation():
    prediction = {"choices": [{"message": {"content": "A"}}]}
    assert extract_answer([prediction, "B"]) == "AB"


def test_extract_answer_dict_is_wrapped_list():
    prediction = {"choices": [{"message": {"content": "solo"}}]}
    assert extract_answer(prediction) == "solo"


def test_extract_answer_json_fallback():
    payload = {"foo": [1, 2]}
    assert extract_answer([payload]) == json.dumps(payload, indent=2, default=str)
