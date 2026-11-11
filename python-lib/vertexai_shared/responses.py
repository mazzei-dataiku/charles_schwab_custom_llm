import json


def extract_text(prediction):
    """Extract text from a single prediction object.

    Mirrors `MyLLM.extract_text` from `python-llms/vertex-ai/llm.py`.
    """

    if isinstance(prediction, dict) and "choices" in prediction:
        content = prediction["choices"][0]["message"].get("content")
        return "" if content is None else content
    if isinstance(prediction, str):
        return prediction
    return json.dumps(prediction, indent=2, default=str)


def extract_answer(predictions) -> str:
    """Extract and concatenate text across predictions."""

    if isinstance(predictions, dict):
        predictions = [predictions]
    answer = ""
    for prediction in predictions:
        text = extract_text(prediction)
        answer += "" if text is None else text
    return answer


def extract_tool_calls(prediction):
    """Extract OpenAI-style tool calls from a single prediction.

    Returns a list of Dataiku-compatible FunctionToolCall dicts:
    {"id": ..., "type": "function", "function": {"name": ..., "arguments": ...}}
    """

    if not (isinstance(prediction, dict) and "choices" in prediction):
        return []

    message = prediction["choices"][0].get("message") or {}
    tool_calls = message.get("tool_calls") or []
    if tool_calls is None:
        return []
    if not isinstance(tool_calls, list):
        return []

    extracted = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        if tc.get("type") != "function":
            continue
        function = tc.get("function") or {}
        extracted.append(
            {
                "id": tc.get("id"),
                "type": "function",
                "function": {
                    "name": function.get("name"),
                    "arguments": function.get("arguments") or "{}",
                },
            }
        )

    return extracted


def extract_tool_call_chunks(predictions):
    """Return tool call chunks for Dataiku streamed output.

    Dataiku agents expect streamed chunks to include a `toolCalls` field.
    The internal agent code reassembles tool calls from chunks.
    """

    if isinstance(predictions, dict):
        predictions = [predictions]
    chunks = []
    for prediction in predictions:
        chunks.extend(extract_tool_calls(prediction))
    return chunks
