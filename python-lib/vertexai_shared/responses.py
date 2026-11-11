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
