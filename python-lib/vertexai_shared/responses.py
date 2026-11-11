import json


def extract_text(prediction):
    """Extract text from a single prediction object.

    Mirrors `MyLLM.extract_text` from `python-llms/vertex-ai/llm.py`.
    """

    if isinstance(prediction, dict) and "choices" in prediction:
        return prediction["choices"][0]["message"]["content"]
    if isinstance(prediction, str):
        return prediction
    return json.dumps(prediction, indent=2, default=str)


def extract_answer(predictions) -> str:
    """Extract and concatenate text across predictions."""

    if isinstance(predictions, dict):
        predictions = [predictions]
    answer = ""
    for prediction in predictions:
        answer += extract_text(prediction)
    return answer

