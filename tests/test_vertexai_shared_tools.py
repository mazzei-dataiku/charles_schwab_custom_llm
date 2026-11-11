import pytest

from vertexai_shared.tools import dataiku_tools_to_openai


def test_dataiku_tools_to_openai_empty_when_none():
    assert dataiku_tools_to_openai(None) == []


def test_dataiku_tools_to_openai_translates_function_tool():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "SQL_dataset_Query__874fe6",
                "description": "Query the dataset",
                "parameters": {"type": "object", "properties": {"question": {"type": "string"}}},
            },
        }
    ]
    out = dataiku_tools_to_openai(tools)
    assert out == tools


def test_dataiku_tools_to_openai_rejects_non_list():
    with pytest.raises(TypeError):
        dataiku_tools_to_openai({})

