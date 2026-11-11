import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _load_gemma_legacy_module():
    llm_path = (
        Path(__file__).resolve().parents[1] / "python-llms" / "gemma-legacy" / "llm.py"
    )
    spec = importlib.util.spec_from_file_location("gemma_legacy_llm", llm_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_gemma_prompt_exact():
    module = _load_gemma_legacy_module()
    llm = module.MyLLM()
    prompt = llm.build_gemma_prompt("sys", " user ")
    assert (
        prompt
        == "<start_of_turn>user\nsys\n\nuser<end_of_turn>\n<start_of_turn>model\n"
    )


def test_process_payload_and_response_structure(monkeypatch):
    module = _load_gemma_legacy_module()
    llm = module.MyLLM()

    monkeypatch.setattr(module.vertex_auth, "build_credentials", lambda plugin_config: None)

    init_mock = MagicMock()
    monkeypatch.setattr(module.vertex_client, "init_aiplatform", init_mock)

    endpoint = MagicMock()
    build_endpoint_mock = MagicMock(return_value=endpoint)
    monkeypatch.setattr(module.vertex_client, "build_endpoint", build_endpoint_mock)

    response = MagicMock(predictions=["ok"])
    predict_mock = MagicMock(return_value=response)
    monkeypatch.setattr(module.vertex_client, "predict", predict_mock)

    extract_answer_mock = MagicMock(return_value="ok")
    monkeypatch.setattr(module.vertex_responses, "extract_answer", extract_answer_mock)

    plugin_config = {
        "project_id": "p",
        "location": "l",
        "psc_address": "10.0.0.1",
    }
    config = {"endpoint_id": "eid"}
    llm.set_config(config, plugin_config)

    query = {"messages": [{"role": "user", "content": "hello"}]}
    out = llm.process(query, settings={}, trace={})

    assert set(out.keys()) == {"text", "prompt"}
    assert out["text"] == "ok"

    prompt = out["prompt"]
    expected_instances = [
        {
            "prompt": prompt,
            "max_tokens": 1024,
            "temperature": 0.2,
            "top_p": 0.95,
            "stop": ["<end_of_turn>"],
            "raw_response": True,
        }
    ]

    predict_mock.assert_called_once_with(
        endpoint=endpoint,
        endpoint_type="PRIVATE_PSC",
        instances=expected_instances,
        psc_address="10.0.0.1",
    )
    extract_answer_mock.assert_called_once_with(response.predictions)


@pytest.mark.parametrize(
    "config_value,expected",
    [
        ({}, (1024, 0.2, 0.95)),
        ({"max_tokens": None}, (1024, 0.2, 0.95)),
        ({"temperature": None}, (1024, 0.2, 0.95)),
        ({"topp": None}, (1024, 0.2, 0.95)),
    ],
)
def test_defaults_preserved(monkeypatch, config_value, expected):
    module = _load_gemma_legacy_module()
    llm = module.MyLLM()

    monkeypatch.setattr(module.vertex_auth, "build_credentials", lambda plugin_config: None)
    monkeypatch.setattr(module.vertex_client, "init_aiplatform", lambda **kwargs: None)
    monkeypatch.setattr(module.vertex_client, "build_endpoint", lambda **kwargs: MagicMock())

    plugin_config = {"project_id": "p", "location": "l", "psc_address": "10.0.0.1"}
    config = {"endpoint_id": "eid", **config_value}
    llm.set_config(config, plugin_config)

    assert llm._config["max_tokens"] == expected[0]
    assert llm._config["temperature"] == expected[1]
    assert llm._config["topp"] == expected[2]

