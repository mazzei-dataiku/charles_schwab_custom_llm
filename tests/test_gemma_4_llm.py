import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _load_gemma_4_module():
    llm_path = (
        Path(__file__).resolve().parents[1] / "python-llms" / "gemma-4" / "llm.py"
    )
    spec = importlib.util.spec_from_file_location("gemma_4_llm", llm_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_default_model_variant_instruction_tuned(monkeypatch):
    module = _load_gemma_4_module()
    llm = module.MyLLM()

    monkeypatch.setattr(module.vertex_auth, "build_credentials", lambda plugin_config: None)
    monkeypatch.setattr(module.vertex_client, "init_aiplatform", lambda **kwargs: None)
    monkeypatch.setattr(module.vertex_client, "build_endpoint", lambda **kwargs: MagicMock())

    llm.set_config(
        {"endpoint_id": "eid"},
        {"project_id": "p", "location": "l", "psc_address": "10.0.0.1"},
    )
    assert llm._config["model_variant"] == "INSTRUCTION_TUNED"


def test_invalid_model_variant_raises():
    module = _load_gemma_4_module()
    llm = module.MyLLM()

    with pytest.raises(TypeError) as exc:
        llm.set_config(
            {"endpoint_id": "eid", "model_variant": "NOPE"},
            {"project_id": "p", "location": "l", "psc_address": "10.0.0.1"},
        )
    assert "Invalid model setting: model_variant must be one of BASE or INSTRUCTION_TUNED" in str(
        exc.value
    )


def test_process_builds_plain_prompt_and_payload(monkeypatch):
    module = _load_gemma_4_module()
    llm = module.MyLLM()

    monkeypatch.setattr(module.vertex_auth, "build_credentials", lambda plugin_config: None)
    monkeypatch.setattr(module.vertex_client, "init_aiplatform", lambda **kwargs: None)

    endpoint = MagicMock()
    monkeypatch.setattr(module.vertex_client, "build_endpoint", lambda **kwargs: endpoint)

    response = MagicMock(predictions=[{"choices": [{"message": {"content": "ok"}}]}])
    predict_mock = MagicMock(return_value=response)
    monkeypatch.setattr(module.vertex_client, "predict", predict_mock)

    monkeypatch.setattr(module.vertex_responses, "extract_answer", lambda preds: "ok")

    llm.set_config(
        {"endpoint_id": "eid", "endpoint_type": "PRIVATE_PSC", "model_variant": "BASE"},
        {"project_id": "p", "location": "l", "psc_address": "10.0.0.1"},
    )

    query = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is the capital of Arizona?"},
        ]
    }
    out = llm.process(query, settings={}, trace={})

    assert set(out.keys()) == {"text", "prompt"}
    assert out["text"] == "ok"
    assert out["prompt"] == "You are helpful.\n\nWhat is the capital of Arizona?"

    predict_mock.assert_called_once()
    called_kwargs = predict_mock.call_args.kwargs
    assert called_kwargs["endpoint"] is endpoint
    assert called_kwargs["endpoint_type"] == "PRIVATE_PSC"
    assert called_kwargs["psc_address"] == "10.0.0.1"

    instances = called_kwargs["instances"]
    assert instances == [{"prompt": out["prompt"], "max_tokens": 128}]


def test_instruction_tuned_builds_chat_completions_request(monkeypatch):
    module = _load_gemma_4_module()
    llm = module.MyLLM()

    monkeypatch.setattr(module.vertex_auth, "build_credentials", lambda plugin_config: None)
    monkeypatch.setattr(module.vertex_client, "init_aiplatform", lambda **kwargs: None)

    endpoint = MagicMock()
    monkeypatch.setattr(module.vertex_client, "build_endpoint", lambda **kwargs: endpoint)

    response = MagicMock(
        predictions={
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Phoenix"},
                }
            ]
        }
    )
    predict_mock = MagicMock(return_value=response)
    monkeypatch.setattr(module.vertex_client, "predict", predict_mock)

    llm.set_config(
        {
            "endpoint_id": "eid",
            "endpoint_type": "DEDICATED",
            "model_variant": "INSTRUCTION_TUNED",
        },
        {"project_id": "p", "location": "l", "psc_address": "10.0.0.1"},
    )

    query = {
        "messages": [
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "What is the capital of Arizona?"},
            {"role": "assistant", "content": "Let me think"},
            {"role": "user", "content": "Just answer"},
        ]
    }
    out = llm.process(query, settings={}, trace={})

    assert set(out.keys()) == {"text", "prompt"}
    assert out["text"] == "Phoenix"
    assert out["prompt"] == "Be concise\n\nWhat is the capital of Arizona?\n\nLet me think\n\nJust answer"

    called_kwargs = predict_mock.call_args.kwargs
    instances = called_kwargs["instances"]
    assert instances == [
        {
            "@requestFormat": "chatCompletions",
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "Be concise"}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is the capital of Arizona?"}
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Let me think"}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Just answer"}],
                },
            ],
            "max_tokens": 128,
        }
    ]


def test_instruction_tuned_rejects_unsupported_role(monkeypatch):
    module = _load_gemma_4_module()
    llm = module.MyLLM()

    monkeypatch.setattr(module.vertex_auth, "build_credentials", lambda plugin_config: None)
    monkeypatch.setattr(module.vertex_client, "init_aiplatform", lambda **kwargs: None)
    monkeypatch.setattr(module.vertex_client, "build_endpoint", lambda **kwargs: MagicMock())

    llm.set_config(
        {"endpoint_id": "eid", "model_variant": "INSTRUCTION_TUNED"},
        {"project_id": "p", "location": "l", "psc_address": "10.0.0.1"},
    )

    query = {"messages": [{"role": "tool", "content": "x"}]}
    with pytest.raises(TypeError) as exc:
        llm.process(query, settings={}, trace={})
    assert "Unsupported message role: tool" in str(exc.value)


def test_instruction_tuned_rejects_non_text_content(monkeypatch):
    module = _load_gemma_4_module()
    llm = module.MyLLM()

    monkeypatch.setattr(module.vertex_auth, "build_credentials", lambda plugin_config: None)
    monkeypatch.setattr(module.vertex_client, "init_aiplatform", lambda **kwargs: None)
    monkeypatch.setattr(module.vertex_client, "build_endpoint", lambda **kwargs: MagicMock())

    llm.set_config(
        {"endpoint_id": "eid", "model_variant": "INSTRUCTION_TUNED"},
        {"project_id": "p", "location": "l", "psc_address": "10.0.0.1"},
    )

    query = {"messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}]}
    with pytest.raises(TypeError) as exc:
        llm.process(query, settings={}, trace={})
    assert "Only text message content is supported" in str(exc.value)
