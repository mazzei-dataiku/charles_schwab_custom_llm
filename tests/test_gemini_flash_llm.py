import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _load_gemini_flash_module():
    llm_path = (
        Path(__file__).resolve().parents[1] / "python-llms" / "gemini-flash" / "llm.py"
    )
    spec = importlib.util.spec_from_file_location("gemini_flash_llm", llm_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_set_config_defaults_model_and_location(monkeypatch):
    module = _load_gemini_flash_module()
    llm = module.MyLLM()

    monkeypatch.setattr(module.vertex_auth, "build_credentials", lambda plugin_config: "creds")
    llm.set_config(
        config={},
        plugin_config={"project_id": "p", "location": "us-central1"},
    )
    assert llm._config["model_id"] == "gemini-3.8-flash"
    # Default Gemini routing must be global, not inherited from plugin location.
    assert llm._plugin_config["location"] == "global"


def test_set_config_allows_explicit_location_override(monkeypatch):
    module = _load_gemini_flash_module()
    llm = module.MyLLM()

    monkeypatch.setattr(module.vertex_auth, "build_credentials", lambda plugin_config: "creds")
    llm.set_config(
        config={"location": "europe-west4"},
        plugin_config={"project_id": "p", "location": "us-central1"},
    )
    assert llm._plugin_config["location"] == "europe-west4"


def test_process_builds_payload_and_returns_text(monkeypatch):
    module = _load_gemini_flash_module()
    llm = module.MyLLM()

    monkeypatch.setattr(module.vertex_auth, "build_credentials", lambda plugin_config: "creds")

    build_url_mock = MagicMock(return_value="url")
    monkeypatch.setattr(module.vertex_gemini, "build_generate_content_url", build_url_mock)
    monkeypatch.setattr(module.vertex_gemini, "get_bearer_token", lambda **kwargs: "token")

    invoke_mock = MagicMock(return_value={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})
    monkeypatch.setattr(module.vertex_gemini, "invoke_generate_content", invoke_mock)

    llm.set_config(
        config={"max_output_tokens": 10, "temperature": 0.1},
        plugin_config={"project_id": "p", "location": "us-central1"},
    )

    out = llm.process(
        query={"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]},
        settings={"tools": []},
        trace={},
    )
    assert out["text"] == "ok"
    assert isinstance(out["toolCalls"], list)

    payload = invoke_mock.call_args.kwargs["payload"]
    assert payload["systemInstruction"]["parts"][0]["text"] == "s"
    assert payload["contents"][0]["role"] == "user"

    # Location used for URL must be global by default
    assert build_url_mock.call_args.kwargs["location"] == "global"


def test_process_stream_emits_tool_calls(monkeypatch):
    module = _load_gemini_flash_module()
    llm = module.MyLLM()

    monkeypatch.setattr(module.vertex_auth, "build_credentials", lambda plugin_config: "creds")
    monkeypatch.setattr(module.vertex_gemini, "build_generate_content_url", lambda **kwargs: "url")
    monkeypatch.setattr(module.vertex_gemini, "get_bearer_token", lambda **kwargs: "token")

    # Response with a functionCall part.
    monkeypatch.setattr(
        module.vertex_gemini,
        "invoke_generate_content",
        lambda **kwargs: {
            "candidates": [
                {"content": {"parts": [{"functionCall": {"name": "t", "args": {"a": 1}}}]}}
            ]
        },
    )

    llm.set_config(
        config={"max_output_tokens": 10, "temperature": 0.1},
        plugin_config={"project_id": "p", "location": "global"},
    )

    events = list(
        llm.process_stream(
            query={"messages": [{"role": "user", "content": "hi"}]},
            settings={"tools": []},
            trace={},
        )
    )
    assert events[0]["chunk"].get("toolCalls")


def test_second_iteration_invokes_vertex_once_with_tool_calls_and_outputs(monkeypatch):
    module = _load_gemini_flash_module()
    llm = module.MyLLM()

    monkeypatch.setattr(module.vertex_auth, "build_credentials", lambda plugin_config: "creds")
    monkeypatch.setattr(module.vertex_gemini, "build_generate_content_url", lambda **kwargs: "url")
    monkeypatch.setattr(module.vertex_gemini, "get_bearer_token", lambda **kwargs: "token")

    invoke_mock = MagicMock(
        return_value={"candidates": [{"content": {"parts": [{"text": "done"}]}}]}
    )
    monkeypatch.setattr(module.vertex_gemini, "invoke_generate_content", invoke_mock)

    llm.set_config(
        config={"max_output_tokens": 10, "temperature": 0.1},
        plugin_config={"project_id": "p", "location": "global"},
    )

    query = {
        "messages": [
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
                "toolOutputs": [{"callId": "c1", "output": "ok"}],
            },
            {"role": "user", "content": "summarize"},
        ]
    }

    events = list(llm.process_stream(query=query, settings={"tools": []}, trace={}))
    assert invoke_mock.call_count == 1
    assert events[0]["chunk"]["text"] == "done"

    sent_payload = invoke_mock.call_args.kwargs["payload"]
    contents = sent_payload["contents"]
    assert contents[0]["role"] == "model"
    assert contents[0]["parts"][0]["functionCall"]["name"] == "t"
    assert contents[1]["role"] == "user"
    assert contents[1]["parts"][0]["functionResponse"]["name"] == "t"


def test_process_stream_logs_and_reraises_on_empty_response(monkeypatch, caplog):
    module = _load_gemini_flash_module()
    llm = module.MyLLM()

    monkeypatch.setattr(module.vertex_auth, "build_credentials", lambda plugin_config: "creds")
    monkeypatch.setattr(module.vertex_gemini, "build_generate_content_url", lambda **kwargs: "url")
    monkeypatch.setattr(module.vertex_gemini, "get_bearer_token", lambda **kwargs: "token")
    monkeypatch.setattr(
        module.vertex_gemini,
        "invoke_generate_content",
        lambda **kwargs: {"_diagnostics": {"http_status": 200}, "candidates": []},
    )

    llm.set_config(
        config={"max_output_tokens": 10, "temperature": 0.1},
        plugin_config={"project_id": "p", "location": "global"},
    )

    with caplog.at_level("INFO"):
        with pytest.raises(module.vertex_gemini.GeminiAPIError):
            list(
                llm.process_stream(
                    query={"messages": [{"role": "user", "content": "hi"}]},
                    settings={"tools": []},
                    trace={},
                )
            )

    assert "GeminiFlash.process_stream: enter" in caplog.text
    assert "GeminiFlash.process_stream: vertex_diag" in caplog.text
