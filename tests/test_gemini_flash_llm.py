import importlib.util
from pathlib import Path
from unittest.mock import MagicMock


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
        plugin_config={"project_id": "p", "location": "global"},
    )
    assert llm._config["model_id"] == "gemini-3.8-flash"
    assert llm._plugin_config["location"] == "global"


def test_process_builds_payload_and_returns_text(monkeypatch):
    module = _load_gemini_flash_module()
    llm = module.MyLLM()

    monkeypatch.setattr(module.vertex_auth, "build_credentials", lambda plugin_config: "creds")

    monkeypatch.setattr(module.vertex_gemini, "build_generate_content_url", lambda **kwargs: "url")
    monkeypatch.setattr(module.vertex_gemini, "get_bearer_token", lambda **kwargs: "token")

    invoke_mock = MagicMock(return_value={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})
    monkeypatch.setattr(module.vertex_gemini, "invoke_generate_content", invoke_mock)

    llm.set_config(
        config={"max_output_tokens": 10, "temperature": 0.1},
        plugin_config={"project_id": "p", "location": "global"},
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

