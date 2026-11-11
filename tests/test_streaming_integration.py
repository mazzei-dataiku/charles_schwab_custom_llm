import importlib.util
from pathlib import Path
from unittest.mock import MagicMock


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_gemma_legacy_process_stream_delegates(monkeypatch):
    module = _load_module(
        "gemma_legacy_llm",
        Path(__file__).resolve().parents[1] / "python-llms" / "gemma-legacy" / "llm.py",
    )
    llm = module.MyLLM()

    llm.process = MagicMock(return_value={"text": "hi", "prompt": "p"})
    events = list(llm.process_stream({"messages": [{"role": "user", "content": "x"}]}, {}, {}))
    assert events[0] == {"chunk": {"text": "hi"}}
    assert events[1] == {"footer": {"toolCalls": []}}
    llm.process.assert_called_once()


def test_gemma_4_process_stream_delegates(monkeypatch):
    module = _load_module(
        "gemma_4_llm",
        Path(__file__).resolve().parents[1] / "python-llms" / "gemma-4" / "llm.py",
    )
    llm = module.MyLLM()

    llm.process = MagicMock(return_value={"text": "ok", "prompt": "p"})
    events = list(llm.process_stream({"messages": [{"role": "user", "content": "x"}]}, {}, {}))
    assert events[0] == {"chunk": {"text": "ok"}}
    assert events[1] == {"footer": {"toolCalls": []}}
    llm.process.assert_called_once()

