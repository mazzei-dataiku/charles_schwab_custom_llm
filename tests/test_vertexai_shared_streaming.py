from vertexai_shared.streaming import stream_single_chunk, stream_tool_calls


def test_stream_single_chunk_yields_chunk_then_footer():
    events = list(stream_single_chunk(text="hello", footer={"toolCalls": []}))
    assert events == [{"chunk": {"text": "hello"}}, {"footer": {"toolCalls": []}}]


def test_stream_single_chunk_default_footer_empty_dict():
    events = list(stream_single_chunk(text="hello"))
    assert events == [{"chunk": {"text": "hello"}}, {"footer": {}}]


def test_stream_tool_calls_yields_chunk_then_footer():
    tool_calls = [{"id": "call_1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]
    events = list(stream_tool_calls(tool_calls=tool_calls, footer={}))
    assert events == [{"chunk": {"toolCalls": tool_calls}}, {"footer": {}}]
