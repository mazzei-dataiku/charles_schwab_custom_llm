from vertexai_shared.streaming import stream_single_chunk


def test_stream_single_chunk_yields_chunk_then_footer():
    events = list(stream_single_chunk(text="hello", footer={"toolCalls": []}))
    assert events == [{"chunk": {"text": "hello"}}, {"footer": {"toolCalls": []}}]


def test_stream_single_chunk_default_footer_empty_dict():
    events = list(stream_single_chunk(text="hello"))
    assert events == [{"chunk": {"text": "hello"}}, {"footer": {}}]

