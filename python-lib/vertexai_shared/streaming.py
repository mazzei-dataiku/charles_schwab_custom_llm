def stream_single_chunk(*, text: str, footer=None):
    """Yield a single Dataiku-compatible streamed completion.

    This intentionally does not implement token-by-token streaming.
    """

    yield {"chunk": {"text": text}}
    if footer is None:
        footer = {}
    yield {"footer": footer}


def stream_tool_calls(*, tool_calls, footer=None):
    """Yield tool calls as a single Dataiku streamed content chunk."""

    yield {"chunk": {"toolCalls": tool_calls}}
    if footer is None:
        footer = {}
    yield {"footer": footer}
