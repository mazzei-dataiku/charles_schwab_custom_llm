def stream_single_chunk(*, text: str, footer=None):
    """Yield a single Dataiku-compatible streamed completion.

    This intentionally does not implement token-by-token streaming.
    """

    yield {"chunk": {"text": text}}
    if footer is None:
        footer = {}
    yield {"footer": footer}
