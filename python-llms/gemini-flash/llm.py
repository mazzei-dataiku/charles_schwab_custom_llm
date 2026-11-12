# This file is the implementation of custom LLM gemini-flash
from dataiku.llm.python import BaseLLM


class MyLLM(BaseLLM):
    def __init__(self):
        pass

    def set_config(self, config: dict, plugin_config: dict) -> None:
        pass

    # define aprocess instead of process for async support
    def process(self, query, settings, trace):
        messages = query["messages"]

        # Add code here, for LLM completion, tool execution, etc...

        return {
            "text": "the custom LLM answer",
            "promptTokens": 46, # Optional
            "completionTokens": 10, # Optional
            "estimatedCost": 0.13, # Optional
            "toolCalls": [],
        }

    # define aprocess_stream instead of process_stream for async support
    def process_stream(self, query, settings, trace):
        messages = query["messages"]

        # Add code here, for LLM completion, tool execution, etc...

        yield {
            "chunk": {
                "text": "The ",
            }
        }
        yield {
            "chunk": {
                "text": "answer.",
            }
        }
        yield {
            "footer": {
                "promptTokens": 46,  # Optional
                "completionTokens": 10,  # Optional
                "estimatedCost": 0.13,  # Optional
                "toolCalls": [],
            }
        }