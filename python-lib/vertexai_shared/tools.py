def dataiku_tools_to_openai(tools):
    """Translate Dataiku CompletionSettings.tools into OpenAI/vLLM tool descriptors.

    Dataiku tools are already close to the OpenAI schema:
    {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
    """

    if tools is None:
        return []
    if not isinstance(tools, list):
        raise TypeError("Invalid tools setting: tools must be a list")

    out = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise TypeError("Invalid tools setting: each tool must be an object")
        if tool.get("type") != "function":
            raise TypeError(f"Unsupported tool type: {tool.get('type')}")
        function = tool.get("function")
        if not isinstance(function, dict):
            raise TypeError("Invalid tools setting: function tool is missing function")
        name = function.get("name")
        if not name or not isinstance(name, str):
            raise TypeError("Invalid tools setting: function name must be a string")

        translated = {
            "type": "function",
            "function": {
                "name": name,
                "description": function.get("description") or "",
            },
        }
        if function.get("parameters") is not None:
            translated["function"]["parameters"] = function.get("parameters")

        out.append(translated)

    return out

