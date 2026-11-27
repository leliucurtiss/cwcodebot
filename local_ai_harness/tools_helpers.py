from __future__ import annotations

import ast
import json
import re
import uuid
from typing import Any, Iterable


def _extract_assistant_text(response: dict[str, Any]) -> str:
    message = response.get("choices", [{}])[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False)


def _reason_sse(content: str) -> bytes:
    event = {"choices": [{"delta": {"reasoning_content": content}}]}
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode()


def _content_sse(content: str) -> bytes:
    event = {"choices": [{"delta": {"content": content}}]}
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode()


def _tool_calls_sse(tool_calls: Any) -> bytes:
    event = {"choices": [{"delta": {"tool_calls": tool_calls}}]}
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode()


def _build_openai_tool_call(name: str, args: Any, tool_id: str | None = None) -> str:
    if isinstance(args, str):
        try:
            parsed_args = json.loads(args)
        except json.JSONDecodeError:
            parsed_args = {"input": args}
        arguments = json.dumps(parsed_args, ensure_ascii=False)
    else:
        arguments = json.dumps(args, ensure_ascii=False)
    return json.dumps(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": tool_id or f"call_{uuid.uuid4().hex[:24]}",
                                "type": "function",
                                "function": {"name": name, "arguments": arguments},
                            }
                        ]
                    }
                }
            ]
        },
        ensure_ascii=False,
    )


def _build_openai_tool_calls(tools: list[tuple[str, Any]]) -> str:
    tool_calls = []
    for index, (name, args) in enumerate(tools):
        if isinstance(args, str):
            arguments = json.dumps({"input": args}, ensure_ascii=False)
        else:
            arguments = json.dumps(args, ensure_ascii=False)
        tool_calls.append(
            {
                "index": index,
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    return json.dumps({"choices": [{"delta": {"tool_calls": tool_calls}}]}, ensure_ascii=False)


def _split_top_level(value: str, separator: str = ",") -> list[str]:
    parts: list[str] = []
    start = 0
    quote: str | None = None
    escaped = False
    depth = 0
    for index, char in enumerate(value):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in "\"'":
            quote = char
        elif char in "[{(":
            depth += 1
        elif char in "]})":
            depth -= 1
        elif char == separator and depth == 0:
            parts.append(value[start:index].strip())
            start = index + 1
    parts.append(value[start:].strip())
    return [part for part in parts if part]


def _parse_fuzzy_value(raw: str) -> Any:
    value = raw.strip()
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value.strip("\"'")


def _parse_json_fuzzy(json_str: str) -> Any:
    value = json_str.strip()
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass

    if value.startswith("{") and value.endswith("}"):
        result: dict[str, Any] = {}
        for pair in _split_top_level(value[1:-1]):
            key, separator, raw_value = pair.partition(":")
            if not separator:
                return None
            key = key.strip().strip("\"'")
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", key):
                return None
            result[key] = _parse_json_fuzzy(raw_value) if raw_value[:1] in "[{" else _parse_fuzzy_value(raw_value)
        return result
    if value.startswith("[") and value.endswith("]"):
        parsed = [_parse_json_fuzzy(part) for part in _split_top_level(value[1:-1])]
        if any(item is None for item in parsed):
            return None
        return parsed
    return _parse_fuzzy_value(value)


def _find_matching(value: str, start: int, opening: str = "{", closing: str = "}") -> int | None:
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(start, len(value)):
        char = value[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in "\"'":
            quote = char
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index
    return None


def _tool_from_mapping(data: dict[str, Any]) -> tuple[str, Any] | None:
    if data.get("type") == "tool_use" and data.get("name"):
        return str(data["name"]), data.get("input", {})

    function = data.get("function")
    if isinstance(function, dict) and function.get("name"):
        arguments = _parse_json_fuzzy(function.get("arguments", "{}"))
        return str(function["name"]), {} if arguments is None else arguments
    if isinstance(function, str) and function:
        return function, data.get("arguments", data.get("input", data.get("parameters", {})))

    for name_key in ("name", "command", "tool", "function_name"):
        name = data.get(name_key)
        if isinstance(name, str) and name:
            args = data.get("arguments", data.get("parameters", data.get("input", data.get("args", {}))))
            if isinstance(args, str):
                args = _parse_json_fuzzy(args)
            return name, {} if args is None else args

    if len(data) == 1:
        name, args = next(iter(data.items()))
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", str(name)):
            return str(name), args
    return None


def _parse_tool_mapping(raw: str) -> tuple[str, Any, str] | None:
    parsed = _parse_json_fuzzy(raw)
    if isinstance(parsed, dict):
        tool = _tool_from_mapping(parsed)
        if tool:
            return tool[0], tool[1], ""
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                tool = _tool_from_mapping(item)
                if tool:
                    return tool[0], tool[1], ""
    return None


def _tool_match(raw_text: str) -> tuple[int, int, str, Any] | None:
    text = raw_text.lstrip()
    leading = len(raw_text) - len(text)

    patterns = (
        r"Action:\s*([A-Za-z_][\w.-]*)\s*\n+\s*Action\s+Input:\s*",
        r"TOOL_NAME:\s*([A-Za-z_][\w.-]*)\s*\n(?:\s*)BEGIN_ARG:",
    )
    action = re.match(patterns[0], text, re.IGNORECASE)
    if action:
        payload_start = action.end()
        name = action.group(1)
        first_line = text[payload_start:].splitlines()[0].strip()
        if first_line.startswith(("{", "[")):
            end = _find_matching(text, payload_start, first_line[0], "}" if first_line[0] == "{" else "]")
            if end is None:
                return None
            args = _parse_json_fuzzy(text[payload_start : end + 1])
            return leading, payload_start + len(text[payload_start : end + 1]), name, args
        value_end = text.find("\n\n", payload_start)
        value_end = len(text) if value_end < 0 else value_end
        value = text[payload_start:value_end].strip()
        return leading, payload_start + len(value), name, {"input": _parse_fuzzy_value(value)}

    line_format = re.match(patterns[1], text, re.IGNORECASE)
    if line_format:
        name = line_format.group(1)
        args: dict[str, Any] = {}
        cursor = line_format.end() - len("BEGIN_ARG:")
        pattern = re.compile(r"\s*BEGIN_ARG:\s*([A-Za-z_][\w.-]*)\s*\n(.*?)\n\s*END_ARG", re.DOTALL)
        matches = list(pattern.finditer(text, line_format.end() - len("BEGIN_ARG:")))
        if not matches:
            return None
        for match in matches:
            args[match.group(1)] = _parse_fuzzy_value(match.group(2).strip())
            cursor = match.end()
        return leading, cursor, name, args

    xml = re.match(
        r"<(?:function_name|tool_name)>\s*([^<]+?)\s*</(?:function_name|tool_name)>\s*"
        r"<(?:parameters|arguments)>\s*(.*?)\s*</(?:parameters|arguments)>",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if xml:
        args = _parse_json_fuzzy(xml.group(2))
        if args is not None:
            return leading, xml.end(), xml.group(1).strip(), args

    tag = re.search(r"<(tool_call|tool_code)>(.*?)</\1>", raw_text, re.DOTALL | re.IGNORECASE)
    if tag:
        parsed = _parse_tool_mapping(tag.group(2).strip())
        if parsed:
            return tag.start(), tag.end(), parsed[0], parsed[1]

    fence = re.search(r"```(?:tool|json)\s*\n?(.*?)```", raw_text, re.DOTALL | re.IGNORECASE)
    if fence:
        parsed = _parse_tool_mapping(fence.group(1).strip())
        if parsed:
            return fence.start(), fence.end(), parsed[0], parsed[1]

    brace = re.search(r"(?:(?:tool)\s+)?([A-Za-z_][\w.-]*)\s*(\{)", raw_text, re.IGNORECASE)
    if brace:
        start = brace.start(2)
        end = _find_matching(raw_text, start)
        if end is not None:
            args = _parse_json_fuzzy(raw_text[start : end + 1])
            if args is not None:
                return brace.start(), end + 1, brace.group(1), args

    stripped = raw_text.strip()
    if stripped.startswith(("{", "[")):
        start = len(raw_text) - len(raw_text.lstrip())
        opening = stripped[0]
        end = _find_matching(raw_text, start, opening, "}" if opening == "{" else "]")
        if end is not None:
            parsed = _parse_tool_mapping(raw_text[start : end + 1])
            if parsed:
                return start, end + 1, parsed[0], parsed[1]
    return None


def _translate_tool_call(raw_text: str) -> tuple[str | None, str]:
    stripped = raw_text.lstrip()
    if stripped.startswith("["):
        start = len(raw_text) - len(stripped)
        end = _find_matching(raw_text, start, "[", "]")
        if end is not None:
            parsed = _parse_json_fuzzy(raw_text[start : end + 1])
            if isinstance(parsed, list):
                tools = [
                    tool
                    for item in parsed
                    if isinstance(item, dict)
                    for tool in [_tool_from_mapping(item)]
                    if tool is not None
                ]
                if tools and len(tools) == len(parsed):
                    return _build_openai_tool_calls(tools), raw_text[:start] + raw_text[end + 1 :]
    match = _tool_match(raw_text)
    if not match:
        return None, raw_text
    start, end, name, args = match
    return _build_openai_tool_call(name, args), raw_text[:start] + raw_text[end:]


def translate_anthropic_to_openai_tools(anthropic_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    openai_tools = []
    for tool in anthropic_tools:
        if not isinstance(tool, dict) or not tool.get("name"):
            continue
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
        )
    return openai_tools


def translate_openai_tool_to_anthropic(openai_tool: dict[str, Any]) -> dict[str, Any]:
    function = openai_tool.get("function", {})
    arguments = function.get("arguments", "{}")
    try:
        parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.JSONDecodeError:
        parsed = {}
    return {
        "type": "tool_use",
        "id": openai_tool.get("id") or f"tool_{uuid.uuid4().hex[:24]}",
        "name": function.get("name") or "unknown_tool",
        "input": parsed if isinstance(parsed, dict) else {"value": parsed},
    }


def _content_parts_to_text(parts: Iterable[Any]) -> str:
    values = []
    for part in parts:
        if isinstance(part, str):
            values.append(part)
        elif isinstance(part, dict):
            if part.get("type") == "text":
                values.append(str(part.get("text", "")))
            elif "text" in part:
                values.append(str(part["text"]))
    return "".join(values)


def translate_anthropic_messages(messages: list[Any]) -> list[dict[str, Any]]:
    translated: list[dict[str, Any]] = []
    for message in messages:
        data = message.model_dump(exclude_none=True) if hasattr(message, "model_dump") else dict(message)
        role = str(data.get("role", "user"))
        content = data.get("content", "")
        if not isinstance(content, list):
            translated.append({"role": role, "content": content})
            continue

        text = _content_parts_to_text(content)
        tool_uses = [part for part in content if isinstance(part, dict) and part.get("type") == "tool_use"]
        tool_results = [part for part in content if isinstance(part, dict) and part.get("type") == "tool_result"]

        if tool_uses:
            translated.append(
                {
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": [
                        {
                            "id": part.get("tool_use_id") or part.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                            "type": "function",
                            "function": {
                                "name": part.get("name") or "unknown_tool",
                                "arguments": json.dumps(part.get("input", {}), ensure_ascii=False),
                            },
                        }
                        for part in tool_uses
                    ],
                }
            )
        elif tool_results:
            if text:
                translated.append({"role": "user", "content": text})
            for result in tool_results:
                result_content = result.get("content", "")
                if isinstance(result_content, list):
                    result_content = _content_parts_to_text(result_content)
                elif not isinstance(result_content, str):
                    result_content = json.dumps(result_content, ensure_ascii=False)
                translated.append(
                    {
                        "role": "tool",
                        "tool_call_id": result.get("tool_use_id") or "",
                        "content": result_content,
                    }
                )
        else:
            translated.append({"role": role, "content": text})
    return translated


def translate_text_to_anthropic_tools(raw_text: str) -> tuple[list[dict[str, Any]], str]:
    tools: list[dict[str, Any]] = []
    remaining = raw_text
    while True:
        array_start = remaining.find("[")
        if array_start >= 0:
            array_end = _find_matching(remaining, array_start, "[", "]")
            if array_end is not None:
                parsed = _parse_json_fuzzy(remaining[array_start : array_end + 1])
                if isinstance(parsed, list):
                    mapped = [
                        tool
                        for item in parsed
                        if isinstance(item, dict)
                        for tool in [_tool_from_mapping(item)]
                        if tool is not None
                    ]
                    if mapped and len(mapped) == len(parsed):
                        tools.extend(
                            {
                                "type": "tool_use",
                                "id": f"tool_{uuid.uuid4().hex[:24]}",
                                "name": name,
                                "input": args if isinstance(args, dict) else {"value": args},
                            }
                            for name, args in mapped
                        )
                        remaining = remaining[:array_start] + remaining[array_end + 1 :]
                        continue
        match = _tool_match(remaining)
        if not match:
            break
        start, end, name, args = match
        tools.append(
            {
                "type": "tool_use",
                "id": f"tool_{uuid.uuid4().hex[:24]}",
                "name": name,
                "input": args if isinstance(args, dict) else {"value": args},
            }
        )
        remaining = remaining[:start] + remaining[end:]
    return tools, remaining.strip()


def anthropic_start_tool_block(index: int, tool_id: str, name: str) -> bytes:
    payload = {
        "type": "content_block_start",
        "index": index,
        "content_block": {"type": "tool_use", "id": tool_id, "name": name, "input": {}},
    }
    return f"event: content_block_start\ndata: {json.dumps(payload)}\n\n".encode()


def anthropic_tool_delta(index: int, partial_json: str) -> bytes:
    payload = {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "input_json_delta", "partial_json": partial_json},
    }
    return f"event: content_block_delta\ndata: {json.dumps(payload)}\n\n".encode()


def anthropic_stop_block(index: int) -> bytes:
    payload = {"type": "content_block_stop", "index": index}
    return f"event: content_block_stop\ndata: {json.dumps(payload)}\n\n".encode()


class OpenAISSEDecoder:
    def __init__(self) -> None:
        self.buffer = b""

    def feed(self, chunk: bytes) -> list[str]:
        self.buffer += chunk
        events: list[str] = []
        while True:
            line_end = self.buffer.find(b"\n")
            if line_end < 0:
                break
            raw_line = self.buffer[:line_end]
            self.buffer = self.buffer[line_end + 1 :]
            line = raw_line.rstrip(b"\r")
            if line.startswith(b"data:"):
                events.append(line[5:].lstrip().decode("utf-8"))
        return events
