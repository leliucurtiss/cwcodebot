import json
import re
import uuid
from typing import Any

def _extract_assistant_text(response: dict[str, Any]) -> str:
    try:
        message = response.get("choices", [{}])[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return ""

def _reason_sse(content: str) -> bytes:
    event = {"choices": [{"delta": {"reasoning_content": content}}]}
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8")

def _content_sse(content: str) -> bytes:
    event = {"choices": [{"delta": {"content": content}}]}
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8")

def _tool_calls_sse(tool_calls: Any) -> bytes:
    event = {"choices": [{"delta": {"tool_calls": tool_calls}}]}
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8")

def _build_openai_tool_call(name: str, args: Any) -> str:
    """Helper to format a tool call as an OpenAI SSE delta."""
    return json.dumps({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": f"call_{re.sub(r'[^a-z0-9]', '', name)[:12]}", 
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args) if isinstance(args, (dict, list)) else args
                    }
                }]
            }
        }]
    }, ensure_ascii=False)

def _translate_tool_call(raw_text: str) -> tuple[str | None, str]:
    """
    Translates MaaS text format into OpenAI tool call SSE.
    Returns (translated_json_or_none, remaining_text).
    """
    # --- Strategy 3: tool_name{json_args} ---
    match = re.match(r'^(?:\s*tool\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*(\{)', raw_text, re.IGNORECASE)
    if match:
        tool_name = match.group(1)
        brace_start = match.start(2)
        count = 0
        for i in range(brace_start, len(raw_text)):
            if raw_text[i] == '{':
                count += 1
            elif raw_text[i] == '}':
                count -= 1
                if count == 0:
                    json_str = raw_text[brace_start : i+1]
                    try:
                        args = json.loads(json_str)
                        return _build_openai_tool_call(tool_name, args), raw_text[i+1:]
                    except json.JSONDecodeError:
                        try:
                            inner = json_str[1:-1].strip()
                            parsed_args = {}
                            kv_pairs = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|true|false|null|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|[^{},]+)', inner)
                            for k, v in kv_pairs:
                                v = v.strip()
                                if v.startswith('"') and v.endswith('"'): v = v[1:-1]
                                elif v.startswith("'") and v.endswith("'"): v = v[1:-1]
                                elif v == 'true': v = True
                                elif v == 'false': v = False
                                elif v == 'null': v = None
                                parsed_args[k] = v
                            return _build_openai_tool_call(tool_name, parsed_args), raw_text[i+1:]
                        except Exception:
                            pass
    return None, raw_text

def translate_anthropic_to_openai_tools(anthropic_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Translates a list of tools from Anthropic format to OpenAI format.
    """
    # print(f"TRACE: translate_anthropic_to_openai_tools INPUT: {str(json.dumps(anthropic_tools, indent=2)[:50])}")
    openai_tools = []
    for tool in anthropic_tools:
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool.get("name"),
                "description": tool.get("description"),
                "parameters": tool.get("input_schema", {}),
            }
        }
        openai_tools.append(openai_tool)
    print(f"TRACE: translate_anthropic_to_openai_tools OUTPUT: {str(json.dumps(openai_tools, indent=2))[:50]}")
    return openai_tools

def translate_openai_tool_to_anthropic(openai_tool: dict[str, Any]) -> dict[str, Any]:
    """
    Translates a single OpenAI tool call into an Anthropic tool_use block.
    """
    func = openai_tool.get("function", {})
    args_str = func.get("arguments", "{}")
    
    try:
        args = json.loads(args_str) if isinstance(args_str, str) else args_str
    except json.JSONDecodeError:
        args = {"error": "failed to parse arguments", "raw": args_str}

    print(f"TRACE: translate_openai_tool_to_anthropic INPUT: {str(args)[:50]}")

    return {
        "type": "tool_use",
        "id": openai_tool.get("id"),
        "name": func.get("name"),
        "input": args
    }

def translate_text_to_anthropic_tools(raw_text: str) -> tuple[list[dict[str, Any]], str]:
    """
    Detects tool calls in text (like <tool_code>{...}</tool_code>, tool_name{...}, or JSON arrays)
    and translates them to a list of Anthropic tool_use blocks.
    """
    tools = []
    remaining_text = raw_text

    # 1. Check for JSON array format [ {...}, {...} [ {...}, {...} ]
    # Find the first '[' and then find its matching ' '[' and then find its matching ']'
    print(f" {raw_text}",end="")
    start_idx = remaining_text.find('[')
    if start_idx != -1:
        count = 0
        for i in range(start_idx, len(remaining_text)):
            if remaining_text[i] == '[': count += 1
            elif remaining_text[i] == ']':
                count -= 1
                if count == 0:
                    json_str = remaining_text[start_idx : i+1]
                    try:
                        tool_list = json.loads(json_str)
                        if isinstance(tool_list, list):
                            found_any_tool = False
                            extracted_text = []
                            for item in tool_list:
                                if not isinstance(item, dict): continue
                                if item.get("type") == "tool_use":
                                    tools.append({
                                        "type": "tool_use",
                                        "id": item.get("id", f"tool_{uuid.uuid4().hex[:12]}"),
                                        "name": item.get("name"),
                                        "input": item.get("input", {})
                                    })
                                    found_any_tool = True
                                elif item.get("type") == "function" or "function" in item:
                                    tools.append(translate_openai_tool_to_anthropic(item))
                                    found_any_tool = True
                                elif "tool" in item:
                                    tools.append({
                                        "type": "tool_use",
                                        "id": f"tool_{uuid.uuid4().hex[:12]}",
                                        "name": item["tool"],
                                        "input": item.get("parameters", {})
                                    })
                                    found_any_tool = True
                                elif item.get("type") == "text":
                                    extracted_text.append(item.get("text", ""))

                            if found_any_tool:
                                replacement = " ".join(extracted_text)
                                remaining_text = remaining_text.replace(json_str, replacement)
                                return tools, remaining_text
                    except json.JSONDecodeError:
                        pass

    # 2. Check for <tool_code> tags (Now supports multi-line)
    tag_match = re.search(r'<tool_code>(.*?)</tool_code>', remaining_text, re.DOTALL)
    if tag_match:
        content = tag_match.group(1).strip()
        # Use fuzzy parser on the entire block instead of just the first line
        tool_data = _parse_json_fuzzy(content)
        if isinstance(tool_data, dict) and len(tool_data) == 1:
            name, args = list(tool_data.items())[0]
            tools.append({
                "type": "tool_use",
                "id": f"tool_{uuid.uuid4().hex[:12]}",
                "name": name,
                "input": args
            })
            remaining_text = remaining_text.replace(tag_match.group(0), "")
            return tools, remaining_text

    # 3. Fallback to tool_name{...} pattern (Now uses re.search for conversational filler)
    match = re.search(r'(?:\s*tool\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*(\{)', remaining_text, re.IGNORECASE)
    if match:
        tool_name = match.group(1)
        brace_start = match.start(2)
        count = 0
        for i in range(brace_start, len(remaining_text)):
            if remaining_text[i] == '{': count += 1
            elif remaining_text[i] == '}':
                count -= 1
                if count == 0:
                    json_str = remaining_text[brace_start : i+1]
                    args = _parse_json_fuzzy(json_str)
                    if args is not None:
                        tools.append({
                            "type": "tool_use",
                            "id": f"tool_{uuid.uuid4().hex[:12]}",
                            "name": tool_name,
                            "input": args
                        })
                        # Remove the tool call and everything before it to clean up the text
                        remaining_text = remaining_text[i+1:]
                        return tools, remaining_text

    return tools, remaining_text

def anthropic_start_tool_block(index: int, tool_id: str, name: str) -> bytes:
    """Generates the content_block_start event for a tool use."""
    payload = {
        "type": "content_block_start",
        "index": index,
        "content_block": {
            "type": "tool_use",
            "id": tool_id,
            "name": name,
            "input": {}
        }
    }
    return f"event: content_block_start\ndata: {json.dumps(payload)}\n\n".encode("utf-8")

def anthropic_tool_delta(index: int, partial_json: str) -> bytes:
    """Generates the content_block_delta event for tool input."""
    payload = {
        "type": "content_block_delta",
        "index": index,
        "delta": {
            "type": "input_json_delta",
            "partial_json": partial_json
        }
    }
    return f"event: content_block_delta\ndata: {json.dumps(payload)}\n\n".encode("utf-8")

def anthropic_stop_block(index: int) -> bytes:
    """Generates the content_block_stop event."""
    payload = {
        "type": "content_block_stop",
        "index": index
    }
    return f"event: content_block_stop\ndata: {json.dumps(payload)}\n\n".encode("utf-8")

def _parse_json_fuzzy(json_str: str) -> Any:
    """Attempts to parse JSON strictly, then falls back to a fuzzy regex-based parser."""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            # Remove outer braces if present
            inner = json_str.strip()
            if inner.startswith('{') and inner.endswith('}'):
                inner = inner[1:-1].strip()

            parsed_args = {}
            # Regex to find key:value pairs, handling strings, booleans, and numbers
            kv_pairs = re.findall(r'''([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|true|false|null|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|[^{},]+)''', inner)
            for k, v in kv_pairs:
                v = v.strip()
                if v.startswith('"') and v.endswith('"'): v = v[1:-1]
                elif v.startswith("'") and v.endswith("'"): v = v[1:-1]
                elif v == 'true': v = True
                elif v == 'false': v = False
                elif v == 'null': v = None
                parsed_args[k] = v
            return parsed_args
        except Exception:
            return None
