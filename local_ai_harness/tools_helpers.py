import json
import re
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
    print(f"DEBUG: translate_anthropic_to_openai_tools INPUT: {str(json.dumps(anthropic_tools, indent=2)[:50])}")
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
    print(f"DEBUG: translate_anthropic_to_openai_tools OUTPUT: {str(json.dumps(openai_tools, indent=2))[:50]}")
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

    return {
        "type": "tool_use",
        "id": openai_tool.get("id"),
        "name": func.get("name"),
        "input": args
    }

def translate_text_to_anthropic_tool(raw_text: str) -> tuple[dict[str, Any] | None, str]:
    """
    Detects tool calls in text (like <tool_code>{...}</tool_code> or tool_name{...})
    and translates them to Anthropic tool_use blocks.
    """
    # 1. Check for <tool_code> tags first
    tag_match = re.search(r'<tool_code>(.*?)</tool_code>', raw_text, re.DOTALL)
    if tag_match:
        content = tag_match.group(1).strip()
        try:
            # Claude Code often sends multiple JSON objects per line in <tool_code>
            # We take the first one for this translation
            first_line = content.split('\n')[0]
            tool_data = json.loads(first_line)
            # tool_data is usually {"tool_name": {args}}
            if isinstance(tool_data, dict) and len(tool_data) == 1:
                name, args = list(tool_data.items())[0]
                return {
                    "type": "tool_use",
                    "id": f"tool_{uuid.uuid4().hex[:12]}",
                    "name": name,
                    "input": args
                }, raw_text.replace(tag_match.group(0), "")
        except Exception:
            pass

    # 2. Fallback to the tool_name{...} pattern (similar to existing _translate_tool_call)
    # ... (implementation of the regex match) ...
    
    return None, raw_text