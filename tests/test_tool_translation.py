import json

import pytest

from local_ai_harness.app import _translate_tool_call
from local_ai_harness.tools_helpers import translate_text_to_anthropic_tools


@pytest.mark.parametrize(
    ("raw", "expected_tool", "expected_args", "remaining"),
    [
        (
            'Action: read_file\nAction Input: {"path": "test.txt"}\n\nNext step...',
            "read_file",
            {"path": "test.txt"},
            "\n\nNext step...",
        ),
        (
            "Action: list_dir\nAction Input: /home/user\n\nNext step...",
            "list_dir",
            {"input": "/home/user"},
            "\n\nNext step...",
        ),
        (
            '<function_name>read_file</function_name><parameters>{"path": "test.txt"}</parameters>',
            "read_file",
            {"path": "test.txt"},
            "",
        ),
        (
            '<tool_call>{"command": "list_dir", "parameters": {"path": "/home"}}</tool_call>',
            "list_dir",
            {"path": "/home"},
            "",
        ),
        (
            '```tool\n{"name": "grep_search", "arguments": {"query": "foo"}}\n```',
            "grep_search",
            {"query": "foo"},
            "",
        ),
        (
            '{"command": "read_file", "parameters": {"path": "main.py"}}',
            "read_file",
            {"path": "main.py"},
            "",
        ),
        (
            "TOOL_NAME: list_dir\nBEGIN_ARG: path\n/home/user\nEND_ARG",
            "list_dir",
            {"path": "/home/user"},
            "",
        ),
        ('read_file{"path": "test.py"} trailing', "read_file", {"path": "test.py"}, " trailing"),
        (
            "list_dir{path: /home/user, recursive: true}",
            "list_dir",
            {"path": "/home/user", "recursive": True},
            "",
        ),
        (
            "I will call read_file{path: C:\\repo\\main.py, line: 4} now",
            "read_file",
            {"path": "C:\\repo\\main.py", "line": 4},
            "I will call  now",
        ),
    ],
)
def test_translate_tool_call_formats(raw, expected_tool, expected_args, remaining):
    translated, actual_remaining = _translate_tool_call(raw)

    assert translated is not None
    call = json.loads(translated)["choices"][0]["delta"]["tool_calls"][0]
    assert call["function"]["name"] == expected_tool
    assert json.loads(call["function"]["arguments"]) == expected_args
    assert actual_remaining == remaining


def test_translate_tool_call_rejects_incomplete_input():
    raw = "TOOL_NAME: read_file\nBEGIN_ARG: filePath\n/path"
    assert _translate_tool_call(raw) == (None, raw)


def test_translate_multiple_text_tools_to_anthropic_blocks():
    tools, remaining = translate_text_to_anthropic_tools(
        '[{"type":"tool_use","name":"read_file","input":{"path":"a.py"}},'
        '{"type":"function","function":{"name":"list_dir","arguments":"{\\"path\\":\\".\\"}"}}]'
    )

    assert remaining == ""
    assert [tool["name"] for tool in tools] == ["read_file", "list_dir"]
    assert [tool["input"] for tool in tools] == [{"path": "a.py"}, {"path": "."}]


def test_translate_multiple_json_tools_to_openai_delta():
    translated, remaining = _translate_tool_call(
        '[{"name":"read_file","arguments":{"path":"a.py"}},'
        '{"name":"list_dir","arguments":{"path":"."}}]'
    )

    calls = json.loads(translated)["choices"][0]["delta"]["tool_calls"]
    assert remaining == ""
    assert [call["index"] for call in calls] == [0, 1]
    assert [call["function"]["name"] for call in calls] == ["read_file", "list_dir"]
