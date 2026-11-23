
import json
from local_ai_harness.app import _translate_tool_call

def test_translation(name, input_text, expected_tool, expected_args):
    print(f"Testing {name}...")
    result, remaining = _translate_tool_call(input_text)
    if result is None:
        print(f"  FAILED: No translation found")
        return False
    
    try:
        data = json.loads(result)
        actual_tool = data["choices"][0]["delta"]["tool_calls"][0]["function"]["name"]
        actual_args = json.loads(data["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"])
        
        if actual_tool == expected_tool and actual_args == expected_args:
            print(f"  PASSED")
            return True
        else:
            print(f"  FAILED: Expected {expected_tool}({expected_args}), got {actual_tool}({actual_args})")
            return False
    except Exception as e:
        print(f"  FAILED: Error parsing result: {e}")
        return False

def main():
    tests = [
        (
            "ReAct Action/Action Input",
            "Action: read_file\nAction Input: {\"path\": \"test.txt\"}\n\nNext step...",
            "read_file",
            {"path": "test.txt"}
        ),
        (
            "ReAct Action/Action Input raw string",
            "Action: list_dir\nAction Input: /home/user\n\nNext step...",
            "list_dir",
            {"input": "/home/user"}
        ),
        (
            "XML function/parameters",
            "<function_name>read_file</function_name><parameters>{\"path\": \"test.txt\"}</parameters>",
            "read_file",
            {"path": "test.txt"}
        ),
        (
            "XML tool_call JSON",
            "<tool_call>{\"command\": \"list_dir\", \"parameters\": {\"path\": \"/home\"}}</tool_call>",
            "list_dir",
            {"path": "/home"}
        ),
        (
            "Markdown tool block",
            "```tool\n{\"name\": \"grep_search\", \"arguments\": {\"query\": \"foo\"}}\n```",
            "grep_search",
            {"query": "foo"}
        ),
        (
            "Direct JSON",
            '{"command": "read_file", "parameters": {"path": "main.py"}}',
            "read_file",
            {"path": "main.py"}
        ),
        (
            "Line-by-line format",
            "TOOL_NAME: list_dir\nBEGIN_ARG: path\n/home/user\nEND_ARG",
            "list_dir",
            {"path": "/home/user"}
        ),
        (
            "Brace format standard",
            "read_file{\"path\": \"test.py\"}",
            "read_file",
            {"path": "test.py"}
        ),
        (
            "Brace format pseudo-JSON",
            "list_dir{path: /home/user, recursive: true}",
            "list_dir",
            {"path": "/home/user", "recursive": True}
        ),
    ]

    passed = 0
    for name, text, tool, args in tests:
        if test_translation(name, text, tool, args):
            passed += 1
    
    print(f"\nSummary: {passed}/{len(tests)} tests passed.")
    if passed != len(tests):
        exit(1)

if __name__ == "__main__":
    main()
