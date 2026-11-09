### Run uvicorn and save all output (stdout and stderr) to harness.log
exec uvicorn local_ai_harness.app:create_app \
--factory \
--host "$HOST" \
--port "$PORT" > harness.log 2>&1

tail -f harness.log

### How to avoid wrong tool call  
  
CRITICAL: TOOL CALLING FORMAT
All tool calls must be returned as a JSON object in the following format:

```
{
  "tool_calls": [
    {
      "id": "call_unique_id",
      "type": "function",
      "function": {
        "name": "tool_name",
        "arguments": "{\"arg_name\": \"value\"}"
      }
    }
  ]
}
```