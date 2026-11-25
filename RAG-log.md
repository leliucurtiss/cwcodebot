### 2026-11-20

DEBUG: Payload keys: ['model', 'messages', 'stream', 'temperature', 
'context_files', 'top_p', 'tools', 'n', 'stream_options', 'max_tokens']

### 2026-11-15

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


### 2026-11-24

Harness added a wrapper for Claudecode compatibility with https and certs

```
# 1. Copy the certificate to the trusted folder (requires sudo)
sudo scp 10.14.16.76:/home/leliu/cwcodebot/cert.pem /usr/local/share/ca-certificates/cwcodebot.crt

# 2. Update the system certificate store
sudo update-ca-certificates
```


if you are writing a python client
```
export REQUESTS_CA_BUNDLE=/home/leliu/cwcodebot/cert.pem
export SSL_CERT_FILE=/home/leliu/cwcodebot/cert.pem
# Now run your python script
python my_client.py
````

