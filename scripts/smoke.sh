#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${HARNESS_BASE_URL:-http://127.0.0.1:8000}"

curl -sS "$BASE_URL/health"
printf '\n'

curl -sS "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "devstral",
    "messages": [
      {"role": "user", "content": "Reply with one short sentence saying the harness works."}
    ],
    "stream": false
  }'
printf '\n'

