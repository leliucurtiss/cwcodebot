#!/usr/bin/env bash
set -euo pipefail

printf $BASE_URL printf '\n'
printf $DEVSTRAL_API_KEY '\n'

curl -sS "$BASE_URL/health" \
  -H "Authorization: Bearer $DEVSTRAL_API_KEY" 

printf '\n'

curl -sS "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $DEVSTRAL_API_KEY" \
  -d '{
    "model": "devstral",
    "messages": [
      {"role": "user", "content": "Reply with one short sentence saying the harness works."}
    ],
    "stream": true
  }'
printf '\n'

