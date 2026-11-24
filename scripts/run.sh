#!/usr/bin/env bash
set -euo pipefail

HOST="${HARNESS_HOST:-0.0.0.0}"
PORT="${HARNESS_PORT:-8000}"
SSL_KEY="${HARNESS_SSL_KEY:-cert.pem}"
SSL_CERT="${HARNESS_SSL_CERT:-key.pem}"

# Check if SSL files exist before adding them to uvicorn command
SSL_ARGS=""
if [[ -f "$SSL_KEY" && -f "$SSL_CERT" ]]; then
  SSL_ARGS="--ssl-keyfile $SSL_KEY --ssl-certfile $SSL_CERT"
fi

exec uvicorn --reload local_ai_harness.app:create_app \
  --factory \
  --host "$HOST" \
  --port "$PORT" \
  $SSL_ARGS \
  --reload-exclude "tests/*" > harness.log 2>&1
#  --reload > harness.log 2>&1

