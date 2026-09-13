#!/usr/bin/env bash
set -euo pipefail

HOST="${HARNESS_HOST:-127.0.0.1}"
PORT="${HARNESS_PORT:-8000}"

exec uvicorn local_ai_harness.app:create_app \
  --factory \
  --host "$HOST" \
  --port "$PORT"

