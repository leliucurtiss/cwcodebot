#!/usr/bin/env bash
set -euo pipefail

HOST="${HARNESS_HOST:-0.0.0.0}"
PORT="${HARNESS_PORT:-8000}"

exec uvicorn --reload local_ai_harness.app:create_app \
  --factory \
  --host "$HOST" \
  --port "$PORT" \
  --reload-exclude "tests/*" > harness.log 2>&1
#  --reload > harness.log 2>&1

