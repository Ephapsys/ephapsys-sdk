#!/usr/bin/env bash
set -euo pipefail

info() {
  echo "[INFO] $*"
}

error() {
  echo "[ERROR] $*" >&2
}

if [ -f ".env" ]; then
  info "Loading environment from .env"
  set -a && source .env && set +a
fi

PORT="${ROBOT_BRAIN_PORT:-8765}"
HOST="${ROBOT_BRAIN_HOST:-127.0.0.1}"

if ! python3 -c "import ephapsys, fastapi, uvicorn, websockets, cv2, webrtcvad, torch, transformers, soundfile" >/dev/null 2>&1; then
  error "Missing runtime dependencies in the current Python environment."
  error "Install ephapsys plus robot brain requirements first."
  exit 1
fi

exec python3 -m uvicorn robot_brain_server:app --host "$HOST" --port "$PORT" --log-level warning
