#!/usr/bin/env bash
# ============================================================
# Run Robot Agent Sample (Ephapsys SDK)
#
# This script loads environment variables from .env (if present)
# and runs the robot_agent.py demo.
# ============================================================

set -euo pipefail

MODE="${1:-run}" # run | smoke

info() {
  echo "[INFO] $*"
}

warn() {
  echo "[WARN] $*"
}

error() {
  echo "[ERROR] $*" >&2
}

# Load .env if available (preserve spaces by using set -a)
if [ -f ".env" ]; then
  info "Loading environment from .env"
  set -a && source .env && set +a
fi

# Config defaults
BASE_URL=${AOC_BASE_URL:-${AOC_API_URL:-${AOC_API:-"http://localhost:7001"}}}
ORG_ID=${AOC_ORG_ID:-""}
BOOTSTRAP_TOKEN=${AOC_PROVISIONING_TOKEN:-""}
AGENT_ID=${AGENT_TEMPLATE_ID:-"agent_robot"}

# Handle anchor properly
if [ -z "${PERSONALIZE_ANCHOR:-}" ]; then
  warn "PERSONALIZE_ANCHOR not set, defaulting to tpm"
  export PERSONALIZE_ANCHOR="tpm"
fi

if [ -z "$ORG_ID" ] || [ -z "$BOOTSTRAP_TOKEN" ]; then
  error "Missing credentials. Set AOC_ORG_ID + AOC_PROVISIONING_TOKEN."
  exit 1
fi

if [ "$MODE" = "smoke" ] || [ "${SAMPLE_CI_SMOKE:-0}" = "1" ]; then
  echo "[CI][smoke] Robot env validation OK."
  echo "[CI][smoke] BASE_URL=$BASE_URL AGENT_ID=$AGENT_ID ANCHOR=$PERSONALIZE_ANCHOR"
  python3 -m py_compile robot_agent.py robot_body.py robot_brain.py robot_brain_server.py robot_face.py
  echo "[CI][smoke] robot sample syntax OK."
  exit 0
fi

SDK_PATH="../../../sdk/python"
if [ -d "$SDK_PATH" ] && [ "${ROBOT_USE_LOCAL_SDK:-}" = "" ]; then
  export ROBOT_USE_LOCAL_SDK=1
fi
if ! python3 -c "import fastapi, uvicorn, websockets" >/dev/null 2>&1; then
  info "Installing robot sample-local requirements"
  PIP_DISABLE_PIP_VERSION_CHECK=1 python3 -m pip install -r requirements.txt --quiet
fi

if ! python3 -c "import ephapsys, transformers, faiss, cv2, sounddevice, webrtcvad, fastapi, uvicorn, websockets" >/dev/null 2>&1; then
  if [ "${ROBOT_USE_LOCAL_SDK:-0}" = "1" ]; then
    VENV="${ROBOT_VENV:-.venv}"
    if [ ! -d "$VENV" ]; then
      info "Creating local virtualenv at $VENV"
      python3 -m venv "$VENV"
    fi
    # shellcheck disable=SC1090
    source "$VENV/bin/activate"
    if [ ! -d "$SDK_PATH" ]; then
      error "Local SDK path not found at $SDK_PATH"
      exit 1
    fi
    info "Installing robot sample requirements into $VENV"
    PIP_DISABLE_PIP_VERSION_CHECK=1 python3 -m pip install --upgrade pip --quiet
    PIP_DISABLE_PIP_VERSION_CHECK=1 python3 -m pip install -r requirements.txt --quiet
    info "Installing local Ephapsys SDK from repo into $VENV"
    PIP_DISABLE_PIP_VERSION_CHECK=1 python3 -m pip install -e "${SDK_PATH}[audio,vision,embedding]" --quiet
  else
    error "Missing runtime dependencies in the current Python environment."
    error "Run: pip install \"ephapsys[audio,vision,embedding]\" && pip install -r requirements.txt"
    error "For repo-local development only, rerun with ROBOT_USE_LOCAL_SDK=1."
    exit 1
  fi
fi

info "Starting Robot Agent..."
if [ -n "${AGENT_TEMPLATE_NAME:-}" ]; then
  echo "  TEMPLATE: $AGENT_TEMPLATE_NAME"
fi

# Enable Python faulthandler output for native crashes; toggle verbose audio debug with AUDIO_DEBUG=1
# Constrain torch/BLAS threads to reduce native crashes in SpeechT5 init
PYTHONFAULTHANDLER=1 \
  AUDIO_DEBUG=${AUDIO_DEBUG:-0} \
  OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} \
  MKL_NUM_THREADS=${MKL_NUM_THREADS:-1} \
  TORCH_NUM_THREADS=${TORCH_NUM_THREADS:-1} \
  python3 robot_agent.py 
