#!/usr/bin/env bash
# ============================================================
# Run Robot Agent Sample (Ephapsys SDK)
#
# This script loads environment variables from .env (if present)
# and runs the robot_agent.py demo.
# ============================================================

set -euo pipefail

MODE="${1:-run}" # run | smoke
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../../" && pwd)"
SDK_SETUP_SH="$REPO_DIR/scripts/setup.sh"

info() {
  echo "[INFO] $*"
}

warn() {
  echo "[WARN] $*"
}

error() {
  echo "[ERROR] $*" >&2
}

ensure_runtime_env() {
  local venv sdk_extras
  venv="${ROBOT_VENV:-.venv}"
  sdk_extras="${ROBOT_SDK_EXTRAS:-audio,vision,embedding}"

  if [ "${ROBOT_USE_LOCAL_SDK:-0}" = "1" ]; then
    if [ ! -d "$venv" ]; then
      info "Creating local virtualenv at $venv"
      python3 -m venv "$venv"
    fi
    # shellcheck disable=SC1090
    source "$venv/bin/activate"
    if [ ! -d "../../../sdk/python" ]; then
      error "Local SDK path not found at ../../../sdk/python"
      exit 1
    fi
    info "Installing robot sample requirements into $venv"
    PIP_DISABLE_PIP_VERSION_CHECK=1 python3 -m pip install --upgrade pip --quiet
    PIP_DISABLE_PIP_VERSION_CHECK=1 python3 -m pip install -r requirements.txt --quiet
    info "Installing local Ephapsys SDK from repo into $venv"
    PIP_DISABLE_PIP_VERSION_CHECK=1 python3 -m pip install -e "../../../sdk/python[${sdk_extras}]" --quiet
    return
  fi

  if [ ! -x "$SDK_SETUP_SH" ]; then
    error "SDK setup helper not found at $SDK_SETUP_SH"
    exit 1
  fi

  info "Ensuring latest published Ephapsys SDK in $venv"
  "$SDK_SETUP_SH" --pypi --venv "$venv" --extras "$sdk_extras"
  # shellcheck disable=SC1090
  source "$venv/bin/activate"

  if ! python3 -c "import fastapi, uvicorn, websockets" >/dev/null 2>&1; then
    info "Installing robot sample-local requirements"
    PIP_DISABLE_PIP_VERSION_CHECK=1 python3 -m pip install -r requirements.txt --quiet
  fi

  if ! python3 -c "import ephapsys, transformers, faiss, cv2, sounddevice, webrtcvad, fastapi, uvicorn, websockets" >/dev/null 2>&1; then
    error "Published SDK environment is missing required runtime dependencies."
    error "Install failed or the selected package does not include the required extras."
    exit 1
  fi
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
  python3 -m py_compile robot_agent.py robot_body.py robot_brain.py robot_brain_server.py robot_face.py robot_remote_agent.py
  echo "[CI][smoke] robot sample syntax OK."
  exit 0
fi

ensure_runtime_env

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
