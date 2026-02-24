#!/usr/bin/env bash
# ============================================================
# Run Robot Agent Sample (Ephapsys SDK)
#
# This script loads environment variables from .env (if present)
# and runs the robot_agent.py demo.
# ============================================================

set -euo pipefail

# Load .env if available (preserve spaces by using set -a)
if [ -f ".env" ]; then
  echo "[INFO] Loading environment from .env"
  set -a && source .env && set +a
fi

# Config defaults
BASE_URL=${AOC_BASE_URL:-${AOC_API_URL:-${AOC_API:-"http://localhost:7001"}}}
ORG_ID=${AOC_ORG_ID:-""}
BOOTSTRAP_TOKEN=${AOC_PROVISIONING_TOKEN:-""}
AGENT_ID=${AGENT_TEMPLATE_ID:-"agent_robot"}

# Handle anchor properly
if [ -z "${PERSONALIZE_ANCHOR:-}" ]; then
  echo "[WARN] PERSONALIZE_ANCHOR not set, defaulting to tpm"
  export PERSONALIZE_ANCHOR="tpm"
fi

if [ -z "$ORG_ID" ] || [ -z "$BOOTSTRAP_TOKEN" ]; then
  echo "[ERROR] Missing credentials. Set AOC_ORG_ID + AOC_PROVISIONING_TOKEN."
  exit 1
fi

# Bootstrap local venv to avoid system Python conflicts (PEP 668)
VENV=".venv"
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1090
source "$VENV/bin/activate"

echo "[INFO] Starting Robot Agent..."
echo "  BASE_URL: $BASE_URL"
echo "  AGENT_ID: $AGENT_ID"
echo "  ANCHOR:   $PERSONALIZE_ANCHOR"

PIP_DISABLE_PIP_VERSION_CHECK=1 python3 -m pip install --upgrade pip --quiet
PIP_DISABLE_PIP_VERSION_CHECK=1 python3 -m pip install -r requirements.txt --quiet
# Install local Ephapsys SDK (editable) for TrustedAgent
SDK_PATH="../../../SDK/python"
if [ ! -d "$SDK_PATH" ]; then
  echo "[ERROR] SDK path not found at $SDK_PATH"
  exit 1
fi
PIP_DISABLE_PIP_VERSION_CHECK=1 python3 -m pip install -e "$SDK_PATH" --quiet


# Enable Python faulthandler output for native crashes; toggle verbose audio debug with AUDIO_DEBUG=1
# Constrain torch/BLAS threads to reduce native crashes in SpeechT5 init
PYTHONFAULTHANDLER=1 \
  AUDIO_DEBUG=${AUDIO_DEBUG:-0} \
  OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} \
  MKL_NUM_THREADS=${MKL_NUM_THREADS:-1} \
  TORCH_NUM_THREADS=${TORCH_NUM_THREADS:-1} \
  python3 robot_agent.py 
