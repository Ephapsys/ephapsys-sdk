#!/usr/bin/env bash
# ============================================================
# Run HelloWorld Agent Sample (Ephapsys SDK)
#
# This script loads environment variables from .env (if present)
# and runs the helloworld_agent.py demo.
# ============================================================

set -euo pipefail

# Load .env if available
if [ -f ".env" ]; then
  # echo "[INFO] Loading environment from .env"
  # shellcheck disable=SC2046
  set -a && source .env && set +a
fi

# Config defaults
BASE_URL=${AOC_API_URL:-""}
API_KEY=${AOC_API_KEY:-""}
ORG_ID=${AOC_ORG_ID:-""}
BOOTSTRAP_TOKEN=${AOC_BOOTSTRAP_TOKEN:-""}
AGENT_ID=${AGENT_TEMPLATE_ID:-"agent_helloworld"}

# Handle anchor properly
if [ -z "${PERSONALIZE_ANCHOR:-}" ]; then
  echo "[WARN] PERSONALIZE_ANCHOR not set, defaulting to tpm"
  export PERSONALIZE_ANCHOR="tpm"
fi

if [ -z "$API_KEY" ] && { [ -z "$ORG_ID" ] || [ -z "$BOOTSTRAP_TOKEN" ]; }; then
  echo "[ERROR] Missing credentials. Set AOC_ORG_ID + AOC_BOOTSTRAP_TOKEN (preferred) or legacy AOC_API_KEY."
  exit 1
fi

echo "[INFO] Starting HelloWorld Agent..."
echo "  BASE_URL: $BASE_URL"
echo "  AGENT_ID: $AGENT_ID"
echo "  ANCHOR:   $PERSONALIZE_ANCHOR"

python3 helloworld_agent.py
