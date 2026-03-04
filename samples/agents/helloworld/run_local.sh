#!/usr/bin/env bash
# ============================================================
# Run HelloWorld Agent Sample (Ephapsys SDK)
#
# This script loads environment variables from .env (if present)
# and runs the helloworld_agent.py demo.
# ============================================================

set -euo pipefail

MODE="${1:-run}" # run | smoke | oneshot

# Load .env if available
if [ -f ".env" ]; then
  # echo "[INFO] Loading environment from .env"
  # shellcheck disable=SC2046
  set -a && source .env && set +a
fi

# Config defaults
BASE_URL=${AOC_BASE_URL:-${AOC_API_URL:-${AOC_API:-"http://localhost:7001"}}}
ORG_ID=${AOC_ORG_ID:-""}
BOOTSTRAP_TOKEN=${AOC_PROVISIONING_TOKEN:-""}
AGENT_ID=${AGENT_TEMPLATE_ID:-"agent_helloworld"}

# Handle anchor properly
if [ -z "${PERSONALIZE_ANCHOR:-}" ]; then
  echo "[WARN] PERSONALIZE_ANCHOR not set, defaulting to tpm"
  export PERSONALIZE_ANCHOR="tpm"
fi

if [ -z "$ORG_ID" ] || [ -z "$BOOTSTRAP_TOKEN" ]; then
  echo "[ERROR] Missing credentials. Set AOC_ORG_ID + AOC_PROVISIONING_TOKEN."
  exit 1
fi

if [ "$MODE" = "smoke" ] || [ "${SAMPLE_CI_SMOKE:-0}" = "1" ]; then
  echo "[CI][smoke] HelloWorld env validation OK."
  echo "[CI][smoke] BASE_URL=$BASE_URL AGENT_ID=$AGENT_ID ANCHOR=$PERSONALIZE_ANCHOR"
  python3 -m py_compile helloworld_agent.py
  echo "[CI][smoke] helloworld_agent.py syntax OK."
  exit 0
fi

if [ "$MODE" = "oneshot" ]; then
  export HELLOWORLD_CI_ONESHOT=1
fi

echo "[INFO] Starting HelloWorld Agent..."
echo "  BASE_URL: $BASE_URL"
echo "  AGENT_ID: $AGENT_ID"
echo "  ANCHOR:   $PERSONALIZE_ANCHOR"

python3 helloworld_agent.py
