#!/usr/bin/env bash
set -euo pipefail

if [ -f ".env" ]; then
  set -a && source .env && set +a
fi

BASE_URL=${AOC_BASE_URL:-${AOC_API_URL:-${AOC_API:-"http://localhost:7001"}}}
ORG_ID=${AOC_ORG_ID:-""}
PROVISIONING_TOKEN=${AOC_PROVISIONING_TOKEN:-""}
AGENT_ID=${AGENT_TEMPLATE_ID:-""}

if [ -z "$ORG_ID" ] || [ -z "$PROVISIONING_TOKEN" ] || [ -z "$AGENT_ID" ]; then
  echo "[ERROR] Missing credentials. Set AOC_ORG_ID + AOC_PROVISIONING_TOKEN + AGENT_TEMPLATE_ID."
  exit 1
fi

export PERSONALIZE_ANCHOR="${PERSONALIZE_ANCHOR:-none}"
echo "[Robot CI] BASE_URL=$BASE_URL AGENT_ID=$AGENT_ID ANCHOR=$PERSONALIZE_ANCHOR"

python3 robot_ci_mock.py

