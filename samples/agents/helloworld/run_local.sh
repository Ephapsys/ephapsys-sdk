#!/usr/bin/env bash
# ============================================================
# Run HelloWorld Agent Sample (Ephapsys SDK)
#
# This script loads environment variables from .env (if present)
# and runs the helloworld_agent.py demo.
# ============================================================

set -euo pipefail

MODE="${1:-run}" # run | check | smoke | oneshot

info() {
  echo "[INFO] $*"
}

warn() {
  echo "[WARN] $*"
}

error() {
  echo "[ERROR] $*" >&2
}

pick_anchor() {
  if [ -n "${PERSONALIZE_ANCHOR:-}" ]; then
    printf '%s\n' "$PERSONALIZE_ANCHOR"
    return
  fi

  case "$(uname -s)" in
    Linux)
      if command -v tpm2_getcap >/dev/null 2>&1; then
        printf 'tpm\n'
      else
        printf 'none\n'
      fi
      ;;
    *)
      printf 'none\n'
      ;;
  esac
}

if [ "$MODE" != "smoke" ] && [ ! -f ".env" ] && [ -f ".env.example" ]; then
  cp .env.example .env
  info "Created .env from .env.example. Fill in AOC_ORG_ID, AOC_PROVISIONING_TOKEN, and AGENT_TEMPLATE_ID, then rerun."
  exit 0
fi

# Load .env if available
if [ -f ".env" ]; then
  info "Loading environment from .env"
  # shellcheck disable=SC1091
  set -a && source .env && set +a
fi

# Config defaults
BASE_URL=${AOC_BASE_URL:-${AOC_API_BASE:-${AOC_API_URL:-${AOC_API:-"http://localhost:7001"}}}}
ORG_ID=${AOC_ORG_ID:-""}
BOOTSTRAP_TOKEN=${AOC_PROVISIONING_TOKEN:-""}
AGENT_ID=${AGENT_TEMPLATE_ID:-""}

export PERSONALIZE_ANCHOR="$(pick_anchor)"

if [ -z "${AGENT_ID}" ] && [ "$MODE" != "smoke" ] && [ "${SAMPLE_CI_SMOKE:-0}" != "1" ]; then
  error "Missing AGENT_TEMPLATE_ID. Set it in .env before running the sample."
  exit 1
fi

if [ "$MODE" = "smoke" ] || [ "${SAMPLE_CI_SMOKE:-0}" = "1" ]; then
  echo "[CI][smoke] HelloWorld env validation OK."
  echo "[CI][smoke] BASE_URL=$BASE_URL AGENT_ID=${AGENT_ID:-agent_helloworld} ANCHOR=$PERSONALIZE_ANCHOR"
  python3 -m py_compile helloworld_agent.py
  echo "[CI][smoke] helloworld_agent.py syntax OK."
  exit 0
fi

if [ "$MODE" = "oneshot" ]; then
  export HELLOWORLD_CI_ONESHOT=1
fi

if [ -z "$ORG_ID" ] || [ -z "$BOOTSTRAP_TOKEN" ]; then
  error "Missing credentials. Set AOC_ORG_ID and AOC_PROVISIONING_TOKEN in .env."
  exit 1
fi

VENV="${HELLOWORLD_VENV:-.venv}"
if [ ! -d "$VENV" ]; then
  info "Creating local virtualenv at $VENV"
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

SDK_PATH="../../../sdk/python"
if ! python3 -c "import ephapsys, transformers" >/dev/null 2>&1; then
  if [ "${HELLOWORLD_AUTO_INSTALL:-1}" = "1" ]; then
    if [ -d "$SDK_PATH" ]; then
      info "Installing local Ephapsys SDK + modulation extras into $VENV"
      PIP_DISABLE_PIP_VERSION_CHECK=1 python3 -m pip install --upgrade pip --quiet
      PIP_DISABLE_PIP_VERSION_CHECK=1 python3 -m pip install -e "${SDK_PATH}[modulation]" --quiet
    else
      error "Ephapsys SDK not installed and local SDK path $SDK_PATH was not found."
      exit 1
    fi
  else
    error "Missing runtime dependencies in $VENV. Run: python3 -m pip install -e \"${SDK_PATH}[modulation]\""
    exit 1
  fi
fi

if [ "$MODE" = "check" ]; then
  info "Running HelloWorld preflight checks..."
  export AOC_BASE_URL="$BASE_URL"
  python3 - <<'PY'
import os
import sys

from ephapsys.auth import check_helloworld_bootstrap

base_url = os.environ["AOC_BASE_URL"]
org_id = os.environ["AOC_ORG_ID"]
provisioning_token = os.environ["AOC_PROVISIONING_TOKEN"]
agent_template_id = os.environ["AGENT_TEMPLATE_ID"]
verify_ssl = os.getenv("AOC_VERIFY_SSL", "1").strip().lower() not in ("0", "false", "no", "")

result = check_helloworld_bootstrap(
    base_url=base_url,
    org_id=org_id,
    provisioning_token=provisioning_token,
    agent_template_id=agent_template_id,
    verify_ssl=verify_ssl,
)

print("[CHECK] HelloWorld backend preflight")
for item in result.get("checks", []):
    prefix = "PASS" if item.get("ok") else "FAIL"
    print(f"[CHECK] {prefix} {item.get('code')}: {item.get('message')}")

if not result.get("ready"):
    sys.exit(1)

print("[CHECK] Ready to run HelloWorld.")
PY
  exit 0
fi

info "Starting HelloWorld Agent..."
echo "  BASE_URL: $BASE_URL"
echo "  AGENT_ID: $AGENT_ID"
echo "  ANCHOR:   $PERSONALIZE_ANCHOR"

python3 helloworld_agent.py
