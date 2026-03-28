#!/usr/bin/env bash
# ============================================================
# Run HelloWorld Agent Sample (Ephapsys SDK)
#
# This script loads environment variables from .env (if present)
# and runs the helloworld_agent.py demo.
# ============================================================

set -euo pipefail

MODE="${1:-run}" # run | check
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
  local venv sdk_extras requirements_ok sdk_source
  venv="${HELLOWORLD_VENV:-.venv}"
  sdk_extras="${HELLOWORLD_SDK_EXTRAS:-modulation}"
  sdk_source="${HELLOWORLD_SDK_PACKAGE_SOURCE:-${SDK_PACKAGE_SOURCE:-pypi}}"

  if [ "${HELLOWORLD_USE_LOCAL_SDK:-0}" = "1" ]; then
    if [ ! -d "$venv" ]; then
      info "Creating local virtualenv at $venv"
      python3 -m venv "$venv"
    fi
    # shellcheck disable=SC1091
    source "$venv/bin/activate"
    if [ ! -d "../../../sdk/python" ]; then
      error "Local SDK path not found at ../../../sdk/python"
      exit 1
    fi
    info "Installing local Ephapsys SDK into $venv"
    PIP_DISABLE_PIP_VERSION_CHECK=1 python3 -m pip install --upgrade pip --quiet
    PIP_DISABLE_PIP_VERSION_CHECK=1 python3 -m pip install -e "../../../sdk/python[${sdk_extras}]" --quiet
    return
  fi

  if [ ! -x "$SDK_SETUP_SH" ]; then
    error "SDK setup helper not found at $SDK_SETUP_SH"
    exit 1
  fi

  case "$sdk_source" in
    pypi|testpypi)
      ;;
    *)
      error "Unsupported HELLOWORLD_SDK_PACKAGE_SOURCE: $sdk_source"
      error "Use pypi, testpypi, or set HELLOWORLD_USE_LOCAL_SDK=1 for repo-local development."
      exit 1
      ;;
  esac

  info "Ensuring latest published Ephapsys SDK from $sdk_source in $venv"
  "$SDK_SETUP_SH" "--$sdk_source" --venv "$venv" --extras "$sdk_extras"
  # shellcheck disable=SC1091
  source "$venv/bin/activate"

  requirements_ok=1
  if ! python3 -c "import ephapsys, transformers" >/dev/null 2>&1; then
    requirements_ok=0
  fi
  if [ "$requirements_ok" -ne 1 ]; then
    error "Published SDK environment is missing required runtime dependencies."
    exit 1
  fi
}

run_preflight() {
  info "Running HelloWorld preflight checks..."
  export AOC_BASE_URL="$BASE_URL"
if ! python3 - <<'PY'
import os
import sys

try:
    from ephapsys.auth import check_helloworld_bootstrap
except ImportError:
    check_helloworld_bootstrap = None

base_url = os.environ["AOC_BASE_URL"]
org_id = os.environ["AOC_ORG_ID"]
provisioning_token = os.environ["AOC_PROVISIONING_TOKEN"]
agent_template_id = os.environ["AGENT_TEMPLATE_ID"]
verify_ssl = os.getenv("AOC_VERIFY_SSL", "1").strip().lower() not in ("0", "false", "no", "")

if check_helloworld_bootstrap is None:
    print("[CHECK] WARN preflight helper unavailable in installed SDK; skipping backend bootstrap checks.")
    print("[CHECK] Ready to run HelloWorld.")
    sys.exit(0)

try:
    result = check_helloworld_bootstrap(
        base_url=base_url,
        org_id=org_id,
        provisioning_token=provisioning_token,
        agent_template_id=agent_template_id,
        verify_ssl=verify_ssl,
    )
except Exception as exc:
    print(f"[CHECK] FAIL preflight: {exc}")
    sys.exit(1)

print("[CHECK] HelloWorld backend preflight")
for item in result.get("checks", []):
    prefix = "PASS" if item.get("ok") else "FAIL"
    print(f"[CHECK] {prefix} {item.get('code')}: {item.get('message')}")

if not result.get("ready"):
    failed = {item.get("code"): item.get("message", "") for item in result.get("checks", []) if not item.get("ok")}
    if "provisioning_token_valid" in failed:
        print("[CHECK] ACTION: refresh AOC_PROVISIONING_TOKEN in .env from the AOC UI for this environment, then rerun ./run.sh --local.")
    if "language_model_missing_artifacts" in failed:
        print("[CHECK] ACTION: rerun ./push.sh --local so the linked language model is published with the required artifacts.")
    sys.exit(1)

print("[CHECK] Ready to run HelloWorld.")
PY
  then
    error "HelloWorld preflight failed."
    error "See the failed check and action lines above, then rerun ./run.sh --local."
    exit 1
  fi
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

if [ ! -f ".env" ] && [ -f ".env.example" ]; then
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

if [ -z "${AGENT_ID}" ]; then
  error "Missing AGENT_TEMPLATE_ID. Set it in .env before running the sample."
  exit 1
fi

if [ -z "$ORG_ID" ] || [ -z "$BOOTSTRAP_TOKEN" ]; then
  error "Missing credentials. Set AOC_ORG_ID and AOC_PROVISIONING_TOKEN in .env."
  exit 1
fi

ensure_runtime_env

if [ "$MODE" = "check" ]; then
  run_preflight
  exit 0
fi

run_preflight

info "Starting HelloWorld Agent..."
echo "  BASE_URL: $BASE_URL"
echo "  AGENT_ID: $AGENT_ID"
echo "  ANCHOR:   $PERSONALIZE_ANCHOR"

python3 helloworld_agent.py
