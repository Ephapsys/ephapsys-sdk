#!/usr/bin/env bash
# ============================================================
# Bash wrapper to run the Ephaptic Language Trainer
# ============================================================

set -euo pipefail

MODE="${1:-run}" # run | smoke
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../../" && pwd)"
SDK_SETUP_SH="$REPO_DIR/scripts/setup.sh"

info() {
  echo "[INFO] $*"
}

error() {
  echo "[ERROR] $*" >&2
}

ensure_runtime_env() {
  local venv sdk_extras sdk_source sdk_version
  venv="${MODULATOR_VENV:-.venv}"
  sdk_extras="${MODULATOR_SDK_EXTRAS:-modulation}"
  sdk_source="${MODULATOR_SDK_PACKAGE_SOURCE:-${SDK_PACKAGE_SOURCE:-local}}"
  sdk_version="${MODULATOR_SDK_VERSION:-${SDK_VERSION:-}}"

  if [ ! -x "$SDK_SETUP_SH" ]; then
    error "SDK setup helper not found at $SDK_SETUP_SH"
    exit 1
  fi

  case "$sdk_source" in
    local|pypi|testpypi)
      ;;
    *)
      error "Unsupported MODULATOR_SDK_PACKAGE_SOURCE: $sdk_source"
      error "Use local, pypi, or testpypi."
      exit 1
      ;;
  esac

  info "Ensuring Ephapsys SDK from $sdk_source in $venv"
  if [ -n "$sdk_version" ] && [ "$sdk_source" != "local" ]; then
    "$SDK_SETUP_SH" "--$sdk_source" --venv "$venv" --extras "$sdk_extras" --version "$sdk_version"
  else
    "$SDK_SETUP_SH" "--$sdk_source" --venv "$venv" --extras "$sdk_extras"
  fi
  # shellcheck disable=SC1090
  source "$venv/bin/activate"

  if ! python3 -c "import ephapsys" >/dev/null 2>&1; then
    error "Prepared modulator environment does not provide the ephapsys package."
    exit 1
  fi

  if [ -f "requirements.txt" ]; then
    info "Syncing modulator-local requirements into $venv"
    PIP_DISABLE_PIP_VERSION_CHECK=1 python3 -m pip install -r requirements.txt --quiet
  fi
}

ensure_python_deps() {
  if python3 -c "import pkg_resources" >/dev/null 2>&1; then
    return
  fi
  info "Installing local Python bootstrap dependency: setuptools<74"
  python3 -m pip install 'setuptools<74'
  python3 -c "import pkg_resources" >/dev/null 2>&1 || {
    error "pkg_resources is still unavailable after installing setuptools<74"
    exit 1
  }
}

# --- Load environment from .env if present ---
if [ -f ".env" ]; then
  info "Loading environment from .env"
  # shellcheck disable=SC2046
  export $(grep -v '^#' .env | xargs)
fi

# --- Default values ---
BASE_URL=${AOC_BASE_URL:-${BASE_URL:-"http://localhost:7001"}}
AOC_ORG_ID=${AOC_ORG_ID:-""}
AOC_MODULATION_TOKEN=${AOC_MODULATION_TOKEN:-""}
MODEL_TEMPLATE_ID=${MODEL_TEMPLATE_ID:-""}
OUTDIR=${OUTDIR:-"./artifacts"}
TRAIN_MODE=${TRAIN_MODE:-"1"}   # ✅ Default = 1 (training enabled)

# --- Sanity checks ---
if [ -z "$AOC_ORG_ID" ] || [ -z "$AOC_MODULATION_TOKEN" ] || [ -z "$MODEL_TEMPLATE_ID" ]; then
  error "AOC_ORG_ID, AOC_MODULATION_TOKEN and MODEL_TEMPLATE_ID must be set."
  exit 1
fi

if [ "$MODE" = "smoke" ] || [ "${SAMPLE_CI_SMOKE:-0}" = "1" ]; then
  echo "[CI][smoke] Language modulator env validation OK."
  echo "[CI][smoke] BASE_URL=$BASE_URL TEMPLATE=$MODEL_TEMPLATE_ID"
  ensure_runtime_env
  python3 -m py_compile train_language.py
  echo "[CI][smoke] train_language.py syntax OK."
  exit 0
fi

ensure_runtime_env
ensure_python_deps

info "Starting Ephaptic Language Trainer..."
echo "  BASE_URL:          $BASE_URL"
echo "  AOC_ORG_ID:        $AOC_ORG_ID"
echo "  AOC_MODULATION_TOKEN:           ${AOC_MODULATION_TOKEN:0:8}********"
echo "  MODEL_TEMPLATE_ID: $MODEL_TEMPLATE_ID"
echo "  OUTDIR:            $OUTDIR"
echo "  TRAIN_MODE:        $TRAIN_MODE (1=train enabled, 0=evaluation only)"

# --- Build Python command dynamically ---
CMD=(
  python3 train_language.py
  --base_url "$BASE_URL"
  --api_key "$AOC_MODULATION_TOKEN"
  --model_template_id "$MODEL_TEMPLATE_ID"
  --outdir "$OUTDIR"
)

if [ "$TRAIN_MODE" = "1" ]; then
  info "Training flag active (gradient updates ON)"
  CMD+=(--train)   # ✅ new unified flag (backward-compatible internally)
else
  info "Evaluation-only mode (baseline + ephaptic comparison)"
fi

# --- Execute trainer ---
"${CMD[@]}"

info "Trainer finished successfully."
info "Artifacts stored in: $OUTDIR"
