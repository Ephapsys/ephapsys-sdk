#!/usr/bin/env bash

set -euo pipefail

MODULATOR_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULATOR_REPO_DIR="$(cd "$MODULATOR_COMMON_DIR/../../" && pwd)"
MODULATOR_SDK_SETUP_SH="$MODULATOR_REPO_DIR/scripts/setup.sh"

modulator_info() {
  echo "[INFO] $*"
}

modulator_error() {
  echo "[ERROR] $*" >&2
}

modulator_load_env_file() {
  local env_file line key value current_value
  env_file="${1:-.env}"

  if [ ! -f "$env_file" ]; then
    return
  fi

  modulator_info "Loading defaults from $env_file"
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    case "$line" in
      ''|'#'*)
        continue
        ;;
    esac
    if [[ "$line" != *=* ]]; then
      continue
    fi
    key="${line%%=*}"
    value="${line#*=}"
    current_value="${!key-}"
    if [ -n "$current_value" ]; then
      continue
    fi
    printf -v "$key" '%s' "$value"
    export "$key"
  done < "$env_file"
}

modulator_prepare_env() {
  local venv sdk_extras sdk_source sdk_version
  venv="${MODULATOR_VENV:-$MODULATOR_REPO_DIR/scripts/venvs/modulators}"
  sdk_extras="${MODULATOR_SDK_EXTRAS:-modulation,audio,vision,embedding,eval}"
  sdk_source="${MODULATOR_SDK_PACKAGE_SOURCE:-${SDK_PACKAGE_SOURCE:-local}}"
  sdk_version="${MODULATOR_SDK_VERSION:-${SDK_VERSION:-}}"

  if [ "${MODULATOR_SKIP_SDK_SETUP:-0}" = "1" ]; then
    modulator_info "Using pre-provisioned Python environment (MODULATOR_SKIP_SDK_SETUP=1)"
    if ! python3 -c "import ephapsys" >/dev/null 2>&1; then
      modulator_error "Pre-provisioned environment does not provide the ephapsys package."
      exit 1
    fi
    return
  fi

  if [ ! -x "$MODULATOR_SDK_SETUP_SH" ]; then
    modulator_error "SDK setup helper not found at $MODULATOR_SDK_SETUP_SH"
    exit 1
  fi

  case "$sdk_source" in
    local|pypi|testpypi)
      ;;
    *)
      modulator_error "Unsupported MODULATOR_SDK_PACKAGE_SOURCE: $sdk_source"
      modulator_error "Use local, pypi, or testpypi."
      exit 1
      ;;
  esac

  modulator_info "Ensuring Ephapsys SDK from $sdk_source in $venv"
  if [ -n "$sdk_version" ] && [ "$sdk_source" != "local" ]; then
    "$MODULATOR_SDK_SETUP_SH" "--$sdk_source" --venv "$venv" --extras "$sdk_extras" --version "$sdk_version"
  else
    "$MODULATOR_SDK_SETUP_SH" "--$sdk_source" --venv "$venv" --extras "$sdk_extras"
  fi

  # shellcheck disable=SC1090
  source "$venv/bin/activate"

  if ! python3 -c "import ephapsys" >/dev/null 2>&1; then
    modulator_error "Prepared modulator environment does not provide the ephapsys package."
    exit 1
  fi

  if [ -f "$MODULATOR_COMMON_DIR/requirements.gcp.txt" ]; then
    modulator_info "Syncing shared modulator requirements into $venv"
    PIP_DISABLE_PIP_VERSION_CHECK=1 python3 -m pip install -r "$MODULATOR_COMMON_DIR/requirements.gcp.txt" --quiet
  fi

  if [ -f "requirements.txt" ]; then
    modulator_info "Syncing modulator-local requirements into $venv"
    PIP_DISABLE_PIP_VERSION_CHECK=1 python3 -m pip install -r requirements.txt --quiet
  fi
}
