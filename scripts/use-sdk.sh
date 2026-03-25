#!/usr/bin/env bash

# Source this script to install/upgrade the SDK environment and activate it in
# the current shell.
#
# Examples:
#   source ./use-sdk.sh --testpypi
#   source ./use-sdk.sh --local

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "[ERROR] use-sdk.sh must be sourced, not executed." >&2
  echo "Run: source ./use-sdk.sh --testpypi" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_VENV="$SCRIPT_DIR/venvs/sdk"
VENV_PATH="$DEFAULT_VENV"
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --venv requires a path" >&2
        return 1
      fi
      VENV_PATH="$2"
      ARGS+=("$1" "$2")
      shift 2
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

"$SCRIPT_DIR/setup.sh" "${ARGS[@]}"

# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"

echo "[DONE] Activated SDK environment: $VENV_PATH"
echo "[INFO] python: $(command -v python)"
echo "[INFO] ephapsys: $(command -v ephapsys)"
echo "[INFO] version: $(ephapsys --version)"
