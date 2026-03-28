#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run.sh --local
  ./run.sh --local smoke
  ./run.sh --gcp [--staging|--production] [other run_gcp.sh flags]

Notes:
  --local launches the robot sample locally
  --local smoke runs the existing local smoke check only
  --gcp provisions the brain remotely and keeps body + terminal local
EOF
}

MODE=""
ACTION="run"
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local)
      if [[ -n "$MODE" && "$MODE" != "local" ]]; then
        echo "[ERROR] Choose only one of --local or --gcp." >&2
        exit 1
      fi
      MODE="local"
      shift
      ;;
    --gcp)
      if [[ -n "$MODE" && "$MODE" != "gcp" ]]; then
        echo "[ERROR] Choose only one of --local or --gcp." >&2
        exit 1
      fi
      MODE="gcp"
      shift
      ;;
    smoke)
      ACTION="smoke"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$MODE" ]]; then
  usage
  exit 1
fi

if [[ "$MODE" == "local" ]]; then
  case "${ARGS[*]:-}" in
    "")
      ;;
    *)
      echo "[ERROR] Unknown argument(s) for local mode: ${ARGS[*]}" >&2
      usage
      exit 1
      ;;
  esac
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ "$MODE" == "local" && "$ACTION" == "smoke" ]]; then
  exec ./run_local.sh smoke
fi

if [[ "$MODE" == "local" ]]; then
  exec ./run_local.sh
fi

exec ./run_gcp.sh "${ARGS[@]}"
