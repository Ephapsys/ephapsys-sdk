#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run.sh --local
  ./run.sh --local smoke

Notes:
  --local launches the robot sample locally
  --local smoke runs the existing local smoke check only
EOF
}

MODE=""
ACTION="run"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local)
      MODE="local"
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
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "$MODE" != "local" ]]; then
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ "$ACTION" == "smoke" ]]; then
  exec ./run_local.sh smoke
fi

exec ./run_local.sh
