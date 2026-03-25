#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run.sh --local [check]
  ./run.sh --gcp [--staging|--production] [other run_gcp.sh flags]

Examples:
  ./run.sh --local
  ./run.sh --local check
  ./run.sh --gcp --staging
  ./run.sh --gcp --production --gpu --gpu-type t4

Notes:
  --local dispatches to ./run_local.sh
  --gcp dispatches to ./run_gcp.sh
EOF
}

MODE=""
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

case "$MODE" in
  local)
    exec ./run_local.sh "${ARGS[@]}"
    ;;
  gcp)
    exec ./run_gcp.sh "${ARGS[@]}"
    ;;
esac
