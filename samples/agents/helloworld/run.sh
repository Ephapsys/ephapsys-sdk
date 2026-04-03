#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run.sh
  ./run.sh --local
  ./run.sh --gcp [other run_gcp.sh flags]

Examples:
  ./run.sh
  ./run.sh --gcp
  ./run.sh --gcp --gpu --gpu-type t4

Notes:
  no flag defaults to local and runs preflight automatically, then launches ./run_local.sh
  --local does the same explicitly
  --gcp dispatches to ./run_gcp.sh
EOF
}

MODE=""
ARGS=()
ARGS_COUNT=0

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
      if [[ "$MODE" == "local" && "$1" == "check" ]]; then
        echo "[ERROR] ./run.sh --local already performs preflight automatically. Use ./run_local.sh check only if you explicitly want preflight without launch." >&2
        exit 1
      fi
      ARGS+=("$1")
      ARGS_COUNT=$((ARGS_COUNT + 1))
      shift
      ;;
  esac
done

if [[ -z "$MODE" ]]; then
  MODE="local"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

case "$MODE" in
  local)
    if [[ "$ARGS_COUNT" -gt 0 ]]; then
      echo "[ERROR] ./run.sh --local does not take extra arguments." >&2
      exit 1
    fi
    exec ./run_local.sh
    ;;
  gcp)
    if [[ "$ARGS_COUNT" -gt 0 ]]; then
      exec ./run_gcp.sh "${ARGS[@]}"
    fi
    exec ./run_gcp.sh
    ;;
esac
