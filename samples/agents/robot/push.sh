#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./push.sh --local [--idempotent|--no-idempotent] [--label "Robot Agent Template"]

Notes:
  --local bootstraps the robot starter templates locally
  idempotent publish is the default; use --no-idempotent for full modulation
  all additional flags are forwarded to ./push_local.sh
EOF
}

MODE=""
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local)
      MODE="local"
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

if [[ "$MODE" != "local" ]]; then
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ${#ARGS[@]} -eq 0 ]]; then
  exec ./push_local.sh
fi

exec ./push_local.sh "${ARGS[@]}"
