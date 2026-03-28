#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./push.sh [--idempotent|--no-idempotent] [--label "Robot Agent Template"]
  ./push.sh --local [--idempotent|--no-idempotent] [--label "Robot Agent Template"]
  ./push.sh --gcp [--idempotent|--no-idempotent] [--label "Robot Agent Template"]

Notes:
  no flag defaults to bootstrapping the robot starter templates locally
  --local does the same explicitly
  --gcp bootstraps the same templates for GCP brain deployment
  idempotent publish is the default; use --no-idempotent for full modulation
  all additional flags are forwarded to ./push_local.sh or ./push_gcp.sh
EOF
}

MODE="local"
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local)
      MODE="local"
      shift
      ;;
    --gcp)
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TARGET="./push_local.sh"
if [[ "$MODE" == "gcp" ]]; then
  TARGET="./push_gcp.sh"
fi

if [[ ${#ARGS[@]} -eq 0 ]]; then
  exec "$TARGET"
fi

exec "$TARGET" "${ARGS[@]}"
