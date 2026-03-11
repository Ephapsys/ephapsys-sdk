#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f ".env" && -f ".env.example" ]]; then
  cp .env.example .env
  echo "[INFO] Created .env from .env.example"
fi

./push.sh --mode local "$@"
./run_local.sh check
./run_local.sh
