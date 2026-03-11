#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f ".env" && -f ".env.example" ]]; then
  cp .env.example .env
  cat <<'EOF'
[INFO] Created .env from .env.example

Before continuing, edit .env and set:
  - AOC_BASE_URL
  - AOC_ORG_ID
  - AOC_PROVISIONING_TOKEN
  - AOC_MODULATION_TOKEN
  - HF_TOKEN

If you do not have an Ephapsys account yet, visit https://ephapsys.com, sign up, and generate the required organization and token values first.

Then rerun:
  ./quickstart.sh
EOF
  exit 0
fi

./push.sh --mode local "$@"
./run_local.sh
