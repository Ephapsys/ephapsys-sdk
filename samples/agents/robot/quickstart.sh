#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

info() {
  printf '[INFO] %s\n' "$*"
}

warn() {
  printf '[WARN] %s\n' "$*" >&2
}

has_robot_runtime_deps() {
  python3 -c "import ephapsys, transformers, faiss, cv2, sounddevice, webrtcvad" >/dev/null 2>&1
}

if [[ ! -f ".env" && -f ".env.example" ]]; then
  cp .env.example .env
  cat <<'EOF'
[INFO] Created .env from .env.example

Before continuing, edit .env and set:
  - AOC_BASE_URL
  - AOC_ORG_ID
  - AOC_PROVISIONING_TOKEN
  - AOC_MODULATION_TOKEN or API_TOKEN
  - HF_TOKEN (only if your chosen model repos require it)

Then rerun:
  ./quickstart.sh
EOF
  exit 0
fi

./push.sh --local "$@"

if ! has_robot_runtime_deps && [[ "${ROBOT_USE_LOCAL_SDK:-0}" != "1" ]]; then
  cat <<'EOF'
[WARN] Robot templates were created successfully, but the local runtime dependencies are not installed yet.

Install them in your active environment with:
  pip install "ephapsys[audio,vision,embedding]"
  pip install -r requirements.txt

Then rerun:
  ./run.sh --local

For repo-local SDK development only, you can instead run:
  ROBOT_USE_LOCAL_SDK=1 ./run.sh --local
EOF
  exit 0
fi

./run.sh --local
