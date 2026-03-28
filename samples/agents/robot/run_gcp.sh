#!/usr/bin/env bash
set -euo pipefail

BLUE="\033[36m"
GREEN="\033[32m"
YELLOW="\033[33m"
MAGENTA="\033[35m"
RESET="\033[0m"

info() {
  printf "${BLUE}[INFO]${RESET} %s\n" "$*"
}

warn() {
  printf "${YELLOW}[WARN]${RESET} %s\n" "$*" >&2
}

error() {
  printf "${MAGENTA}[ERROR]${RESET} %s\n" "$*" >&2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PYPROJECT="$REPO_ROOT/Product/ephapsys-sdk/sdk/python/pyproject.toml"
META_FILE="$SCRIPT_DIR/.last_gcp_instance"
GCP_ENV_FILE="${GCP_ENV_FILE:-${ROBOT_GCP_ENV_FILE:-$SCRIPT_DIR/.env.gcp}}"

usage() {
  cat <<'EOF'
Usage:
  ./run_gcp.sh
  ./run_gcp.sh --gpu
  ./run_gcp.sh --local-port 48765

Notes:
  - Deploys only the Robot brain to GCP.
  - Keeps microphone, camera, speaker, and terminal face local.
  - Opens an SSH tunnel to the remote brain and runs ./robot_remote_agent.py locally.
  - Infrastructure and package-source settings are loaded from ./.env.gcp.
EOF
}

if [[ -f "$GCP_ENV_FILE" ]]; then
  info "Loading GCP settings from $GCP_ENV_FILE"
  set -a && source "$GCP_ENV_FILE" && set +a
fi

PROJECT_ID="${PROJECT_ID:-}"
ZONE="${ZONE:-}"
MACHINE_TYPE="${MACHINE_TYPE:-}"
DISK_SIZE="${DISK_SIZE:-}"
IMAGE_FAMILY="${IMAGE_FAMILY:-}"
IMAGE_PROJECT="${IMAGE_PROJECT:-}"
INSTANCE_PREFIX="${INSTANCE_PREFIX:-}"
REMOTE_DIR="${REMOTE_DIR:-robot}"
REMOTE_PORT="${REMOTE_PORT:-8765}"
LOCAL_TUNNEL_PORT="${LOCAL_TUNNEL_PORT:-48765}"
CPU_ONLY=true
GPU_TYPE="${GPU_TYPE:-}"
GPU_COUNT="${GPU_COUNT:-1}"
GPU_MACHINE_TYPE="${GPU_MACHINE_TYPE:-}"
SDK_PACKAGE_SOURCE="${SDK_PACKAGE_SOURCE:-${ROBOT_SDK_PACKAGE_SOURCE:-pypi}}"
SDK_INDEX_URL="${SDK_INDEX_URL:-${ROBOT_SDK_INDEX_URL:-}}"
SDK_EXTRA_INDEX_URL="${SDK_EXTRA_INDEX_URL:-${ROBOT_SDK_EXTRA_INDEX_URL:-}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zone) ZONE="$2"; shift 2 ;;
    --project) PROJECT_ID="$2"; shift 2 ;;
    --machine-type) MACHINE_TYPE="$2"; shift 2 ;;
    --disk-size) DISK_SIZE="$2"; shift 2 ;;
    --instance-prefix) INSTANCE_PREFIX="$2"; shift 2 ;;
    --local-port) LOCAL_TUNNEL_PORT="$2"; shift 2 ;;
    --remote-port) REMOTE_PORT="$2"; shift 2 ;;
    --gpu)
      CPU_ONLY=false
      shift
      ;;
    --gpu-type)
      CPU_ONLY=false
      GPU_TYPE="$2"
      shift 2
      ;;
    --gpu-count)
      CPU_ONLY=false
      GPU_COUNT="$2"
      shift 2
      ;;
    --gpu-machine-type)
      CPU_ONLY=false
      GPU_MACHINE_TYPE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      error "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if ! command -v gcloud >/dev/null 2>&1; then
  error "gcloud CLI not found. Install and authenticate first."
  exit 1
fi

"$SCRIPT_DIR/check_gcp.sh" >/dev/null

for var in PROJECT_ID ZONE MACHINE_TYPE DISK_SIZE IMAGE_FAMILY IMAGE_PROJECT INSTANCE_PREFIX; do
  if [[ -z "${!var:-}" ]]; then
    error "$var must be set in $GCP_ENV_FILE"
    exit 1
  fi
done

SDK_VERSION="$(PYPROJECT_PATH="$PYPROJECT" python3 - <<'PY'
import os, pathlib
from typing import Any
try:
    import tomllib as toml
except ModuleNotFoundError:
    import tomli as toml  # type: ignore
pyproject = pathlib.Path(os.environ["PYPROJECT_PATH"])
data: dict[str, Any] = toml.loads(pyproject.read_text())
print(data.get("project", {}).get("version", "0.0.0"))
PY
)"

if [[ "$SDK_VERSION" == "0.0.0" || -z "$SDK_VERSION" ]]; then
  error "Unable to read SDK version from $PYPROJECT"
  exit 1
fi

ENV_FILE_LOCAL="${GCP_RUNTIME_ENV_FILE:-$SCRIPT_DIR/.env}"
ACTIVE_ENV_FILE="$ENV_FILE_LOCAL"

if [[ ! -f "$ACTIVE_ENV_FILE" ]]; then
  error "Missing runtime env file: $ENV_FILE_LOCAL"
  exit 1
fi

set -a
source "$ACTIVE_ENV_FILE"
set +a

if [[ -z "${AOC_BASE_URL:-${AOC_API_URL:-}}" ]]; then
  error "AOC_BASE_URL or AOC_API_URL must be set in $ACTIVE_ENV_FILE"
  exit 1
fi
if [[ -z "${AOC_ORG_ID:-}" || -z "${AOC_PROVISIONING_TOKEN:-}" || -z "${AGENT_TEMPLATE_ID:-}" ]]; then
  error "AOC_ORG_ID, AOC_PROVISIONING_TOKEN, and AGENT_TEMPLATE_ID must be set in $ACTIVE_ENV_FILE"
  exit 1
fi

INSTANCE_NAME="${INSTANCE_PREFIX}-$(date +%s)"
REMOTE_HOST="127.0.0.1"
TEMP_ENV_FILE="$(mktemp)"
TEMP_SRC="$(mktemp -d)"
TUNNEL_PID=""

cleanup() {
  if [[ -n "$TUNNEL_PID" ]]; then
    kill "$TUNNEL_PID" >/dev/null 2>&1 || true
  fi
  rm -f "$TEMP_ENV_FILE" >/dev/null 2>&1 || true
  rm -rf "$TEMP_SRC" >/dev/null 2>&1 || true
}
trap cleanup EXIT

cp "$ACTIVE_ENV_FILE" "$TEMP_ENV_FILE"
cat >>"$TEMP_ENV_FILE" <<EOF
ROBOT_BODY_MODE=remote
ROBOT_BRAIN_HOST=127.0.0.1
ROBOT_BRAIN_PORT=${REMOTE_PORT}
ROBOT_ENABLE_LIVE_VISION=${ROBOT_ENABLE_LIVE_VISION:-1}
DISABLE_AUDIO=1
EOF

mkdir -p "$TEMP_SRC"
cp "$SCRIPT_DIR"/robot_*.py "$TEMP_SRC"/
cp "$SCRIPT_DIR"/run_brain_server.sh "$TEMP_SRC"/
cp "$SCRIPT_DIR"/requirements_brain.txt "$TEMP_SRC"/

if $CPU_ONLY; then
  info "Creating CPU VM $INSTANCE_NAME in $PROJECT_ID/$ZONE"
  gcloud compute instances create "$INSTANCE_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --boot-disk-size="$DISK_SIZE" \
    --image-family="$IMAGE_FAMILY" \
    --image-project="$IMAGE_PROJECT" \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --quiet
else
  info "Creating GPU VM $INSTANCE_NAME in $PROJECT_ID/$ZONE"
  gcloud compute instances create "$INSTANCE_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --machine-type="$GPU_MACHINE_TYPE" \
    --boot-disk-size="$DISK_SIZE" \
    --image-family="$IMAGE_FAMILY" \
    --image-project="$IMAGE_PROJECT" \
    --maintenance-policy=TERMINATE \
    --restart-on-failure \
    --accelerator="type=${GPU_TYPE},count=${GPU_COUNT}" \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --quiet
fi

info "Preparing remote VM runtime"
gcloud compute ssh "$INSTANCE_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --command="mkdir -p ~/${REMOTE_DIR} ~/.venvs/robot-brain && sudo apt-get update -y >/dev/null && sudo apt-get install -y python3-venv ffmpeg libsndfile1 >/dev/null"

info "Uploading robot brain sample files"
gcloud compute scp --recurse "$TEMP_SRC/." "${INSTANCE_NAME}:~/${REMOTE_DIR}/" --project="$PROJECT_ID" --zone="$ZONE"
gcloud compute scp "$TEMP_ENV_FILE" "${INSTANCE_NAME}:~/${REMOTE_DIR}/.env" --project="$PROJECT_ID" --zone="$ZONE"
gcloud compute ssh "$INSTANCE_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --command="chmod +x ~/${REMOTE_DIR}/run_brain_server.sh"

case "${SDK_PACKAGE_SOURCE,,}" in
  pypi)
    REMOTE_PIP_INSTALL=$'python3 -m venv ~/.venvs/robot-brain\nsource ~/.venvs/robot-brain/bin/activate\npython -m pip install --upgrade pip >/dev/null\npython -m pip install "ephapsys[audio,vision,embedding]=='"$SDK_VERSION"'" >/dev/null\npython -m pip install -r ~/'"$REMOTE_DIR"'/requirements_brain.txt >/dev/null'
    ;;
  testpypi)
    REMOTE_PIP_INSTALL=$'python3 -m venv ~/.venvs/robot-brain\nsource ~/.venvs/robot-brain/bin/activate\npython -m pip install --upgrade pip >/dev/null\npython -m pip install --extra-index-url https://pypi.org/simple --index-url https://test.pypi.org/simple "ephapsys[audio,vision,embedding]=='"$SDK_VERSION"'" >/dev/null\npython -m pip install -r ~/'"$REMOTE_DIR"'/requirements_brain.txt >/dev/null'
    ;;
  custom)
    if [[ -z "$SDK_INDEX_URL" ]]; then
      error "SDK_INDEX_URL must be set when SDK_PACKAGE_SOURCE=custom"
      exit 1
    fi
    CUSTOM_INDEX_FLAGS="--index-url $(printf '%q' "$SDK_INDEX_URL")"
    if [[ -n "$SDK_EXTRA_INDEX_URL" ]]; then
      CUSTOM_INDEX_FLAGS="$CUSTOM_INDEX_FLAGS --extra-index-url $(printf '%q' "$SDK_EXTRA_INDEX_URL")"
    fi
    REMOTE_PIP_INSTALL=$'python3 -m venv ~/.venvs/robot-brain\nsource ~/.venvs/robot-brain/bin/activate\npython -m pip install --upgrade pip >/dev/null\npython -m pip install '"$CUSTOM_INDEX_FLAGS"$' "ephapsys[audio,vision,embedding]=='"$SDK_VERSION"'" >/dev/null\npython -m pip install -r ~/'"$REMOTE_DIR"'/requirements_brain.txt >/dev/null'
    ;;
  *)
    error "Unsupported SDK_PACKAGE_SOURCE value"
    exit 1
    ;;
esac

info "Installing remote Python dependencies"
gcloud compute ssh "$INSTANCE_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --command="bash -lc $(printf '%q' "$REMOTE_PIP_INSTALL")"

REMOTE_START=$'cd ~/'"$REMOTE_DIR"$'\nsource ~/.venvs/robot-brain/bin/activate\nnohup ./run_brain_server.sh > ~/robot_brain.log 2>&1 < /dev/null &\n'
info "Starting remote robot brain"
gcloud compute ssh "$INSTANCE_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --command="bash -lc $(printf '%q' "$REMOTE_START")"

printf 'INSTANCE_NAME=%s\nPROJECT_ID=%s\nZONE=%s\nLOCAL_PORT=%s\nREMOTE_PORT=%s\n' \
  "$INSTANCE_NAME" "$PROJECT_ID" "$ZONE" "$LOCAL_TUNNEL_PORT" "$REMOTE_PORT" >"$META_FILE"

info "Opening SSH tunnel on localhost:${LOCAL_TUNNEL_PORT}"
gcloud compute ssh "$INSTANCE_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  -- -N -L "${LOCAL_TUNNEL_PORT}:${REMOTE_HOST}:${REMOTE_PORT}" &
TUNNEL_PID=$!
sleep 5

info "Launching local body + terminal face against remote brain"
export ROBOT_BRAIN_WS_URL="ws://127.0.0.1:${LOCAL_TUNNEL_PORT}/ws/state"
export ROBOT_BRAIN_AUDIO_WS_URL="ws://127.0.0.1:${LOCAL_TUNNEL_PORT}/ws/body/audio"
export ROBOT_BRAIN_VIDEO_WS_URL="ws://127.0.0.1:${LOCAL_TUNNEL_PORT}/ws/body/video"
export ROBOT_BRAIN_CONTROL_WS_URL="ws://127.0.0.1:${LOCAL_TUNNEL_PORT}/ws/body/control"
python3 "$SCRIPT_DIR/robot_remote_agent.py"
