#!/usr/bin/env bash
set -euo pipefail

BLUE="\033[36m"
GREEN="\033[32m"
YELLOW="\033[33m"
RESET="\033[0m"

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULATOR_DIR="${MODULATOR_DIR:-$(pwd)}"
MODULATOR_KIND="${MODULATOR_KIND:-$(basename "$MODULATOR_DIR")}"
TRAINER_SCRIPT="${TRAINER_SCRIPT:-train_${MODULATOR_KIND}.py}"
DEFAULT_OUTDIR="${DEFAULT_OUTDIR:-./artifacts_${MODULATOR_KIND}}"

if [ ! -d "$MODULATOR_DIR" ]; then
  echo "❌ MODULATOR_DIR does not exist: $MODULATOR_DIR"
  exit 1
fi

if [ ! -f "$MODULATOR_DIR/$TRAINER_SCRIPT" ]; then
  echo "❌ Trainer script not found: $MODULATOR_DIR/$TRAINER_SCRIPT"
  exit 1
fi

load_gcp_env_defaults() {
  local env_file="$1"
  local key value current
  [ -f "$env_file" ] || return 0
  echo "📂 Loading GCP defaults from $env_file"
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    case "$line" in
      ''|'#'*)
        continue
        ;;
    esac
    line="${line#export }"
    if [[ "$line" != *=* ]]; then
      continue
    fi
    key="${line%%=*}"
    value="${line#*=}"
    current="${!key-}"
    if [ -n "$current" ]; then
      continue
    fi
    printf -v "$key" '%s' "$value"
    export "$key"
  done < "$env_file"
}

load_gcp_env_defaults "$MODULATOR_DIR/gcp.env"
load_gcp_env_defaults "$COMMON_DIR/gcp.env"

GPU_CHOICE="${1:-t4}"
shift || true

while [ $# -gt 0 ]; do
  case "$1" in
    --no-build)
      echo "ℹ️ Ignoring --no-build: VM-first GCP modulation no longer builds local Docker images."
      ;;
    --no-push)
      echo "ℹ️ Ignoring --no-push: VM-first GCP modulation no longer pushes local Docker images."
      ;;
    *)
      echo "❌ Unknown option: $1"
      exit 1
      ;;
  esac
  shift
done

if [ ! -f "$MODULATOR_DIR/.env" ]; then
  echo "❌ Missing $MODULATOR_DIR/.env"
  exit 1
fi

set -a
# shellcheck source=/dev/null
source "$MODULATOR_DIR/.env"
set +a

BASE_URL="${AOC_BASE_URL:-${BASE_URL:-}}"
AOC_ORG_ID="${AOC_ORG_ID:-}"
AOC_MODULATION_TOKEN="${AOC_MODULATION_TOKEN:-}"
MODEL_TEMPLATE_ID="${MODEL_TEMPLATE_ID:-}"
HF_TOKEN="${HF_TOKEN:-}"

if [ -n "${ZONE:-}" ]; then
  REGION="${ZONE%-*}"
  export REGION
fi

if [ -z "${PROJECT_ID:-}" ] || [ -z "${REGION:-}" ] || [ -z "${ZONE:-}" ]; then
  echo "❌ PROJECT_ID, REGION, and ZONE must be set via environment or gcp.env"
  exit 1
fi

if [ -z "$BASE_URL" ] || [ -z "$AOC_ORG_ID" ] || [ -z "$AOC_MODULATION_TOKEN" ] || [ -z "$MODEL_TEMPLATE_ID" ]; then
  echo "❌ BASE_URL/AOC_ORG_ID/AOC_MODULATION_TOKEN/MODEL_TEMPLATE_ID missing in .env"
  exit 1
fi

SDK_PACKAGE_SOURCE="${MODULATOR_SDK_PACKAGE_SOURCE:-${SDK_PACKAGE_SOURCE:-pypi}}"
REPO_ROOT="$(cd "$COMMON_DIR/../.." && pwd)"
PYPROJECT_PATH="$REPO_ROOT/sdk/python/pyproject.toml"
SDK_VERSION="$(PYPROJECT_PATH="$PYPROJECT_PATH" python3 - <<'PY'
import os, pathlib, re
text = pathlib.Path(os.environ["PYPROJECT_PATH"]).read_text()
match = re.search(r'(?m)^\s*version\s*=\s*"([^"]+)"', text)
print(match.group(1) if match else "0.0.0")
PY
)"
if [ "$SDK_VERSION" = "0.0.0" ] || [ -z "$SDK_VERSION" ]; then
  echo "❌ Unable to determine SDK version from $PYPROJECT_PATH"
  exit 1
fi

INSTANCE_NAME="ec-modulate-${MODULATOR_KIND}-$(date +%s)"
DISK_SIZE="${DISK_SIZE:-200GB}"
case "$DISK_SIZE" in
  *GB|*gb)
    disk_size_num="${DISK_SIZE%[Gg][Bb]}"
    if [[ "$disk_size_num" =~ ^[0-9]+$ && "$disk_size_num" -lt 100 ]]; then
      DISK_SIZE="100GB"
    fi
    ;;
esac
REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-~/ephapsys-modulators}"
REMOTE_DIR="${REMOTE_BASE_DIR}/${MODULATOR_KIND}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-${MODULATOR_KIND}-modulate}"
AUTO_DELETE="${AUTO_DELETE:-true}"
TRAIN_MODE="${TRAIN_MODE:-1}"
RUN_TS="$(date +"%Y%m%d_%H%M%S")"
RESULTS_DIR="$MODULATOR_DIR/results/${EXPERIMENT_TAG}_${RUN_TS}"
mkdir -p "$RESULTS_DIR"
TEMP_SRC="$(mktemp -d)"
SHARED_STATE_FILE="${MODULATOR_GCP_SHARED_STATE_FILE:-}"
SHARED_INSTANCE_REUSED=0

info() {
  printf "${BLUE}[INFO]${RESET} %s\n" "$*"
}

success() {
  printf "${GREEN}[SELECTED]${RESET} %s\n" "$*"
}

warn() {
  printf "${YELLOW}[WARN]${RESET} %s\n" "$*" >&2
}

MASKED="${AOC_MODULATION_TOKEN:0:8}********"
echo "🔐 Env loaded: BASE_URL=$BASE_URL, AOC_ORG_ID=$AOC_ORG_ID, AOC_MODULATION_TOKEN=${MASKED}, MODEL_TEMPLATE_ID=$MODEL_TEMPLATE_ID"

case "$GPU_CHOICE" in
  t4|T4) GPU_TYPE="nvidia-tesla-t4"; GPU_COUNT=1; MACHINE_TYPE="n1-standard-8" ;;
  l4|L4) GPU_TYPE="nvidia-l4"; GPU_COUNT=1; MACHINE_TYPE="g2-standard-8" ;;
  v100|V100) GPU_TYPE="nvidia-tesla-v100"; GPU_COUNT=1; MACHINE_TYPE="n1-standard-8" ;;
  p100|P100) GPU_TYPE="nvidia-tesla-p100"; GPU_COUNT=1; MACHINE_TYPE="n1-standard-8" ;;
  a100|A100) MACHINE_TYPE="a2-highgpu-1g" ;;
  a100-2g|A100-2G) MACHINE_TYPE="a2-highgpu-2g" ;;
  a100-4g|A100-4G) MACHINE_TYPE="a2-highgpu-4g" ;;
  a100-8g|A100-8G) MACHINE_TYPE="a2-highgpu-8g" ;;
  *)
    echo "❌ Unknown GPU choice: $GPU_CHOICE"
    exit 1
    ;;
esac

build_gpu_candidates() {
  local candidates=()
  candidates+=("${GPU_TYPE}:${MACHINE_TYPE}:${GPU_COUNT}:${GPU_CHOICE}")
  if [[ -n "${GPU_FALLBACKS:-}" ]]; then
    IFS=',' read -r -a fallback_items <<< "$GPU_FALLBACKS"
    local item item_lc gpu_type machine_type gpu_count gpu_label
    for item in "${fallback_items[@]}"; do
      item="${item// /}"
      [[ -z "$item" ]] && continue
      item_lc="$(printf '%s' "$item" | tr '[:upper:]' '[:lower:]')"
      case "$item_lc" in
        t4) gpu_type="nvidia-tesla-t4"; machine_type="n1-standard-8"; gpu_count="1"; gpu_label="t4" ;;
        l4) gpu_type="nvidia-l4"; machine_type="g2-standard-8"; gpu_count="1"; gpu_label="l4" ;;
        v100) gpu_type="nvidia-tesla-v100"; machine_type="n1-standard-8"; gpu_count="1"; gpu_label="v100" ;;
        p100) gpu_type="nvidia-tesla-p100"; machine_type="n1-standard-8"; gpu_count="1"; gpu_label="p100" ;;
        *)
          continue
          ;;
      esac
      candidates+=("${gpu_type}:${machine_type}:${gpu_count}:${gpu_label}")
    done
  fi
  printf '%s\n' "${candidates[@]}" | awk '!seen[$0]++'
}

build_zone_candidates() {
  local zones=("$ZONE")
  if [[ -n "${ZONE_FALLBACKS:-}" ]]; then
    IFS=',' read -r -a fallback_zones <<< "$ZONE_FALLBACKS"
    local item
    for item in "${fallback_zones[@]}"; do
      item="${item// /}"
      [[ -n "$item" ]] && zones+=("$item")
    done
  else
    local suffix
    for suffix in a b c d e f; do
      item="${REGION}-${suffix}"
      [[ "$item" != "$ZONE" ]] && zones+=("$item")
    done
  fi
  printf '%s\n' "${zones[@]}" | awk '!seen[$0]++'
}

create_gpu_vm() {
  if [[ -n "$SHARED_STATE_FILE" && -f "$SHARED_STATE_FILE" ]]; then
    local shared_instance_name="" shared_zone="" shared_project_id="" shared_gpu_type="" shared_gpu_count="" shared_machine_type="" shared_gpu_choice=""
    # shellcheck source=/dev/null
    source "$SHARED_STATE_FILE"
    shared_instance_name="${INSTANCE_NAME:-}"
    shared_zone="${ZONE:-}"
    shared_project_id="${PROJECT_ID:-}"
    shared_gpu_type="${GPU_TYPE:-}"
    shared_gpu_count="${GPU_COUNT:-}"
    shared_machine_type="${MACHINE_TYPE:-}"
    shared_gpu_choice="${GPU_CHOICE:-}"
    if [[ -n "$shared_instance_name" && -n "$shared_zone" && -n "$shared_project_id" ]] && \
      gcloud compute instances describe "$shared_instance_name" --project="$shared_project_id" --zone="$shared_zone" >/dev/null 2>&1; then
      INSTANCE_NAME="$shared_instance_name"
      ZONE="$shared_zone"
      PROJECT_ID="$shared_project_id"
      [[ -n "$shared_gpu_type" ]] && GPU_TYPE="$shared_gpu_type"
      [[ -n "$shared_gpu_count" ]] && GPU_COUNT="$shared_gpu_count"
      [[ -n "$shared_machine_type" ]] && MACHINE_TYPE="$shared_machine_type"
      [[ -n "$shared_gpu_choice" ]] && GPU_CHOICE="$shared_gpu_choice"
      SHARED_INSTANCE_REUSED=1
      success "Reusing shared modulation VM: instance=${INSTANCE_NAME} gpu=${GPU_CHOICE} machine=${MACHINE_TYPE} zone=${ZONE} project=${PROJECT_ID}"
      return 0
    fi
  fi

  local gpu_candidates=()
  local zone_candidates=()
  local candidate zone_candidate gpu_type machine_type gpu_count gpu_label
  while IFS= read -r candidate; do
    [[ -n "$candidate" ]] && gpu_candidates+=("$candidate")
  done < <(build_gpu_candidates)
  while IFS= read -r zone_candidate; do
    [[ -n "$zone_candidate" ]] && zone_candidates+=("$zone_candidate")
  done < <(build_zone_candidates)

  if [ ${#gpu_candidates[@]} -eq 0 ] || [ ${#zone_candidates[@]} -eq 0 ]; then
    return 1
  fi

  local gpu_total=${#gpu_candidates[@]}
  local zone_total=${#zone_candidates[@]}
  local gpu_idx=0
  for candidate in "${gpu_candidates[@]}"; do
    gpu_idx=$((gpu_idx + 1))
    IFS=':' read -r gpu_type machine_type gpu_count gpu_label <<< "$candidate"
    local zone_idx=0
    for zone_candidate in "${zone_candidates[@]}"; do
      zone_idx=$((zone_idx + 1))
      echo "⏳ [STEP 1] Creating GCP VM now: $INSTANCE_NAME (gpu=${gpu_label}, project=$PROJECT_ID, zone=${zone_candidate}, candidate ${gpu_idx}/${gpu_total}, zone ${zone_idx}/${zone_total})..."
      if gcloud compute instances create "$INSTANCE_NAME" \
        --project="$PROJECT_ID" --zone="$zone_candidate" \
        --machine-type="$machine_type" \
        --accelerator="type=$gpu_type,count=$gpu_count" \
        --image-family="pytorch-2-7-cu128-ubuntu-2204-nvidia-570" \
        --image-project="deeplearning-platform-release" \
        --boot-disk-size="$DISK_SIZE" --maintenance-policy=TERMINATE \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        --metadata="install-nvidia-driver=True" \
        --network=default --subnet=default --address="" \
        --restart-on-failure --labels=job=modulate,gpu="$gpu_label",tag="$EXPERIMENT_TAG",kind="$MODULATOR_KIND"; then
        GPU_TYPE="$gpu_type"
        GPU_COUNT="$gpu_count"
        MACHINE_TYPE="$machine_type"
        GPU_CHOICE="$gpu_label"
        ZONE="$zone_candidate"
        export ZONE GPU_TYPE GPU_COUNT MACHINE_TYPE GPU_CHOICE
        if [[ -n "$SHARED_STATE_FILE" ]]; then
          cat > "$SHARED_STATE_FILE" <<EOF
PROJECT_ID=$PROJECT_ID
ZONE=$ZONE
INSTANCE_NAME=$INSTANCE_NAME
GPU_TYPE=$GPU_TYPE
GPU_COUNT=$GPU_COUNT
MACHINE_TYPE=$MACHINE_TYPE
GPU_CHOICE=$GPU_CHOICE
EOF
        fi
        success "Selected shared modulation VM: instance=${INSTANCE_NAME} gpu=${GPU_CHOICE} machine=${MACHINE_TYPE} zone=${ZONE} project=${PROJECT_ID}"
        return 0
      fi
    done
  done
  return 1
}

cleanup() {
  rm -rf "$TEMP_SRC" >/dev/null 2>&1 || true
  if [[ -n "$SHARED_STATE_FILE" ]]; then
    return 0
  fi
  if gcloud compute instances describe "$INSTANCE_NAME" --project="$PROJECT_ID" --zone="$ZONE" >/dev/null 2>&1; then
    echo "⚠️ Cleaning up VM: $INSTANCE_NAME"
    if [ "$AUTO_DELETE" = "true" ]; then
      gcloud compute instances delete "$INSTANCE_NAME" --project="$PROJECT_ID" --zone="$ZONE" --quiet || true
    else
      gcloud compute instances stop "$INSTANCE_NAME" --project="$PROJECT_ID" --zone="$ZONE" --quiet || true
    fi
  fi
}
trap cleanup EXIT

echo "☁️ GCP target: project=${PROJECT_ID} region=${REGION} zone=${ZONE}"
echo "🧠 Remote modulation plan: kind=${MODULATOR_KIND} gpu=${GPU_CHOICE} instance=${INSTANCE_NAME}"
echo "📦 VM-first path: samples are copied to the VM and the published SDK is installed there."
echo "📦 Using SDK package source=${SDK_PACKAGE_SOURCE} version=${SDK_VERSION}"
if ! create_gpu_vm; then
  echo "❌ Unable to create a GPU VM for ${MODULATOR_KIND} in ${REGION}. Tried GPU_FALLBACKS=${GPU_FALLBACKS:-$GPU_CHOICE} and zones ${ZONE_FALLBACKS:-${REGION}-{a..f}}"
  exit 1
fi

if [ "$SHARED_INSTANCE_REUSED" = "1" ]; then
  info "Shared modulation VM ready: instance=${INSTANCE_NAME} gpu=${GPU_CHOICE} machine=${MACHINE_TYPE} zone=${ZONE} project=${PROJECT_ID}"
fi

echo "⏳ Waiting 20s for VM to boot..."
sleep 20

echo "⚙️ Preparing base packages on remote VM..."
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command "
  set -e
  sudo apt-get update -y >/dev/null
  sudo apt-get install -y python3-venv ffmpeg libsndfile1 git >/dev/null
" --project="$PROJECT_ID"

rm -rf "$TEMP_SRC"
mkdir -p "$TEMP_SRC/common" "$TEMP_SRC/modulator"
rsync -a --delete \
  --exclude '.env' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude 'artifacts' \
  --exclude 'out' \
  --exclude 'results' \
  --exclude '.docker_build' \
  "$MODULATOR_DIR/" "$TEMP_SRC/modulator/"
cp "$COMMON_DIR/modulate_local_common.sh" "$TEMP_SRC/common/"
cp "$COMMON_DIR/requirements.gcp.txt" "$TEMP_SRC/common/"

echo "📂 Copying modulator sample files to VM..."
gcloud compute ssh "$INSTANCE_NAME" --project="$PROJECT_ID" --zone="$ZONE" --command "mkdir -p $REMOTE_BASE_DIR $REMOTE_DIR"
gcloud compute scp --project="$PROJECT_ID" --recurse "$TEMP_SRC/modulator/." "$INSTANCE_NAME:$REMOTE_DIR/" --zone="$ZONE"
gcloud compute scp --project="$PROJECT_ID" "$TEMP_SRC/common/modulate_local_common.sh" "$INSTANCE_NAME:$REMOTE_BASE_DIR/modulate_local_common.sh" --zone="$ZONE"
gcloud compute scp --project="$PROJECT_ID" "$TEMP_SRC/common/requirements.gcp.txt" "$INSTANCE_NAME:$REMOTE_BASE_DIR/requirements.gcp.txt" --zone="$ZONE"
gcloud compute scp --project="$PROJECT_ID" "$MODULATOR_DIR/.env" "$INSTANCE_NAME:$REMOTE_DIR/.env" --zone="$ZONE"

case "${SDK_PACKAGE_SOURCE,,}" in
  pypi)
    REMOTE_PIP_INSTALL=$'python3 -m venv ~/.venvs/ephapsys-modulator\nsource ~/.venvs/ephapsys-modulator/bin/activate\npython -m pip install --upgrade pip >/dev/null\npython -m pip install "ephapsys[modulation,audio,vision,embedding,eval]=='"$SDK_VERSION"'" >/dev/null\npython -m pip install -r '"$REMOTE_BASE_DIR"'/requirements.gcp.txt >/dev/null\nif [ -f '"$REMOTE_DIR"'/requirements.txt ]; then python -m pip install -r '"$REMOTE_DIR"'/requirements.txt >/dev/null; fi'
    ;;
  testpypi)
    REMOTE_PIP_INSTALL=$'python3 -m venv ~/.venvs/ephapsys-modulator\nsource ~/.venvs/ephapsys-modulator/bin/activate\npython -m pip install --upgrade pip >/dev/null\npython -m pip install --extra-index-url https://pypi.org/simple --index-url https://test.pypi.org/simple "ephapsys[modulation,audio,vision,embedding,eval]=='"$SDK_VERSION"'" >/dev/null\npython -m pip install -r '"$REMOTE_BASE_DIR"'/requirements.gcp.txt >/dev/null\nif [ -f '"$REMOTE_DIR"'/requirements.txt ]; then python -m pip install -r '"$REMOTE_DIR"'/requirements.txt >/dev/null; fi'
    ;;
  *)
    echo "❌ Unsupported SDK package source for GCP modulation: $SDK_PACKAGE_SOURCE"
    exit 1
    ;;
esac

echo "📦 Installing remote Python environment..."
gcloud compute ssh "$INSTANCE_NAME" --project="$PROJECT_ID" --zone="$ZONE" --command "bash -lc $(printf '%q' "$REMOTE_PIP_INSTALL")"

REMOTE_RUN=$'cd '"$REMOTE_DIR"$'\nsource ~/.venvs/ephapsys-modulator/bin/activate\nexport MODULATOR_SKIP_SDK_SETUP=1\nchmod +x ./*.sh ../modulate_local_common.sh >/dev/null 2>&1 || true\nif [ -f ./modulate.sh ]; then exec ./modulate.sh; elif [ -f ./modulate_local.sh ]; then exec ./modulate_local.sh; else echo "❌ No modulate.sh or modulate_local.sh in '"$REMOTE_DIR"$'"; exit 1; fi\n'
if [ "$TRAIN_MODE" = "1" ]; then
  REMOTE_RUN="TRAIN_MODE=1"$'\n'"$REMOTE_RUN"
else
  REMOTE_RUN="TRAIN_MODE=0"$'\n'"$REMOTE_RUN"
fi

echo "🚀 Running $MODULATOR_KIND modulator remotely on VM..."
gcloud compute ssh "$INSTANCE_NAME" --project="$PROJECT_ID" --zone="$ZONE" -- -t "
  set -e
  bash -lc $(printf '%q' "$REMOTE_RUN")
  ls -lhR $REMOTE_DIR/artifacts* $REMOTE_DIR/out $REMOTE_DIR/results 2>/dev/null || true
"

echo "📥 Copying results locally..."
gcloud compute ssh "$INSTANCE_NAME" --project="$PROJECT_ID" --zone="$ZONE" --command "ls -lhR $REMOTE_DIR/artifacts* $REMOTE_DIR/out $REMOTE_DIR/results 2>/dev/null" || echo "⚠️ No remote artifacts found"
gcloud compute scp --project="$PROJECT_ID" --compress --recurse "$INSTANCE_NAME:$REMOTE_DIR/artifacts*" "$RESULTS_DIR/" --zone="$ZONE" 2>/dev/null || true
gcloud compute scp --project="$PROJECT_ID" --compress --recurse "$INSTANCE_NAME:$REMOTE_DIR/out" "$RESULTS_DIR/" --zone="$ZONE" 2>/dev/null || true
gcloud compute scp --project="$PROJECT_ID" --compress --recurse "$INSTANCE_NAME:$REMOTE_DIR/results" "$RESULTS_DIR/" --zone="$ZONE" 2>/dev/null || true

echo "✅ Results saved in $RESULTS_DIR"
