#!/usr/bin/env bash
# ============================================================
# Run HelloWorld agent sample on a persistent GCP VM.
#
# Usage:
#   ./run_gcp.sh --staging          # install SDK from TestPyPI (default)
#   ./run_gcp.sh --production       # install SDK from PyPI
#   ./run_gcp.sh --zone us-central1-a --machine-type e2-standard-4
#   ./run_gcp.sh --attach        # auto-attach to tmux after provisioning
#
# Requirements:
#   - gcloud CLI authenticated with access to ephapsys-development
#   - helloworld/.env populated with AOC creds (copied to VM)
#   - SDK/python/pyproject.toml contains the version to install
# ============================================================

set -euo pipefail

BLUE="\033[36m"
GREEN="\033[32m"
YELLOW="\033[33m"
MAGENTA="\033[35m"
RESET="\033[0m"

success() {
  printf "${GREEN}[DONE]${RESET} %s\n" "$1"
}

warn() {
  printf "${YELLOW}[WARN]${RESET} %s\n" "$1"
}

info() {
  printf "${BLUE}[INFO]${RESET} %s\n" "$1"
}

UPLOAD_ENV_FILE=""
cleanup_temp_files() {
  if [ -n "$UPLOAD_ENV_FILE" ] && [ -f "$UPLOAD_ENV_FILE" ]; then
    rm -f "$UPLOAD_ENV_FILE"
  fi
}
trap cleanup_temp_files EXIT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"
PYPROJECT="$REPO_ROOT/Product/SDK/python/pyproject.toml"
PROJECT_ID="${PROJECT_ID:-ephapsys-development}"
ZONE="${ZONE:-us-central1-a}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-standard-4}"
DISK_SIZE="${DISK_SIZE:-50GB}"
IMAGE_FAMILY="${IMAGE_FAMILY:-ubuntu-2204-lts}"
IMAGE_PROJECT="${IMAGE_PROJECT:-ubuntu-os-cloud}"
MODE="staging"
INSTANCE_PREFIX="${INSTANCE_PREFIX:-hello-agent}"
INTERACTIVE=true
CPU_ONLY=true
GPU_TYPE="${GPU_TYPE:-t4}"
GPU_MACHINE_TYPE="${GPU_MACHINE_TYPE:-n1-standard-8}"
GPU_COUNT="${GPU_COUNT:-1}"
GPU_IMAGE_FAMILY="${GPU_IMAGE_FAMILY:-pytorch-2-7-cu128-ubuntu-2204-nvidia-570}"
META_FILE="$SCRIPT_DIR/.last_gcp_instance"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --staging) MODE="staging"; shift ;;
    --production) MODE="production"; shift ;;
    --zone) ZONE="$2"; shift 2 ;;
    --machine-type) MACHINE_TYPE="$2"; shift 2 ;;
    --disk-size) DISK_SIZE="$2"; shift 2 ;;
    --project) PROJECT_ID="$2"; shift 2 ;;
    --instance-prefix) INSTANCE_PREFIX="$2"; shift 2 ;;
    --interactive) INTERACTIVE=true; shift ;;
    --no-interactive) INTERACTIVE=false; shift ;;
    --gpu)
      CPU_ONLY=false
      shift
      ;;
    --gpu-type)
      CPU_ONLY=false
      GPU_TYPE="$2"
      shift 2
      ;;
    --gpu-machine-type)
      CPU_ONLY=false
      GPU_MACHINE_TYPE="$2"
      shift 2
      ;;
    --gpu-count)
      CPU_ONLY=false
      GPU_COUNT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

RAW_ANCHOR="${PERSONALIZE_ANCHOR:-none}"
ANCHOR_SUFFIX="$(echo "$RAW_ANCHOR" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-')"
ANCHOR_SUFFIX="$(echo "$ANCHOR_SUFFIX" | sed 's/^-*//; s/-*$//')"
if [ -z "$ANCHOR_SUFFIX" ]; then
  ANCHOR_SUFFIX="none"
fi
INSTANCE_NAME="${INSTANCE_PREFIX}-$(date +%s)-${ANCHOR_SUFFIX}"
REMOTE_DIR="helloworld"

if ! command -v gcloud >/dev/null 2>&1; then
  printf "${MAGENTA}❌ gcloud CLI not found. Install and authenticate first.${RESET}\n"
  exit 1
fi

GLOBAL_ENV_DIR="$REPO_ROOT/Product"
ENV_FILE_LOCAL="$SCRIPT_DIR/.env.stag"
GLOBAL_ENV_FILE="$GLOBAL_ENV_DIR/.env.stag"
if [ "$MODE" = "production" ]; then
  ENV_FILE_LOCAL="$SCRIPT_DIR/.env.prod"
  GLOBAL_ENV_FILE="$GLOBAL_ENV_DIR/.env.prod"
fi

if [ -f "$GLOBAL_ENV_FILE" ]; then
  info "Using global env file $GLOBAL_ENV_FILE"
  ACTIVE_ENV_FILE="$GLOBAL_ENV_FILE"
else
  ACTIVE_ENV_FILE="$ENV_FILE_LOCAL"
fi

if [ ! -f "$ACTIVE_ENV_FILE" ]; then
  printf "${MAGENTA}❌ Missing env file. Expected one of %s or %s${RESET}\n" "$GLOBAL_ENV_FILE" "$ENV_FILE_LOCAL"
  exit 1
fi

source "$ACTIVE_ENV_FILE"
export AOC_API_URL AOC_API_KEY AGENT_TEMPLATE_ID PERSONALIZE_ANCHOR
HSM_KMS_KEY="${HSM_KMS_KEY:-}"
HSM_KMS_ENDPOINT="${HSM_KMS_ENDPOINT:-}"
HSM_KMS_CREDENTIALS="${HSM_KMS_CREDENTIALS:-}"
HSM_SLOT="${HSM_SLOT:-}"
HSM_KEY_LABEL="${HSM_KEY_LABEL:-}"
export HSM_KMS_KEY HSM_KMS_ENDPOINT HSM_KMS_CREDENTIALS HSM_SLOT HSM_KEY_LABEL
RAW_ANCHOR="${PERSONALIZE_ANCHOR:-none}"
ANCHOR_SUFFIX="$(echo "$RAW_ANCHOR" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-')"
ANCHOR_SUFFIX="$(echo "$ANCHOR_SUFFIX" | sed 's/^-*//; s/-*$//')"
if [ -z "$ANCHOR_SUFFIX" ]; then
  ANCHOR_SUFFIX="none"
fi
INSTANCE_NAME="${INSTANCE_PREFIX}-$(date +%s)-${ANCHOR_SUFFIX}"
REMOTE_DIR="helloworld"
REQUIRED_VARS=(AOC_API_URL AOC_API_KEY AGENT_TEMPLATE_ID)
for var in "${REQUIRED_VARS[@]}"; do
if [ -z "${!var:-}" ]; then
  printf "${MAGENTA}❌ %s is missing in %s${RESET}\n" "$var" "$ACTIVE_ENV_FILE"
  exit 1
fi
done

ANCHOR_TYPE="$(echo "$RAW_ANCHOR" | tr '[:upper:]' '[:lower:]')"
if [ -z "$ANCHOR_TYPE" ]; then
  ANCHOR_TYPE="none"
fi
ANCHOR_IS_TPM=false
ANCHOR_IS_HSM=false
case "$ANCHOR_TYPE" in
  tpm) ANCHOR_IS_TPM=true ;;
  hsm) ANCHOR_IS_HSM=true ;;
esac

if $ANCHOR_IS_HSM; then
  if [ -z "${HSM_HELPER:-}" ] && [ -z "${HSM_KMS_KEY:-}" ]; then
    info "HSM_KMS_KEY not set; attempting automatic Cloud KMS provisioning..."
    if [ -z "${COMPUTE_SERVICE_ACCOUNT:-}" ]; then
      PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)' 2>/dev/null || true)"
      if [ -n "$PROJECT_NUMBER" ]; then
        COMPUTE_SERVICE_ACCOUNT="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
      else
        printf "${MAGENTA}❌ Unable to determine project number. Set COMPUTE_SERVICE_ACCOUNT and rerun.${RESET}\n"
        exit 1
      fi
    fi
    PROVISION_ARGS=(
      --project "$PROJECT_ID"
      --service-account "$COMPUTE_SERVICE_ACCOUNT"
      --location "$HSM_KMS_LOCATION"
      --keyring "$HSM_KMS_KEY_RING"
      --key "$HSM_KMS_KEY_NAME"
    )
    if [ "${HSM_KMS_USE_HSM}" = "1" ]; then
      PROVISION_ARGS+=(--hsm)
    fi
    PROVISION_OUTPUT="$("$SCRIPT_DIR/provision_kms_key.sh" "${PROVISION_ARGS[@]}")"
    echo "$PROVISION_OUTPUT"
    HSM_KMS_KEY="$(echo "$PROVISION_OUTPUT" | awk -F= '/^HSM_KMS_KEY=/{print $2}' | tail -n1)"
    if [ -z "$HSM_KMS_KEY" ]; then
      printf "${MAGENTA}❌ provision_kms_key.sh did not return an HSM_KMS_KEY. Aborting.${RESET}\n"
      exit 1
    fi
    export HSM_KMS_KEY
    info "🔐 Auto-provisioned Cloud KMS key: ${HSM_KMS_KEY}"
  fi
  if [ -z "${HSM_HELPER:-}" ]; then
    if [ -z "${HSM_KMS_KEY:-}" ]; then
      printf "${MAGENTA}❌ HSM_KMS_KEY is still unset. Provide it in your env or set HSM_HELPER.${RESET}\n"
      exit 1
    fi
    info "🔐 HSM mode: using Cloud KMS key ${HSM_KMS_KEY}"
  else
    info "🔐 HSM mode: using custom helper command (${HSM_HELPER})"
  fi
fi

LOCAL_KMS_CREDS="${HSM_KMS_CREDENTIALS:-}"
REMOTE_KMS_CREDS=""
if $ANCHOR_IS_HSM && [ -n "$LOCAL_KMS_CREDS" ]; then
  if [ ! -f "$LOCAL_KMS_CREDS" ]; then
    printf "${MAGENTA}❌ HSM_KMS_CREDENTIALS points to %s but the file was not found.${RESET}\n" "$LOCAL_KMS_CREDS"
    exit 1
  fi
  REMOTE_KMS_CREDS="~/${REMOTE_DIR}/.secrets/kms-credentials.json"
fi

CURRENT_ACCOUNT="$(gcloud config get-value account 2>/dev/null || true)"
if [ -n "$CURRENT_ACCOUNT" ]; then
  warn "Ensuring $CURRENT_ACCOUNT has compute.instanceAdmin.v1 on $PROJECT_ID (required for gcloud ssh)"
  if ! gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="user:$CURRENT_ACCOUNT" \
    --role="roles/compute.instanceAdmin.v1" \
    --quiet >/dev/null 2>&1; then
    warn "IAM auto-grant skipped (role may already exist or gcloud cannot write credentials). If SSH fails, grant roles/compute.instanceAdmin.v1 manually."
  fi
else
  warn "gcloud has no active account; IAM auto-grant skipped. Ensure your account has roles/compute.instanceAdmin.v1."
fi

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
  printf "${MAGENTA}❌ Unable to read SDK version from SDK/python/pyproject.toml${RESET}\n"
  exit 1
fi

if [ "$MODE" = "staging" ]; then
  PIP_INSTALL_CMD="pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple ephapsys==${SDK_VERSION}"
  TARGET_REGISTRY="TestPyPI"
else
  PIP_INSTALL_CMD="pip install ephapsys==${SDK_VERSION}"
  TARGET_REGISTRY="PyPI"
fi

create_instance() {
  local machine="$MACHINE_TYPE"
  local image_family="$IMAGE_FAMILY"
  local image_project="$IMAGE_PROJECT"
  local extra_args=()

  if [ "$CPU_ONLY" = false ]; then
    machine="$GPU_MACHINE_TYPE"
    image_family="$GPU_IMAGE_FAMILY"
    image_project="deeplearning-platform-release"
    local accel_type
    case "${GPU_TYPE,,}" in
      t4)
        accel_type="nvidia-tesla-t4"
        ;;
      a100)
        accel_type="nvidia-tesla-a100"
        machine="a2-highgpu-${GPU_COUNT}g"
        ;;
      l4)
        accel_type="nvidia-l4"
        machine="g2-standard-${GPU_COUNT}"
        ;;
      *)
        accel_type="$GPU_TYPE"
        ;;
    esac
    extra_args+=(
      --accelerator="type=${accel_type},count=${GPU_COUNT}"
      --metadata="install-nvidia-driver=True"
      --maintenance-policy=TERMINATE
    )
  fi

  gcloud compute instances create "$INSTANCE_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --machine-type="$machine" \
    --shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --image-family="$image_family" \
    --image-project="$image_project" \
    --boot-disk-size="$DISK_SIZE" \
    --boot-disk-type=pd-ssd \
    --tags=helloworld-agent \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    ${extra_args[@]+"${extra_args[@]}"}
}

SELECTED_MACHINE="$MACHINE_TYPE"
if [ "$CPU_ONLY" = false ]; then
  SELECTED_MACHINE="$GPU_MACHINE_TYPE"
fi

info "🚀 Launching HelloWorld VM"
printf "  ${BLUE}Project${RESET}      : %s\n" "$PROJECT_ID"
printf "  ${BLUE}Zone${RESET}         : %s\n" "$ZONE"
printf "  ${BLUE}Machine type${RESET} : %s\n" "$SELECTED_MACHINE"
printf "  ${BLUE}Disk size${RESET}    : %s\n" "$DISK_SIZE"
printf "  ${BLUE}SDK version${RESET}  : %s (%s)\n" "$SDK_VERSION" "$TARGET_REGISTRY"
printf "  ${BLUE}GPU mode${RESET}     : %s\n" "$([ "$CPU_ONLY" = true ] && echo "disabled" || echo "$GPU_TYPE (count=$GPU_COUNT)")"
printf "  ${BLUE}Instance${RESET}     : %s\n" "$INSTANCE_NAME"
printf "  ${BLUE}Anchor${RESET}       : %s\n" "$RAW_ANCHOR"

create_instance

cat <<EOF > "$META_FILE"
INSTANCE_NAME=$INSTANCE_NAME
PROJECT_ID=$PROJECT_ID
ZONE=$ZONE
ANCHOR=$RAW_ANCHOR
EOF

info "📂 Preparing remote workspace"
SSH_CMD=(gcloud compute ssh "$INSTANCE_NAME" --project="$PROJECT_ID" --zone="$ZONE" --command)
if ! "${SSH_CMD[@]}" "mkdir -p ~/${REMOTE_DIR}"; then
  warn "Initial SSH attempt failed (VM may still be booting). Retrying in 5s..."
  sleep 5
  if ! "${SSH_CMD[@]}" "mkdir -p ~/${REMOTE_DIR}"; then
    warn "SSH still failing. Consider running: gcloud compute ssh $INSTANCE_NAME --project=$PROJECT_ID --zone=$ZONE --troubleshoot"
    exit 1
  fi
fi

if [ -n "$REMOTE_KMS_CREDS" ]; then
  info "🔐 Uploading Cloud KMS credentials to VM"
  "${SSH_CMD[@]}" "mkdir -p ~/${REMOTE_DIR}/.secrets && chmod 700 ~/${REMOTE_DIR}/.secrets"
  gcloud compute scp "$LOCAL_KMS_CREDS" "${INSTANCE_NAME}:${REMOTE_KMS_CREDS}" \
    --project="$PROJECT_ID" --zone="$ZONE"
  gcloud compute ssh "$INSTANCE_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --command="chmod 600 ${REMOTE_KMS_CREDS}"
fi

info "📤 Copying sample files to VM"
SYNC_DIR="$(mktemp -d)"
TEMP_SRC="$SYNC_DIR/src"
mkdir -p "$TEMP_SRC"
rsync -a \
  --exclude ".ephapsys_state" \
  --exclude ".DS_Store" \
  --exclude "README.md" \
  --exclude ".env*" \
  "$SCRIPT_DIR/" "$TEMP_SRC/"
gcloud compute scp --recurse "$TEMP_SRC/." "${INSTANCE_NAME}:~/${REMOTE_DIR}/" \
  --project="$PROJECT_ID" --zone="$ZONE"
rm -rf "$SYNC_DIR"

UPLOAD_ENV_FILE="$(mktemp)"
cp "$ACTIVE_ENV_FILE" "$UPLOAD_ENV_FILE"
if [ -n "$REMOTE_KMS_CREDS" ]; then
  cat <<EOF >> "$UPLOAD_ENV_FILE"

# Added by run_gcp.sh for Cloud KMS access on the VM
HSM_KMS_CREDENTIALS=$REMOTE_KMS_CREDS
GOOGLE_APPLICATION_CREDENTIALS=$REMOTE_KMS_CREDS
EOF
fi

info "📄 Uploading env file $(basename "$ACTIVE_ENV_FILE")"
gcloud compute scp "$UPLOAD_ENV_FILE" "${INSTANCE_NAME}:~/${REMOTE_DIR}/.env" \
  --project="$PROJECT_ID" --zone="$ZONE"

BOOTSTRAP=$(cat <<'EOS'
cat <<'SCRIPT' > ~/HELLOROOT/.bootstrap.sh
#!/usr/bin/env bash
set -euo pipefail
cd ~/HELLOROOT
VM_BLUE='\033[36m'
VM_RESET='\033[0m'
export DEBIAN_FRONTEND=noninteractive
echo -e "${VM_BLUE}[VM] STEP 1/4: Updating apt packages...${VM_RESET}"
sudo apt-get update -y -qq >/dev/null
echo -e "${VM_BLUE}[VM] STEP 2/4: Installing Python + system packages...${VM_RESET}"
sudo apt-get install -y -qq python3 python3-venv python3-pip tmux pkg-config ANCHOR_APT_PKGS_PLACEHOLDER >/dev/null
TPM_SNIPPET_PLACEHOLDER
echo -e "${VM_BLUE}[VM] STEP 3/4: Creating virtualenv & upgrading pip...${VM_RESET}"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install 'numpy<2'
TORCH_INSTALL_PLACEHOLDER
echo -e "${VM_BLUE}[VM] STEP 4/4: Installing Ephapsys SDK + deps (sit tight, this download is hefty)...${VM_RESET}"
PIP_INSTALL_PLACEHOLDER
if [ -f .env ]; then
  echo -e "${VM_BLUE}[VM] Loading .env so TrustedAgent has API credentials${VM_RESET}"
  set -a
  source .env
  export AOC_API_URL AOC_API_KEY AGENT_TEMPLATE_ID PERSONALIZE_ANCHOR
  set +a
  echo -e "${VM_BLUE}[VM] Personalization anchor=${PERSONALIZE_ANCHOR:-none}${VM_RESET}"
else
echo "[VM][WARN] Missing .env on VM; set env vars manually before running."
fi
nohup bash -c 'cd ~/HELLOROOT && source .venv/bin/activate && python helloworld_agent.py >> helloworld.log 2>&1' >/dev/null 2>&1 &
echo $! > helloworld.pid
echo -e "${VM_BLUE}[VM] Bot started in background (PID $(cat helloworld.pid)).${VM_RESET}"
SCRIPT
chmod +x ~/HELLOROOT/.bootstrap.sh
bash ~/HELLOROOT/.bootstrap.sh
EOS
)

BOOTSTRAP="${BOOTSTRAP//HELLOROOT/${REMOTE_DIR}}"

APT_EXTRA_PKGS=""
if $ANCHOR_IS_TPM; then
  APT_EXTRA_PKGS="libtss2-dev tpm2-tools"
fi
TPM_BLOCK=""
if $ANCHOR_IS_TPM; then
TPM_BLOCK=$(cat <<'TPM'
if systemctl list-unit-files | grep -q tpm2-abrmd; then
  sudo systemctl enable --now tpm2-abrmd >/dev/null 2>&1 || true
fi
CURRENT_USER="$(whoami)"
if [ -c /dev/tpmrm0 ]; then
  echo -e "${VM_BLUE}[VM] Configuring access to /dev/tpmrm0${VM_RESET}"
  sudo chown "$CURRENT_USER":"$CURRENT_USER" /dev/tpmrm0 /dev/tpm0 >/dev/null 2>&1 || sudo chmod 666 /dev/tpmrm0 /dev/tpm0 >/dev/null 2>&1
else
  echo "[VM][WARN] No /dev/tpmrm0 device detected; TPM anchor may fail."
fi
TPM
)
else
TPM_BLOCK=$(cat <<'TPM'
echo -e "${VM_BLUE}[VM] TPM stack skipped (anchor=ANCHOR_KIND_PLACEHOLDER).${VM_RESET}"
TPM
)
fi

BOOTSTRAP="${BOOTSTRAP/ANCHOR_APT_PKGS_PLACEHOLDER/${APT_EXTRA_PKGS}}"
BOOTSTRAP="${BOOTSTRAP/TPM_SNIPPET_PLACEHOLDER/${TPM_BLOCK}}"
BOOTSTRAP="${BOOTSTRAP//ANCHOR_KIND_PLACEHOLDER/${ANCHOR_TYPE}}"

TORCH_VERSION="${TORCH_VERSION:-2.2.2}"
if [ "$CPU_ONLY" = true ]; then
  TORCH_CMD="pip install --index-url https://download.pytorch.org/whl/cpu torch==${TORCH_VERSION} --no-cache-dir"
else
  TORCH_CMD="echo \"⚠️ Installing CUDA-enabled torch (large download)\"; pip install torch==${TORCH_VERSION} --no-cache-dir"
fi

BOOTSTRAP="${BOOTSTRAP/TORCH_INSTALL_PLACEHOLDER/${TORCH_CMD}}"
BOOTSTRAP="${BOOTSTRAP/PIP_INSTALL_PLACEHOLDER/${PIP_INSTALL_CMD} --no-cache-dir}"

info "⚙️  Provisioning dependencies on VM (this step can take several minutes)"
gcloud compute ssh "$INSTANCE_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --command="$BOOTSTRAP"

cat <<EOF
${GREEN}✅ HelloWorld agent is running in the background on $INSTANCE_NAME.${RESET}

Stream logs with:
  gcloud compute ssh $INSTANCE_NAME --project $PROJECT_ID --zone $ZONE -- -t "tail -f ~/helloworld/helloworld.log"

To stop the bot:
  gcloud compute ssh $INSTANCE_NAME --project $PROJECT_ID --zone $ZONE -- -t "pkill -f helloworld_agent.py"

Remember to stop/delete the VM manually when done:
  gcloud compute instances stop $INSTANCE_NAME --project $PROJECT_ID --zone $ZONE
  # or
  gcloud compute instances delete $INSTANCE_NAME --project $PROJECT_ID --zone $ZONE
EOF
info "Logs: gcloud compute ssh $INSTANCE_NAME --project $PROJECT_ID --zone $ZONE -- -t \"tail -f ~/helloworld/helloworld.log\""
if [ "$INTERACTIVE" = true ]; then
  info "Opening interactive session..."
  gcloud compute ssh "$INSTANCE_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    -- -t "cd ~/helloworld && echo \"[VM] Exporting .env for interactive run\" && set -a && source .env && set +a && source .venv/bin/activate && python helloworld_agent.py"
fi
