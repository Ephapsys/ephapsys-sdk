#!/bin/bash
set -euo pipefail

# USAGE:
#   ./modulate_gcp.sh t4
#   ./modulate_gcp.sh a100-4g
#   ./modulate_gcp.sh t4 --no-build
#   ./modulate_gcp.sh t4 --no-push
#
# Requires: Docker, gcloud, .env with API_KEY + MODEL_TEMPLATE_ID + BASE_URL
# Infra config should be in gcp.env (not committed), copied from gcp.env.example

# === Load infra config if available ===
if [ -f "gcp.env" ]; then
  echo "📂 Loading GCP config from gcp.env"
  # shellcheck source=/dev/null
  source gcp.env
fi

INSTANCE_NAME="ec-modulate-$(date +%s)"
DISK_SIZE="${DISK_SIZE:-200GB}"
REMOTE_DIR="~/epgpt2"
GPU_CHOICE="${1:-t4}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-modulate}"

RUN_TS=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="results/${EXPERIMENT_TAG}_${RUN_TS}"
mkdir -p "$RESULTS_DIR"

AUTO_DELETE="${AUTO_DELETE:-true}"
TRAIN_MODE=${TRAIN_MODE:-"1"}  # ✅ Fix 1: adds local TRAIN_MODE support (default=1)

# === Parse optional flags ===
SKIP_BUILD=false
SKIP_PUSH=false
if [[ "${2:-}" == "--no-build" ]]; then
  SKIP_BUILD=true
elif [[ "${2:-}" == "--no-push" ]]; then
  SKIP_PUSH=true
fi

# === Ensure .env exists ===
if [ ! -f ".env" ]; then
  echo "❌ Missing .env (must contain API_KEY, MODEL_TEMPLATE_ID, BASE_URL)"
  exit 1
fi

# === Source runtime envs from .env ===
export $(grep -v '^#' .env | xargs)

# Include Hugging Face token if present in .env
if grep -q '^HF_TOKEN=' .env; then
  echo "🔑 Found HF_TOKEN in .env"
  export HF_TOKEN=$(grep '^HF_TOKEN=' .env | cut -d '=' -f2)
else
  echo "⚠️ No HF_TOKEN found in .env; proceeding without authentication."
fi

# === Validate required runtime vars ===
if [ -z "${BASE_URL:-}" ]; then echo "❌ BASE_URL missing in .env"; exit 1; fi
if [ -z "${API_KEY:-}" ]; then echo "❌ API_KEY missing in .env"; exit 1; fi
if [ -z "${MODEL_TEMPLATE_ID:-}" ]; then echo "❌ MODEL_TEMPLATE_ID missing in .env"; exit 1; fi

# === Masked .env echo ===
MASKED="${API_KEY:0:8}********"
echo "🔐 Env loaded: BASE_URL=$BASE_URL, API_KEY=${MASKED}, MODEL_TEMPLATE_ID=$MODEL_TEMPLATE_ID"

# ==========================
# ⚡ GPU + MACHINE SELECTION
# ==========================
case "$GPU_CHOICE" in
  t4|T4) GPU_TYPE="nvidia-tesla-t4"; GPU_COUNT=1; MACHINE_TYPE="n1-standard-8";;
  a100|A100) MACHINE_TYPE="a2-highgpu-1g";;
  a100-2g|A100-2G) MACHINE_TYPE="a2-highgpu-2g";;
  a100-4g|A100-4G) MACHINE_TYPE="a2-highgpu-4g";;
  a100-8g|A100-8G) MACHINE_TYPE="a2-highgpu-8g";;
  *) echo "❌ Unknown GPU choice: $GPU_CHOICE"; exit 1;;
esac

# === CLEANUP TRAP ===
cleanup() {
  echo "⚠️ Cleaning up VM: $INSTANCE_NAME"
  if gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" >/dev/null 2>&1; then
    if [ "$AUTO_DELETE" = "true" ]; then
      gcloud compute instances delete "$INSTANCE_NAME" --zone="$ZONE" --quiet || true
    else
      gcloud compute instances stop "$INSTANCE_NAME" --zone="$ZONE" --quiet || true
    fi
  fi
}
trap cleanup EXIT

IMAGE_REF="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE_NAME:$TAG"

# --- Stage the local SDK into the Docker build context (so COPY works) ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCKERFILE_PATH="$SCRIPT_DIR/Dockerfile"

# Absolute path to your SDK (../../../SDK/python/ephapsys relative to language/)
SDK_SRC_REL="$SCRIPT_DIR/../../../SDK/python/ephapsys"
if [ ! -d "$SDK_SRC_REL" ]; then
  echo "❌ SDK not found at $SDK_SRC_REL"
  exit 1
fi

# Stage into context-local folder
TEMP_SDK_DIR="$SCRIPT_DIR/.docker_sdk/ephapsys"
rm -rf "$SCRIPT_DIR/.docker_sdk"
mkdir -p "$TEMP_SDK_DIR"
rsync -a --delete "$SDK_SRC_REL/" "$TEMP_SDK_DIR/"

# (Optional) Quiet Apple Silicon platform warning by default
: "${DOCKER_PLATFORM:=linux/amd64}"
export DOCKER_DEFAULT_PLATFORM="$DOCKER_PLATFORM"

# === Preflight: Ensure Artifact Registry API + repo ===
echo "🧰 Preflight: Checking Artifact Registry API and repo..."
gcloud services enable artifactregistry.googleapis.com --project="$PROJECT_ID" >/dev/null 2>&1 || true
if ! gcloud artifacts repositories describe "$REPO" --location="$REGION" --project="$PROJECT_ID" >/dev/null 2>&1; then
  echo "🗄️ Creating repo $REPO in $REGION"
  gcloud artifacts repositories create "$REPO" \
    --repository-format=docker \
    --location="$REGION" \
    --description="Ephapsys sample Docker images" \
    --project="$PROJECT_ID"
fi
gcloud auth configure-docker "$REGION-docker.pkg.dev" -q

# === Preflight: ensure VM service account can pull from Artifact Registry ===
VM_SA="$(gcloud iam service-accounts list \
  --project="$PROJECT_ID" \
  --filter='Compute Engine default service account' \
  --format='value(email)')"
if [ -z "$VM_SA" ]; then
  PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
  VM_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
fi
echo "🔐 Granting roles/artifactregistry.reader to $VM_SA (if needed)..."
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$VM_SA" \
  --role="roles/artifactregistry.reader" \
  --quiet >/dev/null

# === 1. Build + push Docker image (unless skipped) ===
if [ "$SKIP_BUILD" = false ]; then
  echo "⏳ [STEP 1] Building Docker image (linux/amd64)..."
  docker buildx build --platform linux/amd64 \
  --build-arg HF_TOKEN="$HF_TOKEN" \
  -f "$DOCKERFILE_PATH" -t "$IMAGE_REF" "$SCRIPT_DIR"

  if [ "$SKIP_PUSH" = false ]; then
    echo "⏳ [STEP 2] Pushing image to Artifact Registry..."
    docker push "$IMAGE_REF"
  else
    echo "⚡ Built image locally, skipped push (--no-push)"
  fi
else
  echo "⚡ Skipping Docker build/push, reusing last pushed image..."
fi

# === 1.9 Ensure Cloud NAT exists (fallback for private egress) ===
echo "🌐 Checking for Cloud NAT configuration..."
if ! gcloud compute routers describe ephapsys-nat-router \
      --region="$REGION" --project="$PROJECT_ID" >/dev/null 2>&1; then
  echo "🛠 Creating Cloud NAT router and config (first-time setup)..."
  gcloud compute routers create ephapsys-nat-router \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --network=default
  gcloud compute routers nats create ephapsys-nat-config \
    --project="$PROJECT_ID" \
    --router=ephapsys-nat-router \
    --auto-allocate-nat-external-ips \
    --nat-all-subnet-ip-ranges \
    --region="$REGION"
  echo "✅ Cloud NAT configured for region $REGION"
else
  echo "✅ Cloud NAT already configured."
fi

# === 2. Create VM ===
echo "⏳ [STEP 3] Creating VM: $INSTANCE_NAME (gpu=$GPU_CHOICE)..."
if [[ "$GPU_CHOICE" =~ t4|T4 ]]; then
  gcloud compute instances create "$INSTANCE_NAME" \
    --project="$PROJECT_ID" --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
    --image-family="pytorch-2-7-cu128-ubuntu-2204-nvidia-570" \
    --image-project="deeplearning-platform-release" \
    --boot-disk-size="$DISK_SIZE" --maintenance-policy=TERMINATE \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --metadata="install-nvidia-driver=True" \
    --network=default --subnet=default --address="" \
    --restart-on-failure --labels=job=modulate,gpu="$GPU_CHOICE",tag="$EXPERIMENT_TAG"
else
  gcloud compute instances create "$INSTANCE_NAME" \
    --project="$PROJECT_ID" --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --image-family="pytorch-2-7-cu128-ubuntu-2204-nvidia-570" \
    --image-project="deeplearning-platform-release" \
    --boot-disk-size="$DISK_SIZE" --maintenance-policy=TERMINATE \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --metadata="install-nvidia-driver=True" \
    --network=default --subnet=default --address="" \
    --restart-on-failure --labels=job=modulate,gpu="$GPU_CHOICE",tag="$EXPERIMENT_TAG"
fi

echo "⏳ Waiting 60s for VM to boot..."
sleep 60

# === 2.1 Ensure HTTPS egress rule exists for trainer VMs ===
echo "🌐 Ensuring outbound HTTPS (443) egress is allowed for Hugging Face access..."
if ! gcloud compute firewall-rules describe allow-egress-https --project="$PROJECT_ID" >/dev/null 2>&1; then
  gcloud compute firewall-rules create allow-egress-https \
    --project="$PROJECT_ID" \
    --network=default \
    --direction=EGRESS \
    --priority=1000 \
    --action=ALLOW \
    --rules=tcp:443 \
    --destination-ranges=0.0.0.0/0 \
    --description="Allow HTTPS egress for Hugging Face and external metric downloads"
  echo "✅ Created firewall rule: allow-egress-https"
else
  echo "✅ Firewall rule allow-egress-https already exists."
fi

# === 2.5 Install Docker + NVIDIA runtime on VM (idempotent) ===
echo "⚙️ Ensuring Docker + NVIDIA runtime is installed..."
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command "
  set -e
  if ! command -v docker &>/dev/null; then
    echo '⬇️ Installing Docker + NVIDIA runtime...'
    sudo apt-get update -y
    sudo apt-get install -y docker.io curl gnupg
    sudo systemctl enable docker
    sudo usermod -aG docker \$USER
    distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)
    curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/libnvidia-container/\$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update -y
    sudo apt-get install -y nvidia-docker2
    sudo systemctl restart docker
  else
    echo '⚡ Docker already installed, skipping setup.'
  fi
"

# === 2.6 Ensure Docker default network uses host NAT ===
echo "🌐 Ensuring Docker bridge network can reach the Internet..."
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command "
  set -e
  if ! sudo docker run --rm --network bridge curlimages/curl -s -o /dev/null -w '%{http_code}' https://huggingface.co | grep -q 200; then
    echo '⚙️  Docker bridge has no egress; switching to host networking fallback for this VM...'
    echo '{\"bip\":\"172.17.0.1/16\"}' | sudo tee /etc/docker/daemon.json >/dev/null
    sudo systemctl restart docker
  else
    echo '✅ Docker networking verified (can reach Hugging Face).'
  fi
"

# === 3. Copy .env to VM ===
echo "📂 Copying .env to VM..."
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command "mkdir -p $REMOTE_DIR"
gcloud compute scp ".env" "$INSTANCE_NAME:$REMOTE_DIR/.env" --zone="$ZONE"

# === 4. Run container on VM ===
TRAIN_FLAG=""
if [ "$TRAIN_MODE" = "1" ]; then
  TRAIN_FLAG="--train_mode"
  echo "🧠 TRAIN_MODE enabled (default)."
else
  echo "🧠 TRAIN_MODE disabled."
fi

echo "🚀 Running container remotely..."
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" -- -t "
  set -e
  mkdir -p $REMOTE_DIR/artifacts
  gcloud auth configure-docker $REGION-docker.pkg.dev -q
  gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin https://$REGION-docker.pkg.dev
  docker run --rm --gpus all \
    --network host \
    --env-file $REMOTE_DIR/.env \
    -e HF_TOKEN=\"$HF_TOKEN\" \
    -v $REMOTE_DIR/artifacts:/app/artifacts \
    $IMAGE_REF \
    --base_url $BASE_URL \
    --api_key $API_KEY \
    --model_template_id $MODEL_TEMPLATE_ID \
    --outdir artifacts $TRAIN_FLAG
  echo '🧩 Verifying artifacts on VM...'
  ls -lhR $REMOTE_DIR/artifacts || echo '⚠️  No artifacts found.'
  sync
"

# === 5. Copy results back (retry safe)
echo "📥 Copying results locally..."
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command "ls -lhR $REMOTE_DIR/artifacts" || echo "⚠️ No remote artifacts found"
gcloud compute scp --compress --recurse "$INSTANCE_NAME:$REMOTE_DIR/artifacts/*" "$RESULTS_DIR/" --zone="$ZONE" || true

if [ -z "$(ls -A "$RESULTS_DIR" 2>/dev/null)" ]; then
  echo "⚠️ No artifacts copied on first attempt, retrying..."
  gcloud compute scp --recurse "$INSTANCE_NAME:$REMOTE_DIR/artifacts" "$RESULTS_DIR/" --zone="$ZONE" || true
fi

echo "✅ Results saved in $RESULTS_DIR"
echo "✅ VM cleanup will run automatically (trap)."
