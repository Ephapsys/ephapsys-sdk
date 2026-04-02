#!/usr/bin/env bash
set -euo pipefail

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

if [ -f "$MODULATOR_DIR/gcp.env" ]; then
  echo "📂 Loading GCP config from $MODULATOR_DIR/gcp.env"
  # shellcheck source=/dev/null
  source "$MODULATOR_DIR/gcp.env"
elif [ -f "$COMMON_DIR/gcp.env" ]; then
  echo "📂 Loading GCP config from $COMMON_DIR/gcp.env"
  # shellcheck source=/dev/null
  source "$COMMON_DIR/gcp.env"
fi

GPU_CHOICE="${1:-t4}"
shift || true

SKIP_BUILD=false
SKIP_PUSH=false
while [ $# -gt 0 ]; do
  case "$1" in
    --no-build)
      SKIP_BUILD=true
      ;;
    --no-push)
      SKIP_PUSH=true
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

if [ -z "${PROJECT_ID:-}" ] || [ -z "${REGION:-}" ] || [ -z "${ZONE:-}" ] || [ -z "${REPO:-}" ]; then
  echo "❌ PROJECT_ID, REGION, ZONE, and REPO must be set via gcp.env"
  exit 1
fi

if [ -z "$BASE_URL" ] || [ -z "$AOC_ORG_ID" ] || [ -z "$AOC_MODULATION_TOKEN" ] || [ -z "$MODEL_TEMPLATE_ID" ]; then
  echo "❌ BASE_URL/AOC_ORG_ID/AOC_MODULATION_TOKEN/MODEL_TEMPLATE_ID missing in .env"
  exit 1
fi

INSTANCE_NAME="ec-modulate-${MODULATOR_KIND}-$(date +%s)"
DISK_SIZE="${DISK_SIZE:-200GB}"
REMOTE_DIR="${REMOTE_DIR:-~/ephapsys-${MODULATOR_KIND}}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-${MODULATOR_KIND}-modulate}"
AUTO_DELETE="${AUTO_DELETE:-true}"
TRAIN_MODE="${TRAIN_MODE:-1}"
RUN_TS="$(date +"%Y%m%d_%H%M%S")"
RESULTS_DIR="$MODULATOR_DIR/results/${EXPERIMENT_TAG}_${RUN_TS}"
mkdir -p "$RESULTS_DIR"

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

REPO_ROOT="$(cd "$COMMON_DIR/../.." && pwd)"
SDK_SRC="$REPO_ROOT/sdk/python/ephapsys"
DOCKERFILE_PATH="$COMMON_DIR/Dockerfile.gcp"
BUILD_CTX="$MODULATOR_DIR/.docker_build"
IMAGE_NAME="${IMAGE_NAME:-ephapsys-modulator-${MODULATOR_KIND}}"
TAG="${TAG:-$RUN_TS}"
IMAGE_REF="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE_NAME:$TAG"

rm -rf "$BUILD_CTX"
mkdir -p "$BUILD_CTX/.docker_sdk/ephapsys" "$BUILD_CTX/modulator"
rsync -a --delete "$SDK_SRC/" "$BUILD_CTX/.docker_sdk/ephapsys/"
rsync -a --delete \
  --exclude '.env' \
  --exclude '.venv' \
  --exclude '.docker_build' \
  --exclude '.docker_sdk' \
  --exclude 'artifacts' \
  --exclude 'results' \
  "$MODULATOR_DIR/" "$BUILD_CTX/modulator/"
cp "$COMMON_DIR/requirements.gcp.txt" "$BUILD_CTX/requirements.gcp.txt"

: "${DOCKER_PLATFORM:=linux/amd64}"
export DOCKER_DEFAULT_PLATFORM="$DOCKER_PLATFORM"

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

if [ "$SKIP_BUILD" = false ]; then
  echo "⏳ [STEP 1] Building Docker image for $MODULATOR_KIND..."
  docker buildx build --platform linux/amd64 \
    --build-arg HF_TOKEN="$HF_TOKEN" \
    -f "$DOCKERFILE_PATH" \
    -t "$IMAGE_REF" \
    "$BUILD_CTX"

  if [ "$SKIP_PUSH" = false ]; then
    echo "⏳ [STEP 2] Pushing image to Artifact Registry..."
    docker push "$IMAGE_REF"
  else
    echo "⚡ Built image locally, skipped push (--no-push)"
  fi
else
  echo "⚡ Skipping Docker build/push, reusing last pushed image..."
fi

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
else
  echo "✅ Cloud NAT already configured."
fi

echo "⏳ [STEP 3] Creating VM: $INSTANCE_NAME (gpu=$GPU_CHOICE)..."
if [[ "$GPU_CHOICE" =~ ^(t4|T4|l4|L4|v100|V100|p100|P100)$ ]]; then
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
    --restart-on-failure --labels=job=modulate,gpu="$GPU_CHOICE",tag="$EXPERIMENT_TAG",kind="$MODULATOR_KIND"
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
    --restart-on-failure --labels=job=modulate,gpu="$GPU_CHOICE",tag="$EXPERIMENT_TAG",kind="$MODULATOR_KIND"
fi

echo "⏳ Waiting 60s for VM to boot..."
sleep 60

echo "🌐 Ensuring outbound HTTPS (443) egress is allowed..."
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
fi

echo "⚙️ Ensuring Docker + NVIDIA runtime is installed..."
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command "
  set -e
  if ! command -v docker >/dev/null 2>&1; then
    sudo apt-get update -y
    sudo apt-get install -y docker.io curl gnupg
    sudo systemctl enable docker
    sudo usermod -aG docker \$USER
    distribution=\$(. /etc/os-release; echo \$ID\$VERSION_ID)
    curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/libnvidia-container/\$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update -y
    sudo apt-get install -y nvidia-docker2
    sudo systemctl restart docker
  fi
"

echo "🌐 Ensuring Docker bridge has egress..."
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command "
  set -e
  if ! sudo docker run --rm --network bridge curlimages/curl -s -o /dev/null -w '%{http_code}' https://huggingface.co | grep -q 200; then
    echo '{\"bip\":\"172.17.0.1/16\"}' | sudo tee /etc/docker/daemon.json >/dev/null
    sudo systemctl restart docker
  fi
"

echo "📂 Copying .env to VM..."
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command "mkdir -p $REMOTE_DIR"
gcloud compute scp "$MODULATOR_DIR/.env" "$INSTANCE_NAME:$REMOTE_DIR/.env" --zone="$ZONE"

TRAIN_FLAG=""
if [ "$TRAIN_MODE" = "1" ] && [ "$MODULATOR_KIND" = "language" ]; then
  TRAIN_FLAG="--train_mode"
fi

echo "🚀 Running $MODULATOR_KIND modulator remotely..."
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" -- -t "
  set -e
  mkdir -p $REMOTE_DIR/output
  gcloud auth configure-docker $REGION-docker.pkg.dev -q
  gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin https://$REGION-docker.pkg.dev
  docker run --rm --gpus all \
    --network host \
    --env-file $REMOTE_DIR/.env \
    -e HF_TOKEN=\"$HF_TOKEN\" \
    -v $REMOTE_DIR/output:/app/output \
    $IMAGE_REF \
    python3 $TRAINER_SCRIPT \
    --base_url $BASE_URL \
    --api_key $AOC_MODULATION_TOKEN \
    --model_template_id $MODEL_TEMPLATE_ID \
    --outdir /app/output $TRAIN_FLAG
  ls -lhR $REMOTE_DIR/output || true
  sync
"

echo "📥 Copying results locally..."
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command "ls -lhR $REMOTE_DIR/output" || echo "⚠️ No remote artifacts found"
gcloud compute scp --compress --recurse "$INSTANCE_NAME:$REMOTE_DIR/output/*" "$RESULTS_DIR/" --zone="$ZONE" || true
if [ -z "$(ls -A "$RESULTS_DIR" 2>/dev/null)" ]; then
  echo "⚠️ No artifacts copied on first attempt, retrying..."
  gcloud compute scp --recurse "$INSTANCE_NAME:$REMOTE_DIR/output" "$RESULTS_DIR/" --zone="$ZONE" || true
fi

echo "✅ Results saved in $RESULTS_DIR"
