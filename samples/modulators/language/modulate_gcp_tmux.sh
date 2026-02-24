#!/bin/bash

# =========================================================
# 🧠 Auto-tmux + Logging Wrapper for modulate_gcp.sh (local)
# =========================================================
if command -v tmux &>/dev/null; then
  if [ -z "$TMUX" ]; then
    TS=$(date +"%Y%m%d_%H%M%S")
    SESSION_NAME="modulate_gcp_${TS}"
    LOG_DIR="$(pwd)/logs"
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/${SESSION_NAME}.log"

    echo "🧠 Starting local tmux session: $SESSION_NAME"
    echo "📜 Logs will be saved to: $LOG_FILE"

    exec tmux new -s "$SESSION_NAME" "bash -c '$0 \"$@\" 2>&1 | tee -a \"$LOG_FILE\"'"
  fi
else
  echo "⚠️ tmux not installed. Running directly (no local session persistence)."
  mkdir -p logs
  LOG_FILE="logs/modulate_gcp_$(date +%Y%m%d_%H%M%S).log"
  echo "📜 Logging to: $LOG_FILE"
  exec bash -c "$0 \"$@\" 2>&1 | tee -a \"$LOG_FILE\""
fi
# =========================================================

set -euo pipefail

# === Load infra config if available ===
if [ -f "gcp.env" ]; then
  echo "📂 Loading GCP config from gcp.env"
  source gcp.env
fi

# === Detect --resume mode ===
if [[ "${1:-}" == "--resume" ]]; then
  if [[ -z "${2:-}" ]]; then
    echo "❌ Usage: ./modulate_gcp.sh --resume <instance_name>"
    exit 1
  fi

  INSTANCE_NAME="$2"
  echo "🔁 Resuming sync for existing VM: $INSTANCE_NAME"

  REMOTE_DIR="~/epgpt2"
  RESULTS_DIR="results/resume_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$RESULTS_DIR"

  echo "📡 Restarting background log + artifact sync..."
  (
    while true; do
      if ! gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" >/dev/null 2>&1; then
        echo "🛑 VM deleted or stopped — ending log sync."
        break
      fi
      gcloud compute scp "$INSTANCE_NAME:$REMOTE_DIR/artifacts/modulate_*.log" "logs/" --zone="$ZONE" --quiet 2>/dev/null || true
      sleep 120
    done
  ) &

  (
    while true; do
      if ! gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" >/dev/null 2>&1; then
        echo "🛑 VM deleted or stopped — ending artifact sync."
        break
      fi
      gcloud compute scp --compress --recurse "$INSTANCE_NAME:$REMOTE_DIR/artifacts/*" "$RESULTS_DIR/" --zone="$ZONE" --quiet 2>/dev/null || true
      sleep 300
    done
  ) &

  echo
  echo "🧩 Active tmux sessions on $INSTANCE_NAME:"
  gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" -- tmux ls || echo "⚠️ No tmux sessions found."

  echo
  echo "🪄 You can attach to a running session with:"
  echo "    gcloud compute ssh $INSTANCE_NAME --zone=$ZONE -- tmux attach -t <session_name>"
  echo "📜 Logs will continue syncing every 2 min → ./logs/"
  echo "📦 Artifacts will continue syncing every 5 min → ./results/"
  exit 0
fi

# === Normal (non-resume) mode ===
INSTANCE_NAME="ec-modulate-$(date +%s)"
DISK_SIZE="${DISK_SIZE:-200GB}"
REMOTE_DIR="~/epgpt2"
GPU_CHOICE="${1:-t4}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-modulate}"

RUN_TS=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="results/${EXPERIMENT_TAG}_${RUN_TS}"
mkdir -p "$RESULTS_DIR"

AUTO_DELETE="${AUTO_DELETE:-false}"   # safer default

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
  echo "❌ Missing .env (must contain AOC_ORG_ID, AOC_MODULATION_TOKEN, MODEL_TEMPLATE_ID, BASE_URL)"
  exit 1
fi

export $(grep -v '^#' .env | xargs)

if grep -q '^HF_TOKEN=' .env; then
  echo "🔑 Found HF_TOKEN in .env"
  export HF_TOKEN=$(grep '^HF_TOKEN=' .env | cut -d '=' -f2)
else
  echo "⚠️ No HF_TOKEN found in .env; proceeding without authentication."
fi

if [ -z "${BASE_URL:-}" ] || [ -z "${AOC_ORG_ID:-}" ] || [ -z "${AOC_MODULATION_TOKEN:-}" ] || [ -z "${MODEL_TEMPLATE_ID:-}" ]; then
  echo "❌ Missing required runtime vars (BASE_URL, AOC_ORG_ID, AOC_MODULATION_TOKEN, MODEL_TEMPLATE_ID)"
  exit 1
fi

MASKED="${AOC_MODULATION_TOKEN:0:8}********"
echo "🔐 Env loaded: BASE_URL=$BASE_URL, AOC_ORG_ID=$AOC_ORG_ID, AOC_MODULATION_TOKEN=${MASKED}, MODEL_TEMPLATE_ID=$MODEL_TEMPLATE_ID"

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

cleanup() {
  if [ "$AUTO_DELETE" = "true" ]; then
    echo "⚠️ Cleaning up VM: $INSTANCE_NAME"
    gcloud compute instances delete "$INSTANCE_NAME" --zone="$ZONE" --quiet || true
  else
    echo "⚙️ AUTO_DELETE disabled — VM will persist after run."
  fi
}
trap cleanup EXIT

IMAGE_REF="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE_NAME:$TAG"

# === 3. Copy .env to VM ===
echo "📂 Copying .env to VM..."
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command "mkdir -p $REMOTE_DIR"
gcloud compute scp ".env" "$INSTANCE_NAME:$REMOTE_DIR/.env" --zone="$ZONE"

# === 4. Run container on VM inside tmux ===
echo "🚀 Running container remotely inside tmux..."
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" -- -t "
  set -e
  mkdir -p $REMOTE_DIR/artifacts
  LOG_FILE=\$REMOTE_DIR/artifacts/modulate_\$(date +%Y%m%d_%H%M%S).log

  gcloud auth configure-docker $REGION-docker.pkg.dev -q
  gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin https://$REGION-docker.pkg.dev

  TMUX_SESSION=\"modulate_vm_\$(date +%H%M%S)\"
  echo \"🧠 Starting tmux session on VM: \$TMUX_SESSION\"
  echo \"📜 Logs: \$LOG_FILE\"

  tmux new -d -s \$TMUX_SESSION \"
    docker run --rm --gpus all \
      --network host \
      --env-file $REMOTE_DIR/.env \
      -e HF_TOKEN='$HF_TOKEN' \
      -v $REMOTE_DIR/artifacts:/app/artifacts \
      $IMAGE_REF \
      --base_url $BASE_URL \
      --api_key $AOC_MODULATION_TOKEN \
      --model_template_id $MODEL_TEMPLATE_ID \
      --outdir artifacts 2>&1 | tee -a \$LOG_FILE
      --train_mode
    echo '✅ Job complete inside tmux session \$TMUX_SESSION' | tee -a \$LOG_FILE
    if [ \"$AUTO_DELETE\" = \"true\" ]; then
      echo '🧹 Cleaning up VM in 2 minutes...' | tee -a \$LOG_FILE
      sleep 120
      gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --quiet
    fi
  \"

  echo
  echo \"🧩 To monitor or reattach later:\"
  echo \"  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE -- tmux attach -t \$TMUX_SESSION\"
  echo \"  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE -- tail -f \$LOG_FILE\"
"

# === 5. Background log sync with auto-stop ===
echo "📡 Starting background log sync..."
(
  while true; do
    if ! gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" >/dev/null 2>&1; then
      echo "🛑 VM deleted or stopped — ending log sync."
      break
    fi
    gcloud compute scp "$INSTANCE_NAME:$REMOTE_DIR/artifacts/modulate_*.log" "logs/" --zone="$ZONE" --quiet 2>/dev/null || true
    sleep 120
  done
) &

# === 6. Background artifact sync with auto-stop ===
echo "📥 Starting background artifact sync..."
(
  while true; do
    if ! gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" >/dev/null 2>&1; then
      echo "🛑 VM deleted or stopped — ending artifact sync."
      break
    fi
    gcloud compute scp --compress --recurse "$INSTANCE_NAME:$REMOTE_DIR/artifacts/*" "$RESULTS_DIR/" --zone="$ZONE" --quiet 2>/dev/null || true
    sleep 300
  done
) &

echo
echo "✅ Modulation job launched on GCP VM: $INSTANCE_NAME"
echo "📍 You can safely close your terminal or shut down your Mac."
echo
echo "🧩 To check progress later:"
echo "    ./modulate_gcp.sh --resume $INSTANCE_NAME"
echo
echo "📜 Logs will sync locally every 2 min → ./logs/"
echo "📦 Artifacts will sync locally every 5 min → ./results/"
