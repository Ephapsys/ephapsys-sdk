#!/usr/bin/env bash
# ============================================================
# Bash wrapper to run Ephaptic Embedding Trainer (Gemma-300M)
# ============================================================

set -euo pipefail

if [ -f ".env" ]; then
  echo "[INFO] Loading environment from .env"
  export $(grep -v '^#' .env | xargs)
fi

BASE_URL=${BASE_URL:-"http://localhost:7001"}
API_KEY=${API_KEY:-""}
MODEL_TEMPLATE_ID=${MODEL_TEMPLATE_ID:-""}
OUTDIR=${OUTDIR:-"./artifacts_embedding"}

if [ -z "$API_KEY" ] || [ -z "$MODEL_TEMPLATE_ID" ]; then
  echo "[ERROR] API_KEY and MODEL_TEMPLATE_ID must be set."
  exit 1
fi

echo "[INFO] Starting EphapticEmbedding Trainer..."
echo "  BASE_URL:          $BASE_URL"
echo "  API_KEY:           ${API_KEY:0:8}********"
echo "  MODEL_TEMPLATE_ID: $MODEL_TEMPLATE_ID"
echo "  OUTDIR:            $OUTDIR"

python3 train_embedding.py \
  --base_url "$BASE_URL" \
  --api_key "$API_KEY" \
  --model_template_id "$MODEL_TEMPLATE_ID" \
  --outdir "$OUTDIR"

echo "[INFO] Training complete. Artifacts stored in: $OUTDIR"
