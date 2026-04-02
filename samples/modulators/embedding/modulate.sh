#!/usr/bin/env bash
# ============================================================
# Bash wrapper to run Ephaptic Embedding Trainer (Gemma-300M)
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../modulate_local_common.sh"

modulator_load_env_file ".env"

BASE_URL=${AOC_BASE_URL:-${BASE_URL:-"http://localhost:7001"}}
AOC_ORG_ID=${AOC_ORG_ID:-""}
AOC_MODULATION_TOKEN=${AOC_MODULATION_TOKEN:-""}
MODEL_TEMPLATE_ID=${MODEL_TEMPLATE_ID:-""}
OUTDIR=${OUTDIR:-"./artifacts_embedding"}

if [ -z "$AOC_ORG_ID" ] || [ -z "$AOC_MODULATION_TOKEN" ] || [ -z "$MODEL_TEMPLATE_ID" ]; then
  echo "[ERROR] AOC_ORG_ID, AOC_MODULATION_TOKEN and MODEL_TEMPLATE_ID must be set."
  exit 1
fi

modulator_prepare_env

echo "[INFO] Starting EphapticEmbedding Trainer..."
echo "  BASE_URL:          $BASE_URL"
echo "  AOC_ORG_ID:        $AOC_ORG_ID"
echo "  AOC_MODULATION_TOKEN:           ${AOC_MODULATION_TOKEN:0:8}********"
echo "  MODEL_TEMPLATE_ID: $MODEL_TEMPLATE_ID"
echo "  OUTDIR:            $OUTDIR"

python3 train_embedding.py \
  --base_url "$BASE_URL" \
  --api_key "$AOC_MODULATION_TOKEN" \
  --model_template_id "$MODEL_TEMPLATE_ID" \
  --outdir "$OUTDIR"

echo "[INFO] Training complete. Artifacts stored in: $OUTDIR"
