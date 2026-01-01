#!/usr/bin/env bash
# ============================================================
# Bash wrapper to run Ephaptic Language Trainer (Flan-T5 / GPT-2)
# ============================================================

set -euo pipefail

# --- Load environment from .env if present ---
if [ -f ".env" ]; then
  echo "[INFO] Loading environment from .env"
  # shellcheck disable=SC2046
  export $(grep -v '^#' .env | xargs)
fi

# --- Default values ---
BASE_URL=${BASE_URL:-"http://localhost:7001"}
API_KEY=${API_KEY:-""}
MODEL_TEMPLATE_ID=${MODEL_TEMPLATE_ID:-""}
OUTDIR=${OUTDIR:-"./artifacts"}
TRAIN_MODE=${TRAIN_MODE:-"1"}   # ✅ Default = 1 (training enabled)

# --- Sanity checks ---
if [ -z "$API_KEY" ] || [ -z "$MODEL_TEMPLATE_ID" ]; then
  echo "[ERROR] API_KEY and MODEL_TEMPLATE_ID must be set."
  exit 1
fi

echo "[INFO] Starting Ephaptic Language Trainer..."
echo "  BASE_URL:          $BASE_URL"
echo "  API_KEY:           ${API_KEY:0:8}********"
echo "  MODEL_TEMPLATE_ID: $MODEL_TEMPLATE_ID"
echo "  OUTDIR:            $OUTDIR"
echo "  TRAIN_MODE:        $TRAIN_MODE (1=train enabled, 0=evaluation only)"

# --- Build Python command dynamically ---
CMD=(
  python3 train_language.py
  --base_url "$BASE_URL"
  --api_key "$API_KEY"
  --model_template_id "$MODEL_TEMPLATE_ID"
  --outdir "$OUTDIR"
)

if [ "$TRAIN_MODE" = "1" ]; then
  echo "[INFO] Training flag active (gradient updates ON)"
  CMD+=(--train)   # ✅ new unified flag (backward-compatible internally)
else
  echo "[INFO] Evaluation-only mode (baseline + ephaptic comparison)"
fi

# --- Execute trainer ---
"${CMD[@]}"

echo "[INFO] Trainer finished successfully."
echo "[INFO] Artifacts stored in: $OUTDIR"
