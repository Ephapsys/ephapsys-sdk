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
BASE_URL=${AOC_BASE_URL:-${BASE_URL:-"http://localhost:7001"}}
AOC_ORG_ID=${AOC_ORG_ID:-""}
AOC_MODULATION_TOKEN=${AOC_MODULATION_TOKEN:-""}
MODEL_TEMPLATE_ID=${MODEL_TEMPLATE_ID:-""}
OUTDIR=${OUTDIR:-"./artifacts"}
TRAIN_MODE=${TRAIN_MODE:-"1"}   # ✅ Default = 1 (training enabled)

# --- Sanity checks ---
if [ -z "$AOC_ORG_ID" ] || [ -z "$AOC_MODULATION_TOKEN" ] || [ -z "$MODEL_TEMPLATE_ID" ]; then
  echo "[ERROR] AOC_ORG_ID, AOC_MODULATION_TOKEN and MODEL_TEMPLATE_ID must be set."
  exit 1
fi

echo "[INFO] Starting Ephaptic Language Trainer..."
echo "  BASE_URL:          $BASE_URL"
echo "  AOC_ORG_ID:        $AOC_ORG_ID"
echo "  AOC_MODULATION_TOKEN:           ${AOC_MODULATION_TOKEN:0:8}********"
echo "  MODEL_TEMPLATE_ID: $MODEL_TEMPLATE_ID"
echo "  OUTDIR:            $OUTDIR"
echo "  TRAIN_MODE:        $TRAIN_MODE (1=train enabled, 0=evaluation only)"

# --- Build Python command dynamically ---
CMD=(
  python3 train_language.py
  --base_url "$BASE_URL"
  --api_key "$AOC_MODULATION_TOKEN"
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
