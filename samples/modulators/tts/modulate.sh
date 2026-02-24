#!/usr/bin/env bash
# ============================================================
# Bash wrapper to run Ephaptic TTS Trainer (SpeechT5)
#
# Usage flow:
# - Minimal CLI args: --base_url, --api_key, --model_template_id, --outdir
# - All training hyperparameters, dataset config, and model_id are fetched dynamically
#   from the backend template created in the UI.
# - The trainer does not accept manual tuning flags for variant, epsilon, dataset split, etc.;
#   these must be specified in the Modulation config of the template.
#
# Before starting a job in the UI:
#
# 1. Create a Model Template (via the Create Model page):
#    - Source: External repository
#    - Provider: Hugging Face
#    - Repository ID: microsoft/speecht5_tts
#    - Model Kind: TTS
#    - Register immediately (so a provenance certificate is issued)
#
# 2. Go to the Modulator page for this template:
#    - Variant: additive or multiplicative
#    - Hyperparameters: epsilon (ε), lambda0 (λ₀), phi (activation), ecm_init
#    - MaxSteps: number of samples/steps to evaluate
#    - Dataset: name (e.g., librispeech_asr), config (e.g., clean), split (e.g., validation[:1%])
#    - KPI Targets: enable at least one KPI relevant to TTS (e.g., WER, MOS)
# ============================================================

set -euo pipefail

# ---------------- LOAD .env ----------------
if [ -f ".env" ]; then
  echo "[INFO] Loading environment from .env"
  export $(grep -v '^#' .env | xargs)
fi

# ---------------- CONFIG ----------------
BASE_URL=${AOC_BASE_URL:-${BASE_URL:-"http://localhost:7001"}}       # Backend AOC API
AOC_ORG_ID=${AOC_ORG_ID:-""}
AOC_MODULATION_TOKEN=${AOC_MODULATION_TOKEN:-""}                              # API key (must be set in .env or env var)
MODEL_TEMPLATE_ID=${MODEL_TEMPLATE_ID:-""}          # Model Template ID (must be set in .env or env var)
OUTDIR=${OUTDIR:-"./artifacts"}                 # Output folder

if [ -z "$AOC_ORG_ID" ] || [ -z "$AOC_MODULATION_TOKEN" ] || [ -z "$MODEL_TEMPLATE_ID" ]; then
  echo "[ERROR] AOC_ORG_ID, AOC_MODULATION_TOKEN and MODEL_TEMPLATE_ID must be set in .env or env vars."
  exit 1
fi

# ---------------- RUN ----------------
echo "[INFO] Starting EphapticTTS Trainer..."
echo "  BASE_URL:         $BASE_URL"
echo "  AOC_ORG_ID:        $AOC_ORG_ID"
echo "  AOC_MODULATION_TOKEN:          ${AOC_MODULATION_TOKEN:0:8}********"   # mask key for safety
echo "  MODEL_TEMPLATE_ID: $MODEL_TEMPLATE_ID"
echo "  OUTDIR:           $OUTDIR"

python3 train_tts.py \
  --base_url "$BASE_URL" \
  --api_key "$AOC_MODULATION_TOKEN" \
  --model_template_id "$MODEL_TEMPLATE_ID" \
  --outdir "$OUTDIR"

echo "[INFO] Training complete. Artifacts stored in: $OUTDIR"
