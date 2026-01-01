#!/usr/bin/env bash
set -euo pipefail

# === Usage ===
# ./run_test_eval.sh [MODEL_NAME] [MAX_STEPS]
# Example: ./run_test_eval.sh gpt2 20
#          ./run_test_eval.sh google/flan-t5-small 10

MODEL="${1:-gpt2}"
STEPS="${2:-10}"

# Optional: Hugging Face token (recommended for private or gated models)
# Export it once in your shell or add it here
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "⚠️  HF_TOKEN not set. You can export it via:"
  echo "   export HF_TOKEN='hf_xxxxxxxxxxxxxxxxx'"
  echo "   or edit this script to include your token."
else
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
  echo "🔑 Using Hugging Face token: ${HF_TOKEN:0:10}..."
fi

# Run evaluation
echo "🚀 Running test_evaluation.py on model=${MODEL}, steps=${STEPS}"
pip install torch transformers datasets evaluate python-docx matplotlib rouge_score
python3 test_evaluation.py --model "${MODEL}" --max_steps "${STEPS}"
