#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}" # smoke | integration

SAMPLES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

run_smoke() {
  echo "[CI] Running sample smoke checks..."

  # Dummy creds for script-level validation.
  export AOC_BASE_URL="${AOC_BASE_URL:-https://api.ephapsys.com}"
  export AOC_ORG_ID="${AOC_ORG_ID:-org_ci_dummy}"
  export AOC_PROVISIONING_TOKEN="${AOC_PROVISIONING_TOKEN:-prov_ci_dummy}"
  export AOC_MODULATION_TOKEN="${AOC_MODULATION_TOKEN:-mod_ci_dummy}"
  export AGENT_TEMPLATE_ID="${AGENT_TEMPLATE_ID:-agent_temp_ci_dummy}"
  export MODEL_TEMPLATE_ID="${MODEL_TEMPLATE_ID:-model_temp_ci_dummy}"
  export SAMPLE_CI_SMOKE=1

  (cd "$SAMPLES_DIR/agents/helloworld" && bash ./run_local.sh smoke)
  (cd "$SAMPLES_DIR/agents/robot" && bash ./run_local.sh smoke)
  (cd "$SAMPLES_DIR/modulators/language" && bash ./modulate_local.sh smoke)

  # Wider syntax coverage for other modulator samples.
  python3 -m py_compile "$SAMPLES_DIR/modulators/audio/train_audio.py"
  python3 -m py_compile "$SAMPLES_DIR/modulators/vision/train_vision.py"
  python3 -m py_compile "$SAMPLES_DIR/modulators/embedding/train_embedding.py"
  python3 -m py_compile "$SAMPLES_DIR/modulators/stt/train_stt.py"
  python3 -m py_compile "$SAMPLES_DIR/modulators/tts/train_tts.py"
  python3 -m py_compile "$SAMPLES_DIR/modulators/rl/train_rl.py"

  echo "[CI] Sample smoke checks passed."
}

run_integration() {
  echo "[CI] Running sample integration checks..."

  # Required for real backend hit.
  : "${AOC_BASE_URL:?AOC_BASE_URL is required for integration}"
  : "${AOC_ORG_ID:?AOC_ORG_ID is required for integration}"
  : "${AOC_PROVISIONING_TOKEN:?AOC_PROVISIONING_TOKEN is required for integration}"
  : "${AGENT_TEMPLATE_ID:?AGENT_TEMPLATE_ID is required for integration}"

  # Keep this bounded for CI; no interactive loop.
  export HELLOWORLD_CI_ONESHOT=1
  export HELLOWORLD_CI_PROMPT="${HELLOWORLD_CI_PROMPT:-health check}"
  export PERSONALIZE_ANCHOR="${PERSONALIZE_ANCHOR:-none}"
  (cd "$SAMPLES_DIR/agents/helloworld" && timeout 240 bash ./run_local.sh oneshot)

  # Robot integration mock path (hardware-optional) when robot template is provided.
  if [ -n "${ROBOT_AGENT_TEMPLATE_ID:-}" ]; then
    export AGENT_TEMPLATE_ID="$ROBOT_AGENT_TEMPLATE_ID"
    export PERSONALIZE_ANCHOR="${PERSONALIZE_ANCHOR:-none}"
    (cd "$SAMPLES_DIR/agents/robot" && timeout 300 bash ./run_ci_mock.sh)
  else
    echo "[CI] ROBOT_AGENT_TEMPLATE_ID not set; running robot smoke fallback."
    export SAMPLE_CI_SMOKE=1
    (cd "$SAMPLES_DIR/agents/robot" && bash ./run_local.sh smoke)
  fi

  # Modulators remain smoke in CI until test templates/datasets are provisioned.
  export SAMPLE_CI_SMOKE=1
  (cd "$SAMPLES_DIR/modulators/language" && bash ./modulate_local.sh smoke)

  echo "[CI] Sample integration checks passed."
}

case "$MODE" in
  smoke)
    run_smoke
    ;;
  integration)
    run_integration
    ;;
  *)
    echo "Usage: $0 [smoke|integration]"
    exit 2
    ;;
esac
