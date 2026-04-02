#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODULATOR_DIR="$SCRIPT_DIR"
export MODULATOR_KIND="tts"
export TRAINER_SCRIPT="train_tts.py"
export DEFAULT_OUTDIR="./artifacts"

exec "$SCRIPT_DIR/../modulate_gcp_common.sh" "$@"
