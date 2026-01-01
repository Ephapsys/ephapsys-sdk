#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
META_FILE="$SCRIPT_DIR/.last_gcp_instance"

if [ ! -f "$META_FILE" ]; then
  echo "❌ No previous GCP run found. Launch an instance with run_gcp.sh first."
  exit 1
fi

source "$META_FILE"

if [ -z "${INSTANCE_NAME:-}" ] || [ -z "${PROJECT_ID:-}" ] || [ -z "${ZONE:-}" ]; then
  echo "❌ Metadata file is incomplete. Please rerun run_gcp.sh."
  exit 1
fi

echo "🪄 Detaching local client from tmux session on $INSTANCE_NAME..."
exec gcloud compute ssh "$INSTANCE_NAME" \
  --project "$PROJECT_ID" \
  --zone "$ZONE" \
  -- -t 'if tmux has-session -t helloworld 2>/dev/null; then tmux detach-client -s helloworld && echo "✅ Detached. Session keeps running."; else echo "⚠️ No helloworld tmux session running."; fi'
