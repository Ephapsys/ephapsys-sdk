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

if [ -n "${ANCHOR:-}" ]; then
  echo "🔌 Connecting to $INSTANCE_NAME ($ZONE / $PROJECT_ID, anchor=$ANCHOR)..."
else
  echo "🔌 Connecting to $INSTANCE_NAME ($ZONE / $PROJECT_ID)..."
fi
exec gcloud compute ssh "$INSTANCE_NAME" \
  --project "$PROJECT_ID" \
  --zone "$ZONE" \
  -- -t '
cd ~/helloworld
if pgrep -f helloworld_agent.py >/dev/null; then
  echo "[VM] helloworld_agent.py running – dropping into interactive session."
  set -a
  source .env
  set +a
  source .venv/bin/activate
  python helloworld_agent.py
else
  echo "[VM] No chatbot process running; tailing logs instead."
  tail -f helloworld.log
fi
'
