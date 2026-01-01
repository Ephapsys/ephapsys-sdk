#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ID="${PROJECT_ID:-ephapsys-development}"
ACCOUNT="${1:-$(gcloud config get-value account 2>/dev/null || true)}"
ROLE="roles/compute.instanceAdmin.v1"
if [ -z "$ACCOUNT" ]; then
  echo "❌ No gcloud account found. Pass the email as the first argument."
  exit 1
fi
echo "🔎 Checking if $ACCOUNT has $ROLE on $PROJECT_ID..."
gcloud projects get-iam-policy "$PROJECT_ID" \
  --flatten="bindings[].members" \
  --format="table(bindings.role, bindings.members)" \
  --filter="bindings.members:user:$ACCOUNT AND bindings.role:$ROLE"
