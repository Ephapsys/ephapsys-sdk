#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GCP_ENV_FILE="${GCP_ENV_FILE:-${HELLOWORLD_GCP_ENV_FILE:-$SCRIPT_DIR/.env.gcp}}"
RUNTIME_ENV_FILE="${GCP_RUNTIME_ENV_FILE:-$SCRIPT_DIR/.env}"

if [[ ! -f "$GCP_ENV_FILE" ]]; then
  echo "[ERROR] Missing $GCP_ENV_FILE. Copy .env.gcp.example to .env.gcp first." >&2
  exit 1
fi

set -a
source "$GCP_ENV_FILE"
set +a

for var in PROJECT_ID ZONE MACHINE_TYPE DISK_SIZE IMAGE_FAMILY IMAGE_PROJECT INSTANCE_PREFIX; do
  if [[ -z "${!var:-}" ]]; then
    echo "[ERROR] $var must be set in $GCP_ENV_FILE" >&2
    exit 1
  fi
done

if [[ ! -f "$RUNTIME_ENV_FILE" ]]; then
  echo "[ERROR] Missing runtime env file: $RUNTIME_ENV_FILE" >&2
  exit 1
fi

set -a
source "$RUNTIME_ENV_FILE"
set +a

for var in AOC_BASE_URL AOC_ORG_ID AOC_PROVISIONING_TOKEN AGENT_TEMPLATE_ID; do
  if [[ -z "${!var:-}" ]]; then
    echo "[ERROR] $var must be set in $RUNTIME_ENV_FILE" >&2
    exit 1
  fi
done

if ! command -v gcloud >/dev/null 2>&1; then
  echo "[ERROR] gcloud CLI not found." >&2
  exit 1
fi

ACCOUNT="$(gcloud config get-value account 2>/dev/null || true)"
if [[ -z "$ACCOUNT" ]]; then
  echo "[ERROR] No active gcloud account. Run 'gcloud auth login' first." >&2
  exit 1
fi

PROJECT_CHECK="$(gcloud projects describe "$PROJECT_ID" --format='value(projectId)' 2>/dev/null || true)"
if [[ "$PROJECT_CHECK" != "$PROJECT_ID" ]]; then
  echo "[ERROR] gcloud cannot access project '$PROJECT_ID'." >&2
  exit 1
fi

echo "[CHECK] GCP preflight OK for project=$PROJECT_ID zone=$ZONE account=$ACCOUNT"
