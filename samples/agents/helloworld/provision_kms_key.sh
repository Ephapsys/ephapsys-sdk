#!/usr/bin/env bash
# ============================================================
# provision_kms_key.sh
#
# Utility script to create (or reuse) a Google Cloud KMS asymmetric‑signing key
# for the HelloWorld sample and grant a service account permission to use it.
#
# Example:
#   ./provision_kms_key.sh \
#       --project ephapsys-development \
#       --location us-central1 \
#       --keyring agents \
#       --key helloworld \
#       --service-account hello-agent@ephapsys-development.iam.gserviceaccount.com \
#       --hsm
#
# After running, set HSM_KMS_KEY in your .env.* to the printed resource path.
# ============================================================

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-}"
LOCATION="${LOCATION:-us-central1}"
KEY_RING="${KEY_RING:-agents}"
KEY_NAME="${KEY_NAME:-helloworld}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-}"
PROTECTION_LEVEL="software"
ALGORITHM="ec-sign-p256-sha256"

usage() {
  cat <<EOF
Usage: $(basename "$0") --project <id> --service-account <email> [options]

Required flags:
  --project            GCP project ID that hosts the KMS resources.
  --service-account    Service account email to grant signer access (e.g. hello-agent@project.iam.gserviceaccount.com).

Optional flags:
  --location           KMS location (default: ${LOCATION})
  --keyring            Key ring name (default: ${KEY_RING})
  --key                Crypto key name (default: ${KEY_NAME})
  --algorithm          KMS algorithm (default: ${ALGORITHM})
  --hsm                Use protection-level=HSM (default is software)
  --help               Show this message

Environment variables PROJECT_ID, LOCATION, KEY_RING, KEY_NAME, SERVICE_ACCOUNT
can also be used instead of flags.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECT_ID="$2"; shift 2 ;;
    --location) LOCATION="$2"; shift 2 ;;
    --keyring) KEY_RING="$2"; shift 2 ;;
    --key) KEY_NAME="$2"; shift 2 ;;
    --service-account) SERVICE_ACCOUNT="$2"; shift 2 ;;
    --algorithm) ALGORITHM="$2"; shift 2 ;;
    --hsm) PROTECTION_LEVEL="hsm"; shift ;;
    --help|-h) usage; exit 0 ;;
    *)
      echo "Unknown flag: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$PROJECT_ID" || -z "$SERVICE_ACCOUNT" ]]; then
  echo "❌ --project and --service-account are required."
  usage
  exit 1
fi

if ! command -v gcloud >/dev/null 2>&1; then
  echo "❌ gcloud CLI not found. Install and authenticate before running this script."
  exit 1
fi

echo "🔐 Provisioning KMS key"
echo "  Project   : $PROJECT_ID"
echo "  Location  : $LOCATION"
echo "  Key ring  : $KEY_RING"
echo "  Key name  : $KEY_NAME"
echo "  Algorithm : $ALGORITHM"
echo "  Protection: $PROTECTION_LEVEL"
echo "  Service SA: $SERVICE_ACCOUNT"

FULL_KEY_RING="projects/${PROJECT_ID}/locations/${LOCATION}/keyRings/${KEY_RING}"
FULL_KEY="${FULL_KEY_RING}/cryptoKeys/${KEY_NAME}"

# Create key ring if needed
if ! gcloud kms keyrings describe "$KEY_RING" --project "$PROJECT_ID" --location "$LOCATION" >/dev/null 2>&1; then
  echo "📦 Creating key ring $KEY_RING..."
  gcloud kms keyrings create "$KEY_RING" \
    --project "$PROJECT_ID" \
    --location "$LOCATION"
else
  echo "ℹ️  Key ring $KEY_RING already exists."
fi

# Create key if needed
if ! gcloud kms keys describe "$KEY_NAME" --project "$PROJECT_ID" --location "$LOCATION" --keyring "$KEY_RING" >/dev/null 2>&1; then
  echo "🗝️  Creating key $KEY_NAME..."
  gcloud kms keys create "$KEY_NAME" \
    --project "$PROJECT_ID" \
    --location "$LOCATION" \
    --keyring "$KEY_RING" \
    --purpose asymmetric-signing \
    --default-algorithm "$ALGORITHM" \
    --protection-level "$PROTECTION_LEVEL"
else
  echo "ℹ️  Key $KEY_NAME already exists."
fi

echo "👤 Granting roles/cloudkms.signerVerifier to $SERVICE_ACCOUNT..."
gcloud kms keys add-iam-policy-binding "$KEY_NAME" \
  --project "$PROJECT_ID" \
  --location "$LOCATION" \
  --keyring "$KEY_RING" \
  --member "serviceAccount:${SERVICE_ACCOUNT}" \
  --role "roles/cloudkms.signerVerifier" \
  >/dev/null

PRIMARY_VERSION="$(gcloud kms keys describe "$KEY_NAME" --project "$PROJECT_ID" --location "$LOCATION" --keyring "$KEY_RING" --format='value(primary.name)' || true)"

echo
echo "✅ Done. Set the following in your .env.stag / .env.prod:"
echo
if [[ -n "$PRIMARY_VERSION" ]]; then
  echo "HSM_KMS_KEY=${PRIMARY_VERSION}"
else
  echo "HSM_KMS_KEY=${FULL_KEY}"
  echo "# (key has no primary version yet; personalize will use whichever version becomes primary.)"
fi
echo "HSM_KMS_ENDPOINT=privatekms.googleapis.com   # optional, only for private access"
echo "HSM_KMS_CREDENTIALS=/path/to/service-account.json  # if you plan to upload a dedicated key file"
echo
echo "Reminder: grant the service account access to your VM instance profile or upload its JSON and point HSM_KMS_CREDENTIALS at it."
