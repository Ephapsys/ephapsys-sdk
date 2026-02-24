#!/usr/bin/env bash
set -euo pipefail

# =====================================================================
# push.sh
# - Registers baseline model templates (via CLI login)
# - Optionally modulates them (default) OR skips modulation with --idempotent
# - Creates an agent template bundling the model templates
# =====================================================================

USAGE="Usage: $0 [--idempotent] [--label \"Robot Agent Template\"]"
POLL_INTERVAL_SEC=5
POLL_TIMEOUT_SEC=7200

IDEMPOTENT=0
LABEL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --idempotent) IDEMPOTENT=1; shift ;;
    --label) LABEL="$2"; shift 2 ;;
    -h|--help) echo "$USAGE"; exit 0 ;;
    *) echo "Unknown arg: $1"; echo "$USAGE"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------
# Env + defaults
# ---------------------------------------------------------------------
[[ -f .env ]] && set -o allexport && source .env && set +o allexport

AOC_API="${AOC_API:-${BASE_URL:-http://localhost:7001}}"
CLI_API="${AOC_API}/cli"
TOKEN_FILE=".cli_token"
SESSION_FILE="${HOME}/.ephapsys_state/session.json"
API_TOKEN="${API_TOKEN:-${AOC_MODULATION_TOKEN:-}}"
HF_TOKEN="${HF_TOKEN:-""}"

if [[ -z "$API_TOKEN" ]]; then
  echo "[FATAL] API_TOKEN is required to create agent templates. Set API_TOKEN or AOC_MODULATION_TOKEN in .env or environment."
  exit 1
fi

# ---------------------------------------------------------------------
# CLI login (reuse ephapsys login session or cached token if valid)
# ---------------------------------------------------------------------
TOKEN=""
TOKEN_SOURCE=""

# Prefer token from ephapsys CLI session (~/.ephapsys_state/session.json)
if [[ -f "$SESSION_FILE" ]]; then
  TOKEN=$(jq -r '.token // empty' "$SESSION_FILE" 2>/dev/null || true)
  if [[ -n "$TOKEN" ]]; then
    VALID=$(curl -s -o /dev/null -w "%{http_code}" \
      -H "Authorization: Bearer $TOKEN" \
      "$CLI_API/models/list")
    if [[ "$VALID" == "200" ]]; then
      TOKEN_SOURCE="session"
      echo "[INFO] Using CLI token from ephapsys login ($SESSION_FILE)"
    else
      echo "[WARN] Session token invalid/expired. Will fall back."
      TOKEN=""
    fi
  fi
fi

# Fall back to local .cli_token cache
if [[ -z "$TOKEN" && -f "$TOKEN_FILE" ]]; then
  TOKEN=$(cat "$TOKEN_FILE")
  VALID=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: Bearer $TOKEN" \
    "$CLI_API/models/list")
  if [[ "$VALID" == "200" ]]; then
    TOKEN_SOURCE="cache"
    echo "[INFO] Using cached CLI token from $TOKEN_FILE"
  else
    echo "[WARN] Cached token invalid/expired. Re-login required."
    TOKEN=""
    rm -f "$TOKEN_FILE"
  fi
fi

if [[ -z "$TOKEN" ]]; then
  read -p "Enter CLI username (email): " CLI_USER
  read -s -p "Enter CLI password: " CLI_PASS
  echo
  echo "[INFO] Logging in as $CLI_USER ..."
  LOGIN_RESP=$(curl -s -X POST "$CLI_API/login" \
    -H "Content-Type: application/json" \
    -d "{\"username\": \"${CLI_USER}\", \"password\": \"${CLI_PASS}\"}")
  TOKEN=$(echo "$LOGIN_RESP" | jq -r '.token')
  if [[ -z "$TOKEN" || "$TOKEN" == "null" ]]; then
    echo "[ERROR] Failed to get CLI token. Wrong credentials?"
    exit 1
  fi
  echo "$TOKEN" > "$TOKEN_FILE"
  mkdir -p "$(dirname "$SESSION_FILE")"
  echo "$LOGIN_RESP" > "$SESSION_FILE"
  echo "[INFO] Saved token to $TOKEN_FILE"
  echo "[INFO] Saved session to $SESSION_FILE (reused by ephapsys CLI)"
fi

AUTH_HEADER=(-H "Authorization: Bearer ${API_TOKEN}")
CLI_HEADER=(-H "Authorization: Bearer ${TOKEN}")

# Fetch templates from CLI endpoint (preferred) or API fallback
fetch_templates() {
  local resp
  resp=$(curl -sS "${CLI_HEADER[@]}" "${CLI_API}/models/list?type=TEMPLATE") || resp=""
  local items
  items=$(echo "$resp" | jq -r '.items // empty')
  if [[ -z "$items" ]]; then
    resp=$(curl -sS "${AUTH_HEADER[@]}" "${AOC_API}/models?type=TEMPLATE") || resp=""
  fi
  echo "$resp"
}

# ---------------------------------------------------------------------
# Register baseline model templates
# ---------------------------------------------------------------------
register_model () {
  local PROVIDER=$1 ID=$2 KIND=$3 PROVIDER_TOKEN=$4
  echo "------------------------------------------------------------"
  echo "[INFO] Registering model: $ID (kind=$KIND, provider=$PROVIDER)"
  RESPONSE=$(curl -s -X POST "$CLI_API/models/register" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d "{
      \"provider\": \"$PROVIDER\",
      \"provider_token\": \"$PROVIDER_TOKEN\",
      \"ids\": [\"$ID\"],
      \"repo_id\": \"$ID\",
      \"revision\": \"main\",
      \"auto_register\": true,
      \"model_kind\": \"$KIND\"
    }")
  if [[ -n "$PROVIDER_TOKEN" ]]; then
    echo "$RESPONSE" | sed "s/${PROVIDER_TOKEN}/********/g" | jq .
  else
    echo "$RESPONSE" | jq .
  fi
}

echo "============================================================"
echo "[STEP] Register baseline model templates"
echo "============================================================"
register_model "huggingface" "microsoft/speecht5_tts" "TTS" "$HF_TOKEN"
register_model "huggingface" "microsoft/speecht5_hifigan" "vocoder" "$HF_TOKEN"
register_model "huggingface" "openai/whisper-tiny.en" "STT" "$HF_TOKEN"
register_model "huggingface" "google/flan-t5-small" "Language" "$HF_TOKEN"
register_model "huggingface" "google/embeddinggemma-300m" "Embedding" "$HF_TOKEN"
register_model "huggingface" "hustvl/yolos-base" "Vision" "$HF_TOKEN"

# ---------------------------------------------------------------------
# Resolve latest model template IDs
# ---------------------------------------------------------------------
resolve_template() {
  local kind="$1"
  local match
  match=$(fetch_templates \
    | jq -r --arg kind "$kind" '
        (.items // .models // [])
        | map(select((.kind // .model_kind // "" | ascii_downcase)==($kind|ascii_downcase)))
        | sort_by(.created_at) | last | (.ID // ._id // empty // "")')
  if [[ -n "$match" ]]; then
    echo "$match"
    return
  fi
  # Fallback: try to match by repo/name when kind is missing (e.g., legacy vocoder)
  match=$(fetch_templates \
    | jq -r --arg kind "$kind" '
        (.items // .models // [])
        | map(select(.name // "" | test($kind; "i")))
        | sort_by(.created_at) | last | (.ID // ._id // empty // "")')
  echo "$match"
}

ensure_model_kind() {
  local mid="$1" kind="$2"
  [[ -z "$mid" || -z "$kind" ]] && return
  doc=$(curl -s -H "Authorization: Bearer ${API_TOKEN}" "${AOC_API}/models/${mid}" || echo "")
  current=$(echo "$doc" | jq -r '.kind // .model_kind // ""')
  if [[ -z "$current" || "$current" == "unknown" || "$current" == "null" ]]; then
    echo "[INFO] Patching model_kind for ${mid} -> ${kind}"
    curl -s -X PATCH "${AOC_API}/models/${mid}" \
      -H "Authorization: Bearer ${API_TOKEN}" \
      -H "Content-Type: application/json" \
      -d "{\"model_kind\":\"${kind}\"}" >/dev/null || true
  fi
}

MODEL_KINDS=("tts" "vocoder" "stt" "language" "embedding" "vision")
MODEL_IDS=()
VOCODER_ID=""

# ---------------------------------------------------------------------
# List available templates (parity with previous scripts)
# ---------------------------------------------------------------------
list_templates() {
  echo "============================================================"
  echo "[INFO] Available Model Templates (via /models?type=TEMPLATE)"
  echo "============================================================"
  resp=$(fetch_templates) || resp=""
  items=$(echo "$resp" | jq -r '.items // .models // empty')
  if [[ -z "$items" ]]; then
    echo "(no templates returned)"
  else
    echo "$resp" \
      | jq -r '(.items // .models // [])[] | [(.ID // ._id), (.kind // .model_kind // "unknown"), (.name // "noname"), (.status // "unknown")] | @tsv' \
      | awk 'BEGIN { printf "%-28s %-10s %-30s %-10s\n", "TemplateID", "Kind", "Name", "Status" }
             { printf "%-28s %-10s %-30s %-10s\n", $1, $2, $3, $4 }'
  fi
  echo "============================================================"
}

list_templates

# ---------------------------------------------------------------------
# Poll modulation status for a model template
# ---------------------------------------------------------------------
poll_until_done() {
  local model_id="$1"
  local deadline=$(( $(date +%s) + POLL_TIMEOUT_SEC ))
  echo "[POLL] Watching model_template=${model_id}"
  while true; do
    if (( $(date +%s) >= deadline )); then
      echo "[WARN] Timeout waiting for $model_id"
      return 1
    fi
    body=$(curl -sS "${AUTH_HEADER[@]}" "$AOC_API/models/$model_id")
    status=$(echo "$body" | jq -r '(.Modulation.status // "")')
    finished=$(echo "$body" | jq -r '(.Modulation.finished_at // "")')
    if [[ "$status" == "completed" || "$finished" != "null" ]]; then
      echo "[POLL] Completed for $model_id"
      return 0
    fi
    sleep $POLL_INTERVAL_SEC
  done
}

# ---------------------------------------------------------------------
# Optional modulation (skipped when --idempotent)
# ---------------------------------------------------------------------
run_modulator() {
  local kind="$1" path="$2"
  if [[ $IDEMPOTENT -eq 1 ]]; then
    echo "[SKIP] Idempotent mode: bypass modulation for ${kind}"
    local mid
    mid="$(resolve_template "$kind")"
    if [[ -z "$mid" ]]; then
      echo "[WARN] No template found for ${kind}; skipping idempotent publish."
      return 1
    fi
    echo "[INFO] Triggering idempotent publish for ${kind} (template=${mid})"
    curl -s -X POST "${AOC_API}/modulation/start" \
      -H "Authorization: Bearer ${API_TOKEN}" \
      -H "Content-Type: application/json" \
      -d "{
        \"model_template_id\": \"${mid}\",
        \"skip_modulation\": true,
        \"mode\": \"auto\",
        \"variant\": \"baseline\",
        \"kpi\": {\"targets\": [], \"maxSteps\": 1}
      }" >/dev/null || true
    return 0
  fi
  if [[ ! -x "${path}/modulate.sh" ]]; then
    echo "[WARN] No modulate.sh for ${kind}, skipping."
    return 1
  fi
  local mid
  mid="$(resolve_template "$kind")"
  if [[ -z "$mid" ]]; then
    echo "[ERROR] No TEMPLATE found for kind=${kind}"
    return 1
  fi
  cat > "${path}/.env" <<EOF
BASE_URL=${AOC_API}
AOC_MODULATION_TOKEN=${API_TOKEN:-$TOKEN}
MODEL_TEMPLATE_ID=${mid}
OUTDIR=./artifacts
EOF
  pushd "$path" >/dev/null
  ./modulate.sh
  popd >/dev/null
  poll_until_done "$mid" || true
}

echo "============================================================"
echo "[STEP] Modulate models (or skip if --idempotent)"
echo "============================================================"
for entry in "${MODEL_KINDS[@]}"; do
  run_modulator "$entry" "../../modulators/${entry}" || true
  mid=$(resolve_template "$entry")
  if [[ -n "$mid" ]]; then
    MODEL_IDS+=("$mid")
    if [[ "$entry" == "vocoder" ]]; then
      VOCODER_ID="$mid"
    fi
    ensure_model_kind "$mid" "$entry"
  fi
done

if [[ ${#MODEL_IDS[@]} -eq 0 ]]; then
  echo "[ERROR] No model templates resolved. Aborting."
  exit 1
fi

# ---------------------------------------------------------------------
# Create agent template
# ---------------------------------------------------------------------
if [[ -z "$LABEL" ]]; then
  if [[ -n "${AGENT_TEMPLATE_NAME:-}" ]]; then
    LABEL="$AGENT_TEMPLATE_NAME"
    echo "[INFO] Using AGENT_TEMPLATE_NAME from .env: $LABEL"
  else
    read -p "Enter Agent Template label/name: " LABEL
  fi
fi
MODELS_JSON=$(printf '%s\n' "${MODEL_IDS[@]}" | jq -R '{id: .}' | jq -s .)
build_model_entry() {
  local id="$1" kind="$2"
  kind=$(echo "$kind" | tr '[:upper:]' '[:lower:]')
  case "$kind" in
    vocoder)
      jq -n --arg id "$id" --arg kind "$kind" '{
        id: $id,
        config: {
          type: $kind,
          policies: []
        }
      }'
      ;;
    speaker)
      jq -n --arg id "$id" --arg kind "$kind" '{
        id: $id,
        config: {
          type: $kind,
          policies: []
        }
      }'
      ;;
    language)
      jq -n --arg id "$id" --arg kind "$kind" '{
        id: $id,
        config: {
          type: $kind,
          policies: [
            {type:"output_moderation", value:{categories:["self-harm","violence"], action:"block"}, status:"active"},
            {type:"prompt_injection_defense", value:{action:"block"}, status:"active"}
          ]
        }
      }'
      ;;
    vision)
      jq -n --arg id "$id" --arg kind "$kind" '{
        id: $id,
        config: {
          type: $kind,
          policies: [
            {type:"minors_action", value:"blur", status:"active"},
            {type:"sensitive_content", value:{action:"flag"}, status:"active"}
          ]
        }
      }'
      ;;
    stt)
      jq -n --arg id "$id" --arg kind "$kind" '{
        id: $id,
        config: {
          type: $kind,
          policies: [
            {type:"energy_limit", value:2048, status:"active"}
          ]
        }
      }'
      ;;
    tts)
      jq -n --arg id "$id" --arg kind "$kind" '{
        id: $id,
        config: {
          type: $kind,
          policies: [
            {type:"voice_style", value:"calm", status:"active"}
          ]
        }
      }'
      ;;
    rl)
      jq -n --arg id "$id" --arg kind "$kind" '{
        id: $id,
        config: {
          type: $kind,
          policies: [
            {type:"exploration_limit", value:0.2, status:"active"},
            {type:"reward_bounds", value:[-1,1], status:"active"}
          ]
        }
      }'
      ;;
    embedding)
      jq -n --arg id "$id" --arg kind "$kind" '{
        id: $id,
        config: {
          type: $kind,
          policies: [
            {type:"logging", value:"coarse", status:"active"}
          ]
        }
      }'
      ;;
    *)
      jq -n --arg id "$id" --arg kind "$kind" '{id:$id, config:{type:$kind, policies:[]}}'
      ;;
  esac
}

MODEL_ENTRIES=()
for mid in "${MODEL_IDS[@]}"; do
  doc=$(curl -s -H "Authorization: Bearer ${API_TOKEN}" "${AOC_API}/models/${mid}")
  mkind=$(echo "$doc" | jq -r '.kind // .model_kind // "language"')
  entry=$(build_model_entry "$mid" "$mkind")
  # Attach aux refs to TTS config
  if [[ "$mkind" =~ ^(tts|TTS)$ && -n "$VOCODER_ID" ]]; then
    entry=$(echo "$entry" | jq --arg vid "$VOCODER_ID" '
      .config.vocoder_model_id = $vid |
      .config.speaker_embeddings_uri = (.config.speaker_embeddings_uri // "https://huggingface.co/Matthijs/cmu-arctic-xvectors/resolve/main/speaker_embeddings.pt")
    ')
  fi
  MODEL_ENTRIES+=("$entry")
done
MODELS_JSON=$(printf '%s\n' "${MODEL_ENTRIES[@]}" | jq -s .)

tmp_models=$(mktemp)
echo "$MODELS_JSON" > "$tmp_models"
echo "[INFO] Creating agent template via CLI..."
set +e
RESP=$(ephapsys \
  --base-url "$AOC_API" \
  --api-key "$API_TOKEN" \
  agent create-template \
  --label "$LABEL" \
  --models-file "$tmp_models" 2>&1)
STATUS=$?
set -e
rm -f "$tmp_models"
if [[ $STATUS -ne 0 ]]; then
  echo "$RESP"
  echo "[WARN] CLI create-template failed; falling back to direct API call"
  RESP=$(curl -s -X POST "$AOC_API/agents" \
    -H "Authorization: Bearer ${API_TOKEN}" \
    -H "Content-Type: application/json" \
    -d "{
      \"label\": \"$LABEL\",
      \"type\": \"TEMPLATE\",
      \"models\": $MODELS_JSON
    }")
fi
echo "$RESP"

AGENT_ID=$(echo "$RESP" | jq -r '.id // ._id // .ID // .agent.id // .agent.public_id // empty')
if [[ -n "$AGENT_ID" && "$AGENT_ID" != "null" ]]; then
  echo "[SUCCESS] Agent template created. ID=$AGENT_ID"
  # Update AGENT_TEMPLATE_ID in .env if present
  if [[ -f ".env" ]]; then
    if grep -q '^AGENT_TEMPLATE_ID=' .env; then
      sed -i '' "s/^AGENT_TEMPLATE_ID=.*/AGENT_TEMPLATE_ID=${AGENT_ID}/" .env 2>/dev/null || sed -i "s/^AGENT_TEMPLATE_ID=.*/AGENT_TEMPLATE_ID=${AGENT_ID}/" .env
    else
      echo "AGENT_TEMPLATE_ID=${AGENT_ID}" >> .env
    fi
    echo "[INFO] Updated .env with AGENT_TEMPLATE_ID=${AGENT_ID}"
  fi
else
  echo "[ERROR] Failed to create agent template"
  exit 1
fi

if [[ $IDEMPOTENT -eq 1 ]]; then
  echo "[NOTE] Idempotent mode: modulation was skipped; only certificates were issued."
fi
