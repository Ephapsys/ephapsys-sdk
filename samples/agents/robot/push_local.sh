#!/usr/bin/env bash
set -euo pipefail

# =====================================================================
# push.sh
# - Registers baseline model templates (via CLI login)
# - Defaults to idempotent publish; use --no-idempotent for full modulation
# - Creates an agent template bundling the model templates
# =====================================================================

USAGE="Usage: $0 [--idempotent|--no-idempotent] [--label \"Robot Agent Template\"]"
POLL_INTERVAL_SEC=5
POLL_TIMEOUT_SEC=7200
CLI_TOKEN_FILE=".cli_token"

IDEMPOTENT=1
LABEL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --idempotent) IDEMPOTENT=1; shift ;;
    --no-idempotent) IDEMPOTENT=0; shift ;;
    --label) LABEL="$2"; shift 2 ;;
    -h|--help) echo "$USAGE"; exit 0 ;;
    *) echo "Unknown arg: $1"; echo "$USAGE"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------
# Env + defaults
# ---------------------------------------------------------------------
PRESERVE_AOC_BASE_URL="${AOC_BASE_URL-}"
PRESERVE_AOC_API_URL="${AOC_API_URL-}"
PRESERVE_AOC_API="${AOC_API-}"
PRESERVE_API_TOKEN="${API_TOKEN-}"
PRESERVE_AOC_MODULATION_TOKEN="${AOC_MODULATION_TOKEN-}"
PRESERVE_HF_TOKEN="${HF_TOKEN-}"
PRESERVE_AGENT_TEMPLATE_NAME="${AGENT_TEMPLATE_NAME-}"
PRESERVE_ROBOT_TTS_REPO="${ROBOT_TTS_REPO-}"
PRESERVE_ROBOT_TTS_NAME="${ROBOT_TTS_NAME-}"
PRESERVE_ROBOT_VOCODER_REPO="${ROBOT_VOCODER_REPO-}"
PRESERVE_ROBOT_VOCODER_NAME="${ROBOT_VOCODER_NAME-}"
PRESERVE_ROBOT_STT_REPO="${ROBOT_STT_REPO-}"
PRESERVE_ROBOT_STT_NAME="${ROBOT_STT_NAME-}"
PRESERVE_ROBOT_LANGUAGE_REPO="${ROBOT_LANGUAGE_REPO-}"
PRESERVE_ROBOT_LANGUAGE_NAME="${ROBOT_LANGUAGE_NAME-}"
PRESERVE_ROBOT_EMBEDDING_REPO="${ROBOT_EMBEDDING_REPO-}"
PRESERVE_ROBOT_EMBEDDING_NAME="${ROBOT_EMBEDDING_NAME-}"
PRESERVE_ROBOT_VISION_REPO="${ROBOT_VISION_REPO-}"
PRESERVE_ROBOT_VISION_NAME="${ROBOT_VISION_NAME-}"
PRESERVE_ROBOT_WORLD_REPO="${ROBOT_WORLD_REPO-}"
PRESERVE_ROBOT_WORLD_NAME="${ROBOT_WORLD_NAME-}"
PRESERVE_ROBOT_ENABLE_WORLD_MODEL="${ROBOT_ENABLE_WORLD_MODEL-}"

[[ -f .env ]] && set -o allexport && source .env && set +o allexport

[[ -n "${PRESERVE_AOC_BASE_URL}" ]] && export AOC_BASE_URL="${PRESERVE_AOC_BASE_URL}"
[[ -n "${PRESERVE_AOC_API_URL}" ]] && export AOC_API_URL="${PRESERVE_AOC_API_URL}"
[[ -n "${PRESERVE_AOC_API}" ]] && export AOC_API="${PRESERVE_AOC_API}"
[[ -n "${PRESERVE_API_TOKEN}" ]] && export API_TOKEN="${PRESERVE_API_TOKEN}"
[[ -n "${PRESERVE_AOC_MODULATION_TOKEN}" ]] && export AOC_MODULATION_TOKEN="${PRESERVE_AOC_MODULATION_TOKEN}"
[[ -n "${PRESERVE_HF_TOKEN}" ]] && export HF_TOKEN="${PRESERVE_HF_TOKEN}"
[[ -n "${PRESERVE_AGENT_TEMPLATE_NAME}" ]] && export AGENT_TEMPLATE_NAME="${PRESERVE_AGENT_TEMPLATE_NAME}"
[[ -n "${PRESERVE_ROBOT_TTS_REPO}" ]] && export ROBOT_TTS_REPO="${PRESERVE_ROBOT_TTS_REPO}"
[[ -n "${PRESERVE_ROBOT_TTS_NAME}" ]] && export ROBOT_TTS_NAME="${PRESERVE_ROBOT_TTS_NAME}"
[[ -n "${PRESERVE_ROBOT_VOCODER_REPO}" ]] && export ROBOT_VOCODER_REPO="${PRESERVE_ROBOT_VOCODER_REPO}"
[[ -n "${PRESERVE_ROBOT_VOCODER_NAME}" ]] && export ROBOT_VOCODER_NAME="${PRESERVE_ROBOT_VOCODER_NAME}"
[[ -n "${PRESERVE_ROBOT_STT_REPO}" ]] && export ROBOT_STT_REPO="${PRESERVE_ROBOT_STT_REPO}"
[[ -n "${PRESERVE_ROBOT_STT_NAME}" ]] && export ROBOT_STT_NAME="${PRESERVE_ROBOT_STT_NAME}"
[[ -n "${PRESERVE_ROBOT_LANGUAGE_REPO}" ]] && export ROBOT_LANGUAGE_REPO="${PRESERVE_ROBOT_LANGUAGE_REPO}"
[[ -n "${PRESERVE_ROBOT_LANGUAGE_NAME}" ]] && export ROBOT_LANGUAGE_NAME="${PRESERVE_ROBOT_LANGUAGE_NAME}"
[[ -n "${PRESERVE_ROBOT_EMBEDDING_REPO}" ]] && export ROBOT_EMBEDDING_REPO="${PRESERVE_ROBOT_EMBEDDING_REPO}"
[[ -n "${PRESERVE_ROBOT_EMBEDDING_NAME}" ]] && export ROBOT_EMBEDDING_NAME="${PRESERVE_ROBOT_EMBEDDING_NAME}"
[[ -n "${PRESERVE_ROBOT_VISION_REPO}" ]] && export ROBOT_VISION_REPO="${PRESERVE_ROBOT_VISION_REPO}"
[[ -n "${PRESERVE_ROBOT_VISION_NAME}" ]] && export ROBOT_VISION_NAME="${PRESERVE_ROBOT_VISION_NAME}"
[[ -n "${PRESERVE_ROBOT_WORLD_REPO}" ]] && export ROBOT_WORLD_REPO="${PRESERVE_ROBOT_WORLD_REPO}"
[[ -n "${PRESERVE_ROBOT_WORLD_NAME}" ]] && export ROBOT_WORLD_NAME="${PRESERVE_ROBOT_WORLD_NAME}"
[[ -n "${PRESERVE_ROBOT_ENABLE_WORLD_MODEL}" ]] && export ROBOT_ENABLE_WORLD_MODEL="${PRESERVE_ROBOT_ENABLE_WORLD_MODEL}"

BASE_URL="${AOC_BASE_URL:-${AOC_API_URL:-${AOC_API:-${BASE_URL:-http://localhost:7001}}}}"
CLI_API="${BASE_URL}/cli"
SESSION_FILE="${HOME}/.ephapsys_state/session.json"
API_TOKEN="${API_TOKEN:-${AOC_MODULATION_TOKEN:-}}"
HF_TOKEN="${HF_TOKEN:-""}"
ROBOT_TTS_REPO="${ROBOT_TTS_REPO:-microsoft/speecht5_tts}"
ROBOT_TTS_NAME="${ROBOT_TTS_NAME:-Robot TTS Model}"
ROBOT_VOCODER_REPO="${ROBOT_VOCODER_REPO:-microsoft/speecht5_hifigan}"
ROBOT_VOCODER_NAME="${ROBOT_VOCODER_NAME:-Robot Vocoder Model}"
ROBOT_STT_REPO="${ROBOT_STT_REPO:-openai/whisper-tiny.en}"
ROBOT_STT_NAME="${ROBOT_STT_NAME:-Robot STT Model}"
ROBOT_LANGUAGE_REPO="${ROBOT_LANGUAGE_REPO:-Qwen/Qwen3.5-0.8B}"
ROBOT_LANGUAGE_NAME="${ROBOT_LANGUAGE_NAME:-Robot Language Model}"
ROBOT_EMBEDDING_REPO="${ROBOT_EMBEDDING_REPO:-sentence-transformers/all-MiniLM-L6-v2}"
ROBOT_EMBEDDING_NAME="${ROBOT_EMBEDDING_NAME:-Robot Embedding Model}"
ROBOT_VISION_REPO="${ROBOT_VISION_REPO:-hustvl/yolos-base}"
ROBOT_VISION_NAME="${ROBOT_VISION_NAME:-Robot Vision Model}"
ROBOT_WORLD_REPO="${ROBOT_WORLD_REPO:-facebook/vjepa2-vitl-fpc64-256}"
ROBOT_WORLD_NAME="${ROBOT_WORLD_NAME:-Robot World Model}"
ROBOT_ENABLE_WORLD_MODEL="${ROBOT_ENABLE_WORLD_MODEL:-0}"

trim_inline() {
  local value="${1-}"
  value="${value//$'\r'/}"
  value="${value//$'\n'/}"
  printf '%s' "$value"
}

BASE_URL="$(trim_inline "$BASE_URL")"
CLI_API="$(trim_inline "$CLI_API")"
API_TOKEN="$(trim_inline "$API_TOKEN")"
HF_TOKEN="$(trim_inline "$HF_TOKEN")"
ROBOT_ENABLE_WORLD_MODEL="$(trim_inline "$ROBOT_ENABLE_WORLD_MODEL")"

MODEL_SPEC_JSON="$(jq -n \
  --arg tts_repo "$ROBOT_TTS_REPO" --arg tts_name "$ROBOT_TTS_NAME" \
  --arg voc_repo "$ROBOT_VOCODER_REPO" --arg voc_name "$ROBOT_VOCODER_NAME" \
  --arg stt_repo "$ROBOT_STT_REPO" --arg stt_name "$ROBOT_STT_NAME" \
  --arg lang_repo "$ROBOT_LANGUAGE_REPO" --arg lang_name "$ROBOT_LANGUAGE_NAME" \
  --arg emb_repo "$ROBOT_EMBEDDING_REPO" --arg emb_name "$ROBOT_EMBEDDING_NAME" \
  --arg vis_repo "$ROBOT_VISION_REPO" --arg vis_name "$ROBOT_VISION_NAME" \
  --arg world_repo "$ROBOT_WORLD_REPO" --arg world_name "$ROBOT_WORLD_NAME" \
  --arg world_enabled "$ROBOT_ENABLE_WORLD_MODEL" '
  ([
    {kind:"tts", repo:$tts_repo, name:$tts_name},
    {kind:"vocoder", repo:$voc_repo, name:$voc_name},
    {kind:"stt", repo:$stt_repo, name:$stt_name},
    {kind:"language", repo:$lang_repo, name:$lang_name},
    {kind:"embedding", repo:$emb_repo, name:$emb_name},
    {kind:"vision", repo:$vis_repo, name:$vis_name}
  ] + (if ($world_enabled | ascii_downcase) == "1" or ($world_enabled | ascii_downcase) == "true" or ($world_enabled | ascii_downcase) == "yes"
       then [{kind:"world", repo:$world_repo, name:$world_name}]
       else []
       end))'
)"

info() {
  printf '[INFO] %s\n' "$*"
}

warn() {
  printf '[WARN] %s\n' "$*" >&2
}

error() {
  printf '[ERROR] %s\n' "$*" >&2
}

if [[ -z "$API_TOKEN" ]]; then
  error "API_TOKEN is required to create agent templates. Set API_TOKEN or AOC_MODULATION_TOKEN in .env or environment."
  exit 1
fi

# ---------------------------------------------------------------------
# CLI login (reuse ephapsys login session or cached token if valid)
# ---------------------------------------------------------------------
cli_login() {
  local token=""
  local session_base_url=""
  if [[ -f "$SESSION_FILE" ]]; then
    token=$(jq -r '.token // empty' "$SESSION_FILE" 2>/dev/null || true)
    session_base_url=$(jq -r '.base_url // empty' "$SESSION_FILE" 2>/dev/null || true)
    token="$(trim_inline "$token")"
    session_base_url="$(trim_inline "$session_base_url")"
    if [[ -n "$token" ]]; then
      local code
      code=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $token" "$CLI_API/models/list" 2>/dev/null || true)
      if [[ "$code" == "200" ]]; then
        info "Reusing CLI session from ~/.ephapsys_state/session.json" >&2
        printf '%s' "$token"
        return
      fi
      if [[ -n "$session_base_url" && "$session_base_url" != "$BASE_URL" ]]; then
        warn "Saved CLI session targets $session_base_url but push.sh is targeting $BASE_URL" >&2
      fi
    fi
  fi
  if [[ -f "$CLI_TOKEN_FILE" ]]; then
    token=$(cat "$CLI_TOKEN_FILE")
    token="$(trim_inline "$token")"
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $token" "$CLI_API/models/list" 2>/dev/null || true)
    if [[ "$code" == "200" ]]; then
      info "Reusing cached CLI token from .cli_token" >&2
      printf '%s' "$token"
      return
    fi
  fi

  if [[ ! -t 0 ]]; then
    error "No valid CLI session found for model registration. Run 'ephapsys login' first, then rerun push.sh."
    exit 1
  fi

  warn "No reusable CLI session found for ${BASE_URL}; falling back to interactive CLI login." >&2

  local cli_user cli_pass login_resp
  read -r -p "Enter CLI username (email): " cli_user
  read -r -s -p "Enter CLI password: " cli_pass
  printf '\n'
  login_resp=$(curl -sS -X POST "$CLI_API/login" \
    -H "Content-Type: application/json" \
    -d "{\"username\":\"${cli_user}\",\"password\":\"${cli_pass}\"}")
  token=$(echo "$login_resp" | jq -r '.token // empty')
  token="$(trim_inline "$token")"
  if [[ -z "$token" ]]; then
    error "CLI login failed"
    echo "$login_resp" >&2
    exit 1
  fi
  printf '%s' "$token" >"$CLI_TOKEN_FILE"
  mkdir -p "$(dirname "$SESSION_FILE")"
  jq -n --arg token "$token" --arg base_url "$BASE_URL" --argjson raw "$login_resp" '
    ($raw | if type == "object" then . else {} end) + {token: $token, base_url: $base_url}
  ' >"$SESSION_FILE"
  printf '%s' "$token"
}

TOKEN="$(cli_login)"
TOKEN="$(trim_inline "$TOKEN")"

AUTH_HEADER=(-H "Authorization: Bearer ${API_TOKEN}")
CLI_HEADER=(-H "Authorization: Bearer ${TOKEN}")

# Fetch templates from CLI endpoint (preferred) or API fallback
fetch_templates() {
  local resp
  resp=$(curl -sS "${CLI_HEADER[@]}" "${CLI_API}/models/list?type=TEMPLATE") || resp=""
  local items
  items=$(echo "$resp" | jq -r '.items // empty')
  if [[ -z "$items" ]]; then
    resp=$(curl -sS "${AUTH_HEADER[@]}" "${BASE_URL}/models?type=TEMPLATE") || resp=""
  fi
  echo "$resp"
}

# ---------------------------------------------------------------------
# Register baseline model templates
# ---------------------------------------------------------------------
register_model () {
  local PROVIDER=$1 ID=$2 KIND=$3 MODEL_NAME=$4 PROVIDER_TOKEN=$5
  local response=""
  local http_code=""
  local body=""
  echo "------------------------------------------------------------"
  echo "[INFO] Registering model: $ID (kind=$KIND, name=$MODEL_NAME, provider=$PROVIDER)"
  response=$(curl -sS -w $'\n%{http_code}' -X POST "$CLI_API/models/register" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d "{
      \"provider\": \"$PROVIDER\",
      \"provider_token\": \"$PROVIDER_TOKEN\",
      \"ids\": [\"$ID\"],
      \"repo_id\": \"$ID\",
      \"name\": \"$MODEL_NAME\",
      \"revision\": \"main\",
      \"auto_register\": true,
      \"model_kind\": \"$KIND\"
    }") || {
      error "Model registration request failed for $ID"
      return 1
    }
  http_code="${response##*$'\n'}"
  body="${response%$'\n'*}"
  if [[ "$http_code" != "200" && "$http_code" != "201" ]]; then
    error "Model registration failed for $ID (HTTP $http_code)"
    if [[ -n "$PROVIDER_TOKEN" ]]; then
      printf '%s\n' "$body" | sed "s/${PROVIDER_TOKEN}/********/g" >&2
    else
      printf '%s\n' "$body" >&2
    fi
    return 1
  fi
  local inserted_count skipped_count model_id
  inserted_count=$(printf '%s\n' "$body" | jq -r '(.inserted // []) | length' 2>/dev/null || printf '0')
  skipped_count=$(printf '%s\n' "$body" | jq -r '(.skipped // []) | length' 2>/dev/null || printf '0')
  model_id=$(printf '%s\n' "$body" | jq -r '(.inserted // [])[0].model.public_id // (.inserted // [])[0].model._id // empty' 2>/dev/null || true)
  if [[ -n "$model_id" ]]; then
    info "Model registration request accepted for $ID -> ${model_id} (inserted=${inserted_count}, skipped=${skipped_count})."
  else
    info "Model registration request accepted for $ID (inserted=${inserted_count}, skipped=${skipped_count})."
  fi
}

echo "============================================================"
echo "[STEP] Register baseline model templates"
echo "============================================================"
info "First-time registration can pause for several minutes per model while AOC downloads upstream Hugging Face assets."
while IFS=$'\t' read -r kind repo name; do
  existing="$(fetch_templates | jq -r --arg kind "$kind" --arg repo "$repo" --arg name "$name" '
    (.items // .models // [])
    | map(select(
        ((.kind // .model_kind // "" | ascii_downcase)==($kind|ascii_downcase))
        and (((.source_repo // .repo_id // "") == $repo) or ((.name // "") == $name))
      ))
    | sort_by(.created_at) | last | (.ID // ._id // empty // "")'
  )"
  if [[ -n "$existing" ]]; then
    info "Reusing existing ${kind} template: ${existing} (${name})"
    continue
  fi
  info "No existing ${kind} template matched ${name}; registering ${repo}. Waiting on AOC until registration returns."
  register_model "huggingface" "$repo" "$kind" "$name" "$HF_TOKEN"
done < <(echo "$MODEL_SPEC_JSON" | jq -r '.[] | [.kind, .repo, .name] | @tsv')

# ---------------------------------------------------------------------
# Resolve latest model template IDs
# ---------------------------------------------------------------------
resolve_template() {
  local kind="$1"
  local repo="$2"
  local name="$3"
  local match
  match=$(fetch_templates | jq -r --arg kind "$kind" --arg repo "$repo" --arg name "$name" '
        (.items // .models // [])
        | map(select(
            ((.kind // .model_kind // "" | ascii_downcase)==($kind|ascii_downcase))
            and (((.source_repo // .repo_id // "") == $repo) or ((.name // "") == $name))
          ))
        | sort_by(.created_at) | last | (.ID // ._id // empty // "")')
  if [[ -n "$match" ]]; then
    echo "$match"
    return
  fi
  match=$(fetch_templates | jq -r --arg repo "$repo" --arg name "$name" '
        (.items // .models // [])
        | map(select(((.source_repo // .repo_id // "") == $repo) or ((.name // "") == $name)))
        | sort_by(.created_at) | last | (.ID // ._id // empty // "")')
  echo "$match"
}

ensure_model_kind() {
  local mid="$1" kind="$2"
  [[ -z "$mid" || -z "$kind" ]] && return
  doc=$(curl -s -H "Authorization: Bearer ${API_TOKEN}" "${BASE_URL}/models/${mid}" || echo "")
  current=$(echo "$doc" | jq -r '.kind // .model_kind // ""')
  if [[ -z "$current" || "$current" == "unknown" || "$current" == "null" ]]; then
    echo "[INFO] Patching model_kind for ${mid} -> ${kind}"
    curl -s -X PATCH "${BASE_URL}/models/${mid}" \
      -H "Authorization: Bearer ${API_TOKEN}" \
      -H "Content-Type: application/json" \
      -d "{\"model_kind\":\"${kind}\"}" >/dev/null || true
  fi
}

MODEL_IDS=()
VOCODER_ID=""
save_env_var() {
  local key="$1"
  local value="$2"
  if [[ ! -f ".env" ]]; then
    return
  fi
  if grep -q "^${key}=" .env; then
    sed -i '' "s|^${key}=.*|${key}=${value}|" .env 2>/dev/null || sed -i "s|^${key}=.*|${key}=${value}|" .env
  else
    printf '\n%s=%s\n' "$key" "$value" >> .env
  fi
}

resolve_agent_template() {
  local label="$1"
  curl -sS -H "Authorization: Bearer ${API_TOKEN}" "${BASE_URL}/agents?type=TEMPLATE" | jq -r \
    --arg label "$label" '
      map(select((.label // "") == $label))
      | sort_by(.created_at // 0)
      | last
      | (.id // .public_id // .ID // ._id // empty)
    '
}

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
    body=$(curl -sS "${AUTH_HEADER[@]}" "$BASE_URL/models/$model_id")
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
# Optional modulation (skipped when idempotent publish is enabled)
# ---------------------------------------------------------------------
run_modulator() {
  local kind="$1" path="$2"
  if [[ $IDEMPOTENT -eq 1 ]]; then
    echo "[SKIP] Idempotent mode: bypass modulation for ${kind}"
    local mid
    local repo name
    repo=$(echo "$MODEL_SPEC_JSON" | jq -r --arg kind "$kind" '.[] | select(.kind == $kind) | .repo')
    name=$(echo "$MODEL_SPEC_JSON" | jq -r --arg kind "$kind" '.[] | select(.kind == $kind) | .name')
    mid="$(resolve_template "$kind" "$repo" "$name")"
    if [[ -z "$mid" ]]; then
      echo "[WARN] No template found for ${kind}; skipping idempotent publish."
      return 1
    fi
    echo "[INFO] Triggering idempotent publish for ${kind} (template=${mid})"
    curl -s -X POST "${BASE_URL}/modulation/start" \
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
  local repo name
  repo=$(echo "$MODEL_SPEC_JSON" | jq -r --arg kind "$kind" '.[] | select(.kind == $kind) | .repo')
  name=$(echo "$MODEL_SPEC_JSON" | jq -r --arg kind "$kind" '.[] | select(.kind == $kind) | .name')
  mid="$(resolve_template "$kind" "$repo" "$name")"
  if [[ -z "$mid" ]]; then
    echo "[ERROR] No TEMPLATE found for kind=${kind}"
    return 1
  fi
  cat > "${path}/.env" <<EOF
BASE_URL=${BASE_URL}
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
echo "[STEP] Modulate models (or skip in idempotent mode)"
echo "============================================================"
while IFS=$'\t' read -r entry repo name; do
  run_modulator "$entry" "../../modulators/${entry}" || true
  mid=$(resolve_template "$entry" "$repo" "$name")
  if [[ -n "$mid" ]]; then
    MODEL_IDS+=("$mid")
    if [[ "$entry" == "vocoder" ]]; then
      VOCODER_ID="$mid"
    fi
    ensure_model_kind "$mid" "$entry"
    upper_key=$(printf '%s' "$entry" | tr '[:lower:]' '[:upper:]')
    save_env_var "ROBOT_${upper_key}_TEMPLATE_ID" "$mid"
  fi
done < <(echo "$MODEL_SPEC_JSON" | jq -r '.[] | [.kind, .repo, .name] | @tsv')

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
echo "[STEP] Resolving existing agent template"
EXISTING_AGENT_ID="$(resolve_agent_template "$LABEL")"
if [[ -n "$EXISTING_AGENT_ID" && "$EXISTING_AGENT_ID" != "null" ]]; then
  echo "[INFO] Reusing existing agent template: ${EXISTING_AGENT_ID} (${LABEL})"
  save_env_var "AGENT_TEMPLATE_ID" "$EXISTING_AGENT_ID"
  echo "[INFO] Updated .env with AGENT_TEMPLATE_ID=${EXISTING_AGENT_ID}"
  if [[ $IDEMPOTENT -eq 1 ]]; then
    echo "[NOTE] Idempotent mode: modulation was skipped; only certificates were issued."
  fi
  exit 0
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
  doc=$(curl -s -H "Authorization: Bearer ${API_TOKEN}" "${BASE_URL}/models/${mid}")
  mkind=$(echo "$doc" | jq -r '.kind // .model_kind // "language"')
  entry=$(build_model_entry "$mid" "$mkind")
  # Attach aux refs to TTS config
  if [[ "$mkind" =~ ^(tts|TTS)$ && -n "$VOCODER_ID" ]]; then
    entry=$(echo "$entry" | jq --arg vid "$VOCODER_ID" '
      .config.vocoder_model_id = $vid
    ')
  fi
  MODEL_ENTRIES+=("$entry")
done
MODELS_JSON=$(printf '%s\n' "${MODEL_ENTRIES[@]}" | jq -s .)

tmp_models=$(mktemp)
echo "$MODELS_JSON" > "$tmp_models"
echo "[INFO] Creating agent template via API..."
rm -f "$tmp_models"
RESP=$(curl -s -X POST "$BASE_URL/agents" \
  -H "Authorization: Bearer ${API_TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{
    \"label\": \"$LABEL\",
    \"type\": \"TEMPLATE\",
    \"models\": $MODELS_JSON
  }")
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
