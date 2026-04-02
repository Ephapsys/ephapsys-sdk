#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f ".env" && -f ".env.example" ]]; then
  cp .env.example .env
  cat <<'EOF'
[INFO] Created .env from .env.example

Before continuing, edit .env and set:
  - AOC_BASE_URL=https://api.ephapsys.com
  - AOC_ORG_ID
  - AOC_PROVISIONING_TOKEN (boot_...)
  - AOC_MODULATION_TOKEN (mod_...)
  - HF_TOKEN only if your chosen model repo requires it

If you do not have an Ephapsys account yet, visit https://ephapsys.com and sign up first.

AOC means Agent Ops Center. You can retrieve the org ID and tokens in the AOC dashboard under Organization -> Tokens.

Then rerun:
  ./quickstart.sh
EOF
  exit 0
fi

info() {
  printf '[INFO] %s\n' "$*"
}

warn() {
  printf '[WARN] %s\n' "$*" >&2
}

save_env_var() {
  local key="$1"
  local value="$2"
  if grep -q "^${key}=" ".env"; then
    sed -i '' "s|^${key}=.*|${key}=${value}|" ".env" 2>/dev/null || sed -i "s|^${key}=.*|${key}=${value}|" ".env"
  else
    printf '\n%s=%s\n' "$key" "$value" >>".env"
  fi
}

resolve_existing_templates() {
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a

  local aoc_api api_token model_repo model_kind model_name agent_label
  aoc_api="${AOC_BASE_URL:-${AOC_API_URL:-${AOC_API_BASE:-${AOC_API:-http://localhost:7001}}}}"
  api_token="${API_TOKEN:-${AOC_MODULATION_TOKEN:-}}"
  model_repo="${HELLOWORLD_MODEL_REPO:-Qwen/Qwen3.5-0.8B}"
  model_kind="${HELLOWORLD_MODEL_KIND:-language}"
  model_name="${HELLOWORLD_MODEL_NAME:-HelloWorld Starter Model}"
  agent_label="${AGENT_TEMPLATE_NAME:-HelloWorld Agent Template}"

  if [[ -n "${MODEL_TEMPLATE_ID:-}" && -n "${AGENT_TEMPLATE_ID:-}" ]]; then
    info "Using MODEL_TEMPLATE_ID and AGENT_TEMPLATE_ID already present in .env."
    return 0
  fi

  if ! command -v curl >/dev/null 2>&1 || ! command -v jq >/dev/null 2>&1; then
    warn "curl or jq is missing; skipping starter template lookup."
    return 1
  fi

  if [[ -z "$api_token" ]]; then
    warn "AOC_MODULATION_TOKEN is not set; skipping starter template lookup."
    return 1
  fi

  local auth_header model_id agent_id
  auth_header=(-H "Authorization: Bearer ${api_token}")

  if [[ -z "${MODEL_TEMPLATE_ID:-}" ]]; then
    model_id="$(
      curl -sS "${auth_header[@]}" "${aoc_api}/models?type=TEMPLATE" | jq -r \
        --arg repo "$model_repo" \
        --arg kind "$model_kind" \
        --arg name "$model_name" '
        (.items // [])
        | map(select((((.model_kind // .kind // "") | ascii_downcase) == ($kind | ascii_downcase))
          and ((.source_repo // "") == $repo or (.name // "") == $name or (.name // "") == ("HuggingFace " + $repo))))
        | sort_by(.created_at // 0)
        | last
        | (.ID // .public_id // .internal_id // ._id // empty)'
    )"
    if [[ -n "$model_id" ]]; then
      info "Found existing HelloWorld starter model template: ${model_id}"
      save_env_var MODEL_TEMPLATE_ID "$model_id"
    fi
  fi

  if [[ -z "${AGENT_TEMPLATE_ID:-}" ]]; then
    agent_id="$(
      curl -sS "${auth_header[@]}" "${aoc_api}/agents?type=TEMPLATE" | jq -r \
        --arg label "$agent_label" '
        map(select((.label // "") == $label))
        | first
        | (.id // .public_id // .ID // ._id // empty)'
    )"
    if [[ -n "$agent_id" ]]; then
      info "Found existing HelloWorld starter agent template: ${agent_id}"
      save_env_var AGENT_TEMPLATE_ID "$agent_id"
    fi
  fi

  set -a
  # shellcheck disable=SC1091
  source .env
  set +a

  [[ -n "${MODEL_TEMPLATE_ID:-}" && -n "${AGENT_TEMPLATE_ID:-}" ]]
}

if ! resolve_existing_templates; then
  info "No reusable HelloWorld starter templates found; bootstrapping with push.sh."
  ./push.sh --mode local "$@"
fi

./run.sh --local
