# HelloWorld Agent (Sample with Ephapsys SDK)

This sample is the smallest useful Ephapsys agent.
It can:
- personalize an agent instance against AOC
- prepare the runtime automatically
- open an interactive chat session locally or on GCP

`AOC` means **Agent Ops Center**, the Ephapsys control plane where you manage org tokens, model templates, and agent templates.

## Fastest Paths

Local default:

```bash
cd ephapsys-sdk/samples/agents/helloworld
./quickstart.sh
```

GCP default:

```bash
cd ephapsys-sdk/samples/agents/helloworld
./quickstart.sh --gcp
```

`quickstart.sh` does:
- creates `.env` from `.env.example` if needed, then stops so you can fill secrets
- reuses existing `MODEL_TEMPLATE_ID` / `AGENT_TEMPLATE_ID` when already present
- otherwise falls back to `./push.sh` to bootstrap starter templates
- launches `./run.sh` in local mode by default, or GCP mode with `--gcp`

## Required Credentials

Before first run, populate [`.env`](/Users/aidevmac/Projects/Ephapsys/Product/ephapsys-sdk/samples/agents/helloworld/.env) with:
- `AOC_BASE_URL`
- `AOC_ORG_ID`
- `AOC_PROVISIONING_TOKEN`
- `AOC_MODULATION_TOKEN` if you plan to run `./push.sh`

Leave these blank on the first starter-template run unless you already know them:
- `MODEL_TEMPLATE_ID`
- `AGENT_TEMPLATE_ID`

Important:
- `AOC_PROVISIONING_TOKEN` is required at runtime. If it is stale, interactive chat will fail with a `401 invalid provisioning token`.
- `AOC_MODULATION_TOKEN` is used by `./push.sh`, not by the runtime chat session itself.
- `quickstart.sh` and `run_gcp.sh` upload your current local `.env` to the VM, so updating the local token is enough for the next GCP reuse run.

## Local Workflow

Recommended first run:

```bash
cd ephapsys-sdk/samples/agents/helloworld
cp .env.example .env
./quickstart.sh
```

`run.sh` defaults to local mode and delegates to `run_local.sh`.

Useful commands:

```bash
./run.sh
./run.sh --local
./push.sh
./push.sh --local --no-idempotent
```

If you want to develop against the checked-out SDK instead of the published package:

```bash
HELLOWORLD_USE_LOCAL_SDK=1 ./run.sh
```

## GCP Workflow

Recommended first run:

```bash
cd ephapsys-sdk/samples/agents/helloworld
cp .env.example .env
cp .env.gcp.example .env.gcp
./quickstart.sh --gcp
```

### What `./quickstart.sh --gcp` actually does

The current GCP path is VM-first and reuse-friendly:
- reads local [`.env.gcp`](/Users/aidevmac/Projects/Ephapsys/Product/ephapsys-sdk/samples/agents/helloworld/.env.gcp)
- reuses the last successful VM from [`.last_gcp_instance`](/Users/aidevmac/Projects/Ephapsys/Product/ephapsys-sdk/samples/agents/helloworld/.last_gcp_instance) when possible
- otherwise searches across configured GPU / zone / region fallbacks
- waits for SSH to become ready
- copies only the needed sample files to the VM
- uploads your current local `.env`
- skips bootstrap when the VM already has `~/helloworld/.venv/bin/python`
- opens the interactive chat session by default

This means the normal developer loop is:
1. get one working GPU VM
2. keep it running
3. rerun `./quickstart.sh --gcp`
4. let the script reuse the VM and attach quickly

Do not delete the VM between runs unless you want to reacquire GPU capacity from scratch.

### Required GCP setup

Populate [`.env.gcp`](/Users/aidevmac/Projects/Ephapsys/Product/ephapsys-sdk/samples/agents/helloworld/.env.gcp) with at least:
- `PROJECT_ID`
- `ZONE`
- `MACHINE_TYPE`
- `DISK_SIZE`
- `IMAGE_FAMILY`
- `IMAGE_PROJECT`
- `INSTANCE_PREFIX`

Recommended for GPU hunting:
- `USE_GPU=1`
- `GPU_TYPE=t4`
- `GPU_FALLBACKS=t4,l4,a100`
- `ZONE_FALLBACKS=us-central1-a,us-central1-b,us-central1-c,us-central1-f`
- `REGION_FALLBACKS=us-central1,us-east1,us-west1`

Notes:
- `run_gcp.sh` clamps `DISK_SIZE` to at least `100GB` for the current GPU image families.
- The script prints colored candidate search logs and a green `[SELECTED]` line once a real VM is chosen.
- GPU quota and zonal capacity are different constraints. Even with quota, a given zone can still return `ZONE_RESOURCE_POOL_EXHAUSTED`.

### Useful GCP commands

Reuse current VM and attach:

```bash
./quickstart.sh --gcp
```

Direct reconnect helper:

```bash
./reattach_gcp.sh
```

Deploy without opening chat immediately:

```bash
./run.sh --gcp --no-interactive
```

Force a fresh VM instead of reusing the last one:

```bash
./run.sh --gcp --fresh-instance
```

Prefer reusing an existing VM explicitly:

```bash
./run.sh --gcp --reuse-instance
```

Stop the VM when you are done to control cost:

```bash
gcloud compute instances stop <instance> --project <project> --zone <zone>
```

Delete it only when you really want to throw away the disk/runtime state:

```bash
gcloud compute instances delete <instance> --project <project> --zone <zone>
```

### Interactive behavior

GCP mode now defaults to interactive chat.

If the VM is already prepared, reuse mode should do a light refresh only:
- copy `helloworld_agent.py`
- copy `run_local.sh`
- copy `reattach_gcp.sh`
- upload the latest `.env`
- skip full apt / venv / pip bootstrap
- open chat

## Push / Template Bootstrap

If you want the sample to bootstrap starter templates instead of relying on existing IDs:

```bash
./push.sh
```

Or for GCP-based modulation:

```bash
./push.sh --gcp --gpu t4
```

`push.sh` can:
- resolve or register the canonical HelloWorld language model template
- idempotently publish it by default
- create or reuse an agent template bound to that model
- write `MODEL_TEMPLATE_ID` and `AGENT_TEMPLATE_ID` back into `.env`

By default, the starter path prefers idempotent publish mode.
Use `--no-idempotent` if you explicitly want a full modulation run.

## Common Failures

- `invalid provisioning token`
  - `AOC_PROVISIONING_TOKEN` in `.env` is stale or wrong for that AOC environment
- `404 Agent template not found`
  - `AGENT_TEMPLATE_ID` is wrong or points at the wrong environment
- `language_model_missing`
  - the agent template is not bound to a language model
- `language_model_not_ready` or missing artifacts
  - the linked language model exists but is not fully published/modulated
- `ZONE_RESOURCE_POOL_EXHAUSTED`
  - the requested GPU exists in principle, but not in that zone right now
- SSH connection refused on a fresh VM
  - the VM is still booting; the script now retries for SSH readiness automatically

## Files

- [`helloworld_agent.py`](/Users/aidevmac/Projects/Ephapsys/Product/ephapsys-sdk/samples/agents/helloworld/helloworld_agent.py) → minimal TrustedAgent demo
- [`quickstart.sh`](/Users/aidevmac/Projects/Ephapsys/Product/ephapsys-sdk/samples/agents/helloworld/quickstart.sh) → main entrypoint; local by default, GCP with `--gcp`
- [`run.sh`](/Users/aidevmac/Projects/Ephapsys/Product/ephapsys-sdk/samples/agents/helloworld/run.sh) → public execution entrypoint
- [`run_local.sh`](/Users/aidevmac/Projects/Ephapsys/Product/ephapsys-sdk/samples/agents/helloworld/run_local.sh) → local helper
- [`run_gcp.sh`](/Users/aidevmac/Projects/Ephapsys/Product/ephapsys-sdk/samples/agents/helloworld/run_gcp.sh) → GCP helper with reuse, fallback search, bootstrap, and interactive attach
- [`reattach_gcp.sh`](/Users/aidevmac/Projects/Ephapsys/Product/ephapsys-sdk/samples/agents/helloworld/reattach_gcp.sh) → reconnect helper for the current VM
- [`check_gcp.sh`](/Users/aidevmac/Projects/Ephapsys/Product/ephapsys-sdk/samples/agents/helloworld/check_gcp.sh) → preflight and project-access validation
- [`GCP.md`](/Users/aidevmac/Projects/Ephapsys/Product/ephapsys-sdk/samples/agents/helloworld/GCP.md) → focused GCP notes
- [`.env.example`](/Users/aidevmac/Projects/Ephapsys/Product/ephapsys-sdk/samples/agents/helloworld/.env.example) → tracked local env template
- [`.env.gcp.example`](/Users/aidevmac/Projects/Ephapsys/Product/ephapsys-sdk/samples/agents/helloworld/.env.gcp.example) → tracked GCP env template
