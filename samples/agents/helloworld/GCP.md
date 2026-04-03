# HelloWorld GCP Setup

HelloWorld GCP mode runs the sample on a persistent VM and is designed to reuse that VM across developer runs.

## Files

- [`.env`](/Users/aidevmac/Projects/Ephapsys/Product/ephapsys-sdk/samples/agents/helloworld/.env)
  - runtime credentials uploaded to the VM on each run
- [`.env.gcp`](/Users/aidevmac/Projects/Ephapsys/Product/ephapsys-sdk/samples/agents/helloworld/.env.gcp)
  - local GCP deployment settings
- [`.env.gcp.example`](/Users/aidevmac/Projects/Ephapsys/Product/ephapsys-sdk/samples/agents/helloworld/.env.gcp.example)
  - tracked template
- [`.last_gcp_instance`](/Users/aidevmac/Projects/Ephapsys/Product/ephapsys-sdk/samples/agents/helloworld/.last_gcp_instance)
  - local metadata describing the last reusable VM

## Recommended Flow

First successful GPU run:

```bash
cd ephapsys-sdk/samples/agents/helloworld
cp .env.example .env
cp .env.gcp.example .env.gcp
./quickstart.sh --gcp
```

Normal repeat run after that:

```bash
./quickstart.sh --gcp
```

The second command should usually:
- reuse the existing VM
- upload the current `.env`
- skip full bootstrap when `.venv` already exists remotely
- open interactive chat

## Required `.env`

Fill in:
- `AOC_BASE_URL`
- `AOC_ORG_ID`
- `AOC_PROVISIONING_TOKEN`
- `AGENT_TEMPLATE_ID`

Optional for template bootstrap only:
- `AOC_MODULATION_TOKEN`

Important:
- if `AOC_PROVISIONING_TOKEN` is rotated or replaced, just update local `.env` and rerun `./quickstart.sh --gcp`
- the script uploads the fresh token to the VM automatically

## Required `.env.gcp`

Set:
- `PROJECT_ID`
- `ZONE`
- `MACHINE_TYPE`
- `DISK_SIZE`
- `IMAGE_FAMILY`
- `IMAGE_PROJECT`
- `INSTANCE_PREFIX`

## Recommended GPU Search Settings

These make the developer experience materially better when capacity is tight:

```ini
USE_GPU=1
GPU_TYPE=t4
GPU_FALLBACKS=t4,l4,a100
ZONE_FALLBACKS=us-central1-a,us-central1-b,us-central1-c,us-central1-f
REGION_FALLBACKS=us-central1,us-east1,us-west1
INTERACTIVE=1
```

Behavior:
- the script tries the configured zone first
- then explicit fallback zones
- then synthesized zones from fallback regions
- then GPU fallbacks in order
- logs each attempt clearly
- prints a green `[SELECTED]` line once a VM is actually chosen

## Runtime Behavior

On a fresh VM, `run_gcp.sh` will:
1. create or select a VM
2. wait for SSH readiness
3. copy the needed sample files
4. upload `.env`
5. provision apt packages, Python, venv, and SDK deps if needed
6. open interactive chat by default

On a reused prepared VM, it should only:
1. reuse `.last_gcp_instance`
2. verify the VM still exists
3. upload refreshed files and `.env`
4. skip bootstrap
5. attach directly

## Useful Commands

Preflight:

```bash
./check_gcp.sh
```

Run with reuse behavior:

```bash
./quickstart.sh --gcp
```

Run directly without quickstart:

```bash
./run.sh --gcp
```

Force no interactive attach:

```bash
./run.sh --gcp --no-interactive
```

Force a fresh VM:

```bash
./run.sh --gcp --fresh-instance
```

Reconnect to the current VM:

```bash
./reattach_gcp.sh
```

Stop the VM to save cost while preserving disk state:

```bash
gcloud compute instances stop <instance> --project <project> --zone <zone>
```

Delete the VM only if you want to discard the prepared runtime:

```bash
gcloud compute instances delete <instance> --project <project> --zone <zone>
```

## Operational Notes

- `gcloud compute ssh` requires `roles/compute.instanceAdmin.v1`
- GPU quota approval does not guarantee zonal capacity
- `ZONE_RESOURCE_POOL_EXHAUSTED` is a capacity problem, not necessarily a quota problem
- fresh GPU images may need `100GB` boot disks; the script clamps too-small values up automatically
- the current developer loop should prefer **reuse and stop**, not **delete and reprovision**
