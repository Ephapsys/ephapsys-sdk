# Robot GCP Setup

Robot GCP mode deploys only the brain remotely. Your local machine still owns:
- microphone
- camera
- speaker
- terminal face

## Files

- `.env`
  - local runtime credentials for the robot brain
- `.env.gcp`
  - local-only GCP deployment settings
- `.env.gcp.example`
  - tracked template for `.env.gcp`

## Setup

1. Copy `.env.example` to `.env` and fill in:
   - `AOC_BASE_URL`
   - `AOC_ORG_ID`
   - `AOC_PROVISIONING_TOKEN`
   - `AGENT_TEMPLATE_ID`
2. Copy `.env.gcp.example` to `.env.gcp` and fill in:
   - `PROJECT_ID`
   - `ZONE`
   - `MACHINE_TYPE`
   - `DISK_SIZE`
   - `IMAGE_FAMILY`
   - `IMAGE_PROJECT`
   - `INSTANCE_PREFIX`
3. Authenticate gcloud:
   - `gcloud auth login`
4. Run preflight:
   - `./check_gcp.sh`
5. Launch:
   - `./push.sh --gcp`
   - `./run.sh --gcp`

## Optional Settings

GPU:
- `USE_GPU=1`
- `GPU_TYPE`
- `GPU_COUNT`
- `GPU_MACHINE_TYPE`

Package source:
- `SDK_PACKAGE_SOURCE=pypi|custom`
- `SDK_INDEX_URL`
- `SDK_EXTRA_INDEX_URL`

Alternate runtime env upload:
- `GCP_RUNTIME_ENV_FILE=/absolute/path/to/runtime.env`

## Notes

- `run_gcp.sh` creates a VM, starts only the Robot brain remotely, opens an SSH tunnel, then runs the local remote-body client.
- `.last_gcp_instance` is local metadata only and is git-ignored.
