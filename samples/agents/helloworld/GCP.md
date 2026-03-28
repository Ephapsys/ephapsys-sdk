# HelloWorld GCP Setup

HelloWorld GCP mode deploys the sample to a VM using your local `.env` and `.env.gcp`.

## Files

- `.env`
  - local runtime credentials for the sample
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
   - `./run.sh --gcp`

## Optional Settings

Interactive behavior:
- `INTERACTIVE=1`

GPU:
- `USE_GPU=1`
- `GPU_TYPE`
- `GPU_COUNT`
- `GPU_MACHINE_TYPE`
- `GPU_IMAGE_FAMILY`

Package source:
- `SDK_PACKAGE_SOURCE=pypi|custom`
- `SDK_INDEX_URL`
- `SDK_EXTRA_INDEX_URL`

Alternate runtime env upload:
- `GCP_RUNTIME_ENV_FILE=/absolute/path/to/runtime.env`

HSM / KMS:
- `COMPUTE_SERVICE_ACCOUNT`
- `HSM_KMS_KEY`
- `HSM_KMS_ENDPOINT`
- `HSM_KMS_CREDENTIALS`
- `HSM_KMS_LOCATION`
- `HSM_KMS_KEY_RING`
- `HSM_KMS_KEY_NAME`
- `HSM_KMS_USE_HSM`

## Notes

- `run_gcp.sh` starts the bot in the background on the VM and can open an interactive session afterward.
- `.last_gcp_instance` is local metadata only and is git-ignored.
