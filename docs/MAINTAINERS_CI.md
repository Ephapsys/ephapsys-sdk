# Maintainers CI Runbook (Samples)

This runbook is for SDK maintainers managing sample automation.

## Workflow

- File: `.github/workflows/samples-ci.yml`
- Jobs:
  - `smoke` (always on PR/push)
  - `integration` (manual + scheduled, requires secrets)

## Trigger Policy

- `smoke` runs on:
  - `pull_request`
  - push to `main` / `master`
- `integration` runs on:
  - `workflow_dispatch`
  - daily cron

## Required Secrets (Integration)

- `AOC_BASE_URL`
- `AOC_ORG_ID`
- `AOC_PROVISIONING_TOKEN`
- `AGENT_TEMPLATE_ID`

Optional:
- `ROBOT_AGENT_TEMPLATE_ID` (enables robot mock integration path)
- `PERSONALIZE_ANCHOR` (defaults safely when not set)

## Local Repro

```bash
# Fast checks (no live backend)
bash samples/ci/run_samples_ci.sh smoke

# Live checks (requires real env vars)
bash samples/ci/run_samples_ci.sh integration
```

## Sample CI Modes

- HelloWorld:
  - `run_local.sh smoke`
  - `run_local.sh oneshot` (non-interactive CI mode)
- Robot:
  - `run_local.sh smoke`
  - `run_ci_mock.sh` (hardware-optional integration path)
- Language modulator:
  - `modulate_local.sh smoke`

## Maintenance Notes

- Keep sample scripts backward-compatible for local dev.
- Any new sample should add:
  - a smoke-safe script mode
  - optional bounded integration mode
  - entry in `samples/ci/run_samples_ci.sh`
