This folder contains the following:

- **agents:**  sample code of various agents using Ephapsys SDK (TrustedAgent class)
- **modulators:**  sample code for modulating various models using Ephapsys SDK (ModulatorClient class)

## Install profiles for samples

Use the matching install command before running each sample:

| Sample type | Recommended install |
|---|---|
| `agents/helloworld` | `pip install "ephapsys[modulation]"` |
| `agents/robot` | `pip install "ephapsys[modulation,audio,vision,embedding]"` + `pip install webrtcvad sounddevice pyaudio` |
| `modulators/*` (training/modulation only) | `pip install "ephapsys[modulation]"` |
| `modulators/*` (full eval/report stack) | `pip install "ephapsys[all]"` |

For `agents/helloworld`, the local wrapper can bootstrap a fresh checkout for you:

```bash
cd samples/agents/helloworld
cp .env.example .env
./run_local.sh check
./run_local.sh
```

The script creates `.venv`, installs the local SDK with `modulation` extras if needed, picks a sensible default personalization anchor for local development, and supports `./run_local.sh check` for backend preflight before startup.

## Continuous sample testing

Sample automation is defined in `samples/ci/run_samples_ci.sh` with two tiers:

- `smoke`: fast script/env/syntax validation (no live backend required)
- `integration`: bounded HelloWorld one-shot run against a real backend + smoke for robot/modulator wrappers

GitHub Actions workflow: `.github/workflows/samples-ci.yml`

- `smoke` runs on:
  - every `pull_request`
  - every push to `main`/`master`
- `integration` runs on:
  - manual trigger (`workflow_dispatch`)
  - daily schedule
  - only when required secrets are configured
  - robot mock integration executes when `ROBOT_AGENT_TEMPLATE_ID` secret is set

Run locally:

```bash
bash samples/ci/run_samples_ci.sh smoke
```

```bash
# Requires real backend credentials in env.
bash samples/ci/run_samples_ci.sh integration
```

Maintainer reference:
- `docs/MAINTAINERS_CI.md`
