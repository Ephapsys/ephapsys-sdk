# What is Ephapsys SDK?

Ephapsys SDK helps you deploy trustworthy AI agents enhanced and secured using ephaptic coupling with built-in cryptography at the neural level. Simply put, think of it as the built-in performance boost and kill switch for your AI Agents.

With the Ephapsys SDK, you can:
- Cryptographically seal your models to guarantee provenance and integrity during fine-tuning. 
- Leverage ephaptic coupling during and after fine-tuning to boost your model performance.
- Deploy, monitor, audit, and instantly override agent behaviors via your Agent Ops Center (AOC).

## Prerequisites
- Python 3.9+
- pip 23+ (recommended)
- Supported OS: Linux (full TPM/TEE flows), macOS (CLI-only mode; TPM-disabled), Windows WSL2 (CLI-only)
- Optional: `build` + `twine` if you publish wheels; `docker` if you use the samples’ compose files.
- Org provisioning credentials for your registered [Ephapsys](https://ephapsys.com) organization account

## Installation

Latest release from PyPI:
```bash
pip install ephapsys
```

Optional feature groups:
```bash
pip install ephapsys                    # default runtime + language/modulation stack
pip install "ephapsys[audio]"           # audio I/O/runtime helpers
pip install "ephapsys[eval]"            # evaluation toolchain
pip install "ephapsys[vision]"          # vision/camera stack
pip install "ephapsys[all]"             # full SDK dependency set
```

TPM personalization prerequisites (Linux):
```bash
pip install "ephapsys[tpm]"
sudo apt-get install -y tpm2-tools
```
On Ubuntu 22.04, verify that `tpm2-tools` / `tpm2-tss` and `tpm2-pytss` are compatible.
Ubuntu 22.04 typically provides TSS2 3.x; some `tpm2-pytss` builds require TSS2 4.x.

Choose the profile by workload:

| Workload | Install command |
|---|---|
| Lightweight orchestrator/proxy only | `pip install ephapsys` |
| Agent runtime (HelloWorld language) | `pip install ephapsys` |
| Agent runtime (Robot multimodal) | `pip install "ephapsys[modulation,audio,vision,embedding]"` + `pip install webrtcvad sounddevice pyaudio` |
| Agent runtime (GGUF / llama.cpp edge CPU) | `pip install ephapsys` + install `llama-cpp-python` or `llama-cli` |
| Modulators/training scripts | `pip install ephapsys` |
| Modulators with full evaluation/report stack | `pip install "ephapsys[all]"` |

## Quickstart (Python)
```python
from ephapsys import TrustedAgent
from ephapsys.modulation import ModulatorClient

agent = TrustedAgent(
    agent_id="agent_demo",
    api_base="https://api.ephapsys.com",
)

agent.verify()  # fail-closed checks (certs, digests, leases, host binding)

client = ModulatorClient(base_url=agent.api_base, signer=agent.signer)

with agent.session(lease_seconds=1800):
    # guard, then run your inference or modulation workflow
    if not agent.is_enabled():
        raise RuntimeError("Agent disabled by policy")
    # ... run your model via agent.wrap(model) and your inference code ...
```

## Quickstart (CLI)
The Ephapsys SDK comes with a built-in CLI that makes configuring, managing, and deploying agents simple and fast.

```bash
# Authenticate once; the CLI stores a session in ~/.ephapsys_state/session.json
ephapsys login --username your_username

# Use staging explicitly when needed
ephapsys --base-url https://api.staging.ephapsys.ai login

# Register and list models
ephapsys model register --provider huggingface --ids google/gemma-2b
ephapsys model list

# Create and inspect an agent template
ephapsys agent create-template \
  --label "Assistant Suite" \
  --models '[{"id":"google/gemma-2b","config":{"type":"language","policies":{}}}]'
ephapsys agent list

# Verify and manage an agent
ephapsys agent verify --agent-id agent_demo
ephapsys agent enable --agent-id agent_demo

# Start and complete a modulation job
ephapsys tune start \
  --model-template-id google/gemma-2b \
  --variant ec-ann \
  --search-space '{"lr":[0.001,0.0001]}' \
  --kpi '{"accuracy":"max"}'
ephapsys tune complete --job-id job_123 --artifacts '{"report":"./out/report.json"}'
```

## Configuration
The SDK/CLI read environment variables or a `.env` file:
```
AOC_BASE_URL=https://api.ephapsys.com  # Ops Center base URL
AOC_ORG_ID=org_xxxxx                   # org identifier
AOC_PROVISIONING_TOKEN=boot_xxxxx         # provisioning token
EPHAPSYS_AGENT_ID=agent_demo           # agent identity for TrustedAgent.from_env
PERSONALIZE_ANCHOR=none                # or tpm depending on hardware anchors
```

Optional runtime download tuning:
```bash
AOC_DOWNLOAD_PROGRESS=1
AOC_DOWNLOAD_RETRIES=3
AOC_DOWNLOAD_TIMEOUT=60
AOC_DOWNLOAD_CHUNK_KB=256
AOC_DOWNLOAD_PROGRESS_STEP_MB=5
AOC_DOWNLOAD_WORKERS=4
```

Optional GGUF runtime tuning:
```bash
AOC_LLAMA_CPP_CLI=llama-cli
AOC_GGUF_CTX=2048
AOC_GGUF_MAX_NEW_TOKENS=256
```

## Samples
- See `samples/agents/helloworld` for a minimal chatbot agent.
- See `samples/agents/robot` for a full multi-modal reference agent.

Fastest local HelloWorld path from this repo:

```bash
cd samples/agents/helloworld
./quickstart.sh
```

Fill in `.env` with:
- `AOC_BASE_URL` (`AOC_API_URL` remains a compatibility alias)
- `AOC_ORG_ID`
- `AOC_PROVISIONING_TOKEN`
- `AOC_MODULATION_TOKEN`
- leave `HF_TOKEN` commented out unless you need a gated/private Hugging Face repo
- leave `MODEL_TEMPLATE_ID` and `AGENT_TEMPLATE_ID` blank on first run so `quickstart.sh` can populate them

Notes:
- `quickstart.sh` creates `.env` from `.env.example` if needed, then stops so you can fill in the required values.
- on rerun, `quickstart.sh` prefers existing HelloWorld starter templates first and only falls back to `./push.sh --local` if they are missing.
- `AOC_PROVISIONING_TOKEN` is a secret copied from the AOC UI under Organization -> Tokens. `./push.sh` can write template IDs back into `.env`, but it does not create or refresh provisioning credentials for you.
- `AOC_MODULATION_TOKEN` is also retrieved from Organization -> Tokens and is only required when you run `./push.sh`.
- `run.sh --local` is the public local entrypoint and already runs preflight automatically before launch.
- `run.sh --local` uses the currently installed SDK in your active Python environment. For internal repo development only, set `HELLOWORLD_USE_LOCAL_SDK=1` to install from the local checkout instead.
- `run_local.sh` still exists as the underlying helper, but `run.sh` is the supported entrypoint.
- `push.sh` defaults to idempotent publish for the HelloWorld starter path; use `--no-idempotent` if you explicitly want a full modulation run.
- the default HelloWorld language model is `Qwen/Qwen3.5-0.8B`.
- On macOS and non-TPM machines, the sample defaults to `PERSONALIZE_ANCHOR=none` for a smoother local dev flow.
- On Linux with `tpm2-tools` installed, the sample defaults to `PERSONALIZE_ANCHOR=tpm`.

If you are loading configuration from a local `.env` file in custom scripts, install and use `python-dotenv` explicitly:

```python
from dotenv import load_dotenv
from ephapsys import TrustedAgent

load_dotenv()
agent = TrustedAgent.from_env()
```

## Contributions, Support and Security
- **Contributions:** Open issues/PRs with clear repro steps, expected vs. actual results, and environment (OS, Python, SDK version). Run lint/tests (`python -m pip install -r requirements-dev.txt && pytest`) before submitting.
- **Support:** For usage help, file a “question” issue or email support@ephapsys.com with logs, SDK version, and API base URL. We aim to respond within 1 business day.
- **Security:** Report vulnerabilities privately to security@ephapsys.com (optionally PGP). Include affected versions, impact, and minimal reproduction steps. We follow coordinated disclosure and confirm receipt within 24 hours.

## DISCLAIMER
SUBJECT TO THE EPHAPSYS TERMS OF SERVICE (https://ephapsys.com/terms), THIS SOFTWARE AND DOCUMENTATION ARE PROVIDED “AS IS” WITHOUT WARRANTY OF ANY KIND. USE IS AT YOUR OWN RISK. EPHAPSYS DISCLAIMS ALL IMPLIED WARRANTIES, INCLUDING MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT. IN NO EVENT SHALL EPHAPSYS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OF THIS SOFTWARE OR ITS DOCUMENTATION.
