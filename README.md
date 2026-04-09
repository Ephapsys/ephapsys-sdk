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
| Agent runtime (multimodal) | `pip install "ephapsys[modulation,audio,vision,embedding]"` |
| Agent runtime (GGUF / llama.cpp edge CPU) | `pip install ephapsys` + install `llama-cpp-python` or `llama-cli` |
| Modulators/training scripts | `pip install ephapsys` |
| Modulators with full evaluation/report stack | `pip install "ephapsys[all]"` |

## Quickstart (Python)

### Modulation (pre-deployment)

Use `ModulatorClient` to fine-tune models with ephaptic coupling before deployment.

```python
from ephapsys.modulation import ModulatorClient

mod = ModulatorClient.from_env()

# Start a modulation job with EC-ANN variant
job = mod.start_job(
    model_template_id="google/gemma-2b",
    variant="ec-ann",
    search={"lr": [1e-4, 5e-5]},
    kpi={"accuracy": "max"}
)

# Evaluate and report metrics
mod.evaluate_and_report(
    job_id=job["job_id"],
    model_path="./out/modulated",
    metrics=["mmlu"],
    stage="modulated"
)
```

### Runtime (deployment)

Use `TrustedAgent` to deploy and run agents with fail-closed governance.

```python
from ephapsys import TrustedAgent

agent = TrustedAgent.from_env()
agent.verify()                          # fail-closed: certs, digests, keys
agent.personalize(anchor="tpm")         # bind to hardware (or "none" for dev)
agent.prepare_runtime()                 # download and verify modulated models

with agent.session(lease_seconds=1800):
    result = agent.run("What is ephaptic coupling?", model_kind="language")
    print(result)
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

Samples have moved to a dedicated repo: [ephapsys-samples](https://github.com/Ephapsys/ephapsys-samples).

- See `agents/helloworld` for a minimal chatbot agent.


Fastest local HelloWorld path:

```bash
git clone https://github.com/Ephapsys/ephapsys-samples.git
cd ephapsys-samples/agents/helloworld
./quickstart.sh
```

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
