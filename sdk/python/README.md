# Ephapsys SDK

Lightweight SDK for **EC-ANN modulation**, **trusted agent provisioning**, and **runtime security**.


---

## 📦 Installation

```bash
pip install ephapsys
```

Optional feature groups:
```bash
pip install "ephapsys[modulation]"
pip install "ephapsys[audio]"
pip install "ephapsys[eval]"
pip install "ephapsys[vision]"      # alias: [video]
pip install "ephapsys[all]"
```

TPM personalization prerequisites (Linux):
```bash
pip install "ephapsys[tpm]"
sudo apt-get install -y tpm2-tools
```
If you are on Ubuntu 22.04, ensure `tpm2-tools`, `tpm2-tss`, and `tpm2-pytss` are version-compatible.
Ubuntu 22.04 commonly ships TSS2 3.x, while newer `tpm2-pytss` builds may expect TSS2 4.x.

Choose the profile by workload:

| Workload | Install command |
|---|---|
| Lightweight orchestrator/proxy only | `pip install ephapsys` |
| Agent runtime (HelloWorld language) | `pip install "ephapsys[modulation]"` |
| Agent runtime (Robot multimodal) | `pip install "ephapsys[modulation,audio,vision,embedding]"` + `pip install webrtcvad sounddevice pyaudio` |
| Agent runtime (GGUF / llama.cpp edge CPU) | `pip install ephapsys` + install `llama-cpp-python` or `llama-cli` |
| Modulators/training scripts | `pip install "ephapsys[modulation]"` |
| Modulators with full evaluation/report stack | `pip install "ephapsys[all]"` |

---

## 🚀 Quickstart

```python
from ephapsys import TrustedAgent

agent = TrustedAgent.from_env()

ok, report = agent.verify()
if not ok:
    raise RuntimeError(f"Agent blocked: {report}")

agent.prepare_runtime()
print(agent.run("Hello world", model_kind="language"))
```

---

## ⚙️ Environment Variables

| Variable              | Description                                         |
|-----------------------|-----------------------------------------------------|
| `EPHAPSYS_AGENT_ID`   | Agent ID/label assigned by AOC                      |
| `AOC_BASE_URL`        | API endpoint, e.g. `http://localhost:8000`          |
| `AOC_ORG_ID`          | Org identifier (non-secret tenant scope)            |
| `AOC_PROVISIONING_TOKEN` | Provisioning credential exchanged for short-lived device token |
| `EPHAPSYS_STORAGE_DIR`| Optional, defaults to `.ephapsys_state`             |

For edge production, use hardware anchors (`tpm`, `tee`, `dsim`, `hsm`) and avoid `PERSONALIZE_ANCHOR=none`.

Runtime download tuning (optional):
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

---

## 🎛️ ModulatorClient

Start / iterate / complete modulation on **model templates**:

```python
from ephapsys.modulation import ModulatorClient

mod = ModulatorClient(api_base="http://localhost:7001", api_key="dev")
resp = mod.start_job(
    model_template_id="google/gemma-2b",
    variant="ec-ann",
    search_space={"lr": [1e-3, 1e-4]},
    kpi={"accuracy": "max"},
    mode="auto"
)
print(resp)
```

---

## 🖥️ CLI

The SDK includes a CLI (`ephapsys`) for working with agents, models, modulation, and certificates.  
Authentication is required before most commands.

### 🔑 Login

```bash
ephapsys login --username izzo
Password: ****
✅ Logged in.
```

This stores a JWT in `~/.ephapsys_state/session.json`.

---

### 📦 Models

Register, list, and remove models tied to your org.

```bash
# Register
ephapsys model register --provider huggingface --ids google/gemma-2b
ephapsys model register --provider huggingface --ids google/embeddinggemma-300m  google/flan-t5-base
ephapsys model register --provider huggingface --ids microsoft/speecht5_tts
ephapsys model register --provider huggingface --ids google/gemma-2b google/embeddinggemma-300m  microsoft/speecht5_tts

# List (pretty table)
ephapsys model list
name                 provider     status
-----------------------------------------
google/gemma-2b      huggingface  registered
google/embedding...  huggingface  registered

# List (JSON)
ephapsys model list --json

# Remove
ephapsys model remove --provider huggingface --id google/gemma-2b
```

---

### 🤖 Agents

```bash
# List (pretty table)
ephapsys agent list
agent_id     label        status
--------------------------------
agent-123    Sales Bot    registered
agent-456    SupportBot   enabled

# List (JSON)
ephapsys agent list --json
```

Other agent commands (`verify`, `enable`, `disable`, `revoke`, `export-manifest`) are also available.

---

### 📊 Modulation

```bash
# Start job
ephapsys mod start --model-template-id google/gemma-2b --variant ec-ann   --search-space '{"lr":[1e-3,1e-4]}' --kpi '{"accuracy":"max"}'

# Report metrics
ephapsys mod metrics --job-id JOB123 --metrics '[{"step":1,"val":0.84}]'

# Request next step
ephapsys mod next --job-id JOB123 --last-metrics '[{"step":1,"val":0.84}]'

# Complete job
ephapsys mod complete --job-id JOB123 --artifacts '{"weights":"s3://..."}'
```


---

## 🧪 Samples

Access the samples at: [https://github.com/ephapsys](https://github.com/ephapsys)
