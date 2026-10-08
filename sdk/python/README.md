# Ephapsys SDK

Lightweight SDK for **EC-ANN modulation**, **trusted agent provisioning**, and **runtime security**.


---

## 📦 Installation

```bash
pip install ephapsys
```

Optional feature groups:
```bash
pip install ephapsys                    # default runtime + language/modulation stack
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
| Agent runtime (HelloWorld language) | `pip install ephapsys` |
| Agent runtime (Robot multimodal) | `pip install "ephapsys[modulation,audio,vision,embedding]"` + `pip install webrtcvad sounddevice pyaudio` |
| Agent runtime (GGUF / llama.cpp edge CPU) | `pip install ephapsys` + install `llama-cpp-python` or `llama-cli` |
| Modulators/training scripts | `pip install ephapsys` |
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
| `AOC_BASE_URL`        | API endpoint, e.g. `https://api.ephapsys.com`       |
| `AOC_ORG_ID`          | Org identifier (non-secret tenant scope)            |
| `AOC_PROVISIONING_TOKEN` | Provisioning credential exchanged for short-lived device token |
| `EPHAPSYS_STORAGE_DIR`| Optional, defaults to `.ephapsys_state`             |

For edge production, use a hardware anchor (`tpm` or `hsm`) and avoid `PERSONALIZE_ANCHOR=none`.
(`tee` and `dsim` are not yet enabled.)

### PKCS#11 tokens (`hsm` anchor)

Any PKCS#11 token (TEE-backed tokens, HSMs, USB tokens) can back the `hsm` anchor. Install the extra and configure the token; only configuration is device-specific:

```bash
pip install "ephapsys[pkcs11]"

PERSONALIZE_ANCHOR=hsm
PKCS11_MODULE=/path/to/pkcs11-module.so    # the token vendor's PKCS#11 library
PKCS11_TOKEN_LABEL=my-token                 # or PKCS11_TOKEN_SERIAL (slot index is not accepted)
PKCS11_PIN_FILE=/secure/path/pin            # or PKCS11_PIN
PKCS11_SIGN_KEY_ID=01                       # or PKCS11_SIGN_KEY_LABEL: EC P-256 or RSA-2048+
PKCS11_KEM_KEY_ID=02                        # or PKCS11_KEM_KEY_LABEL: EC P-256 with derive permission
EPHAPSYS_DEVICE_ID=device-0001              # stable device identity
```

- Two separate, sensitive, non-extractable token keys: a **sign** key for personalization evidence and device authentication, and a **KEM** key (ECDH) that receives the model key and protects the encrypted-at-rest cache.
- Keys are enrolled with the AOC either by an administrator, or automatically on the device's first personalization when the organization enables first-use enrollment. `ephapsys hsm show-key` prints the public keys and SPKI SHA-256 fingerprints.
- Selection is fail-closed: ambiguous provider configuration (`PKCS11_MODULE` together with `HSM_HELPER`/`HSM_KMS_KEY`/`HSM_EVIDENCE_PATH`), missing or duplicate keys, extractable keys and unsupported mechanisms are all rejected.
- Scope: token-backed key custody. Whether a token is hardware-backed depends on the deployed provider; SoftHSM is for testing only. PKCS#11 provides no measured-boot attestation, and decrypted model material is held in process memory while in use.

### Google Cloud KMS / Cloud HSM (`hsm` anchor, cloud workloads)

Agents running on Google Cloud (for example on GKE) can anchor in two Cloud HSM keys instead of a device token. Requires SDK >= 0.3.1:

```bash
pip install "ephapsys[hsm]"

PERSONALIZE_ANCHOR=hsm
HSM_KMS_KEY=projects/P/locations/L/keyRings/R/cryptoKeys/my-sign        # EC_SIGN_P256_SHA256, protection level HSM
HSM_KMS_DECRYPT_KEY=projects/P/locations/L/keyRings/R/cryptoKeys/my-decrypt  # RSA_DECRYPT_OAEP_3072_SHA256, HSM
EPHAPSYS_DEVICE_ID=my-agent-1               # unique and stable per replica
AOC_ORG_ID=...
AOC_PROVISIONING_TOKEN=...                  # needed only for each device's first personalization
```

- **Identity:** the workload runs as a dedicated Google service account (on GKE through Workload Identity). The SDK obtains a Google-signed ID token for it from the metadata server, with the AOC as audience (override with `EPHAPSYS_WORKLOAD_AUDIENCE`). No key files are needed.
- **Organization setup:** once per organization, an administrator records in the AOC which service accounts the organization trusts for which agent templates and key names. Nothing is done per device after that.
- **Personalization:** two steps. The SDK sends a transcript signed in Cloud HSM together with both keys' Cloud HSM attestations, then answers an encrypted challenge with the decrypt key. The ECM content key is wrapped to the decrypt key (RSA-OAEP).
- **Lifecycle:** before `prepare_runtime()` the SDK reconciles its enrollment with the configured keys (`reconcile_gcp_kms()`).
  - New key versions (a CryptoKey name resolves to its newest enabled version) make it rotate, co-signed by the previous key.
  - A previous key that can no longer be used (disabled, destroyed or not found) makes it complete a recovery that your deployment automation authorized.
  - A revoked enrollment fails closed.
  - Device tokens are renewed automatically.
- **Integrity:** every Cloud KMS call is integrity-checked (CRC32C) and must report protection level `HSM`. Long-term private keys never leave Cloud HSM; a decrypted content key is held in process memory while in use.

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

mod = ModulatorClient(api_base="https://api.ephapsys.com", api_key="dev")
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

## 🔗 A2AClient (Agent-to-Agent)

Send, list, and ack org-scoped A2A messages:

```python
from ephapsys import A2AClient

a2a = A2AClient.from_env()

sent = a2a.send_message(
    from_agent_id="agent_sender",
    to_agent_id="agent_receiver",
    payload={"op": "ping"},
    message_type="event",
)

inbox = a2a.inbox(agent_id="agent_receiver", limit=20)
first = (inbox.get("items") or [None])[0]
if first:
    a2a.ack_message(message_id=first["id"], agent_id="agent_receiver")
```

`A2AClient.from_env()` uses `AOC_A2A_TOKEN` (and falls back to `AOC_MODULATION_TOKEN` for compatibility).

Optional signed mode (recommended for production):

```bash
export AOC_A2A_TOKEN=a2a_xxx
export AOC_ORG_ID=org_xxx
export A2A_SIGN_REQUESTS=1
export A2A_HMAC_SECRET=replace_with_org_secret
```

When `A2A_SIGN_REQUESTS=1`, `A2AClient.send_message()` adds:
- `x-a2a-ts`
- `x-a2a-nonce`
- `x-a2a-sig`

Backend verification controls:
- `A2A_REQUIRE_SIGNATURE=1`
- `A2A_HMAC_SECRET`
- `A2A_REPLAY_WINDOW_SECONDS` (default `300`)

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

Use `--base-url` when you want a non-production environment:

```bash
ephapsys --base-url https://api.staging.ephapsys.ai login
```

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
ephapsys tune start --model-template-id google/gemma-2b --variant ec-ann   --search-space '{"lr":[1e-3,1e-4]}' --kpi '{"accuracy":"max"}'

# Report metrics
ephapsys tune metrics --job-id JOB123 --metrics '[{"step":1,"val":0.84}]'

# Request next step
ephapsys tune next --job-id JOB123 --last-metrics '[{"step":1,"val":0.84}]'

# Complete job
ephapsys tune complete --job-id JOB123 --artifacts '{"weights":"s3://..."}'
```


---

## 🧪 Samples

Access the samples at: [https://github.com/ephapsys](https://github.com/ephapsys)
