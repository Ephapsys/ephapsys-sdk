# Ephapsys Python SDK & CLI

Empower and govern AI Agents with ephaptic‑coupling. This SDK is the **security control plane** around models and agents; the CLI mirrors the same surface for ops and CI.

---

## Highlights

- **TrustedAgent**: fail‑closed verification (certs, digests, ECM presence, host binding, leases) + lifecycle control.
- **ModulatorClient**: start/stream/finish modulation jobs; attach artifacts, ECM digests, RMS signatures.
- **ECM utilities**: initialize Λ, persist to disk, compute stable digests.
- **Batteries‑included CLI**: models, modulation, agents, certs, guard checks.
- **Mock backend & EJBCA‑ready**: develop locally, flip to real PKI later.

---

## Requirements

- Python 3.9+
- Optional: Git, make, Docker (if you use the provided compose)
- `google-cloud-kms>=2.21.0` ships with the SDK to enable the built-in Cloud KMS / Cloud HSM flow. It is a no-op unless you set `PERSONALIZE_ANCHOR=hsm` together with `HSM_KMS_KEY`; other HSM vendors should rely on the helper interface described in `Product/DOCS/HSM.md`.
- API key from Ephapsys Platform (https://ephapsys.com)

---

## Installation

### 1) Local development (editable install: Recommended if you are actively working on the DK)

```bash
cd SDK/python
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install build
pip install -U pip wheel build
pip install -e .
```

### 2) Local wheel install (no edit: Recommonded if you're not actively working on the SDK)

```bash
sh update_sdh.locally.sh # Shell script to simplify our lives
```


**Note:**

- On macOS, the SDK installs in CLI-only mode. TPM-backed functions are disabled.
- On Linux, full TPM support is available (requires tpm2-tss >= 2.4.0).
- On most cloud VMs (GCE, AWS, Azure), there is no hardware TPM exposed by default. If you need `PERSONALIZE_ANCHOR=tpm`, provision a Shielded/vTPM-enabled instance or switch to a software anchor (`PERSONALIZE_ANCHOR=none`) for quick demos. Installing `tpm2-tools` alone is not enough without a TPM device behind `/dev/tpmrm0`.
- For centralized custody, set `PERSONALIZE_ANCHOR=hsm` and supply `HSM_KMS_KEY` (plus optional `HSM_KMS_ENDPOINT` / `HSM_KMS_CREDENTIALS`). The SDK uses `google-cloud-kms` to talk to Cloud KMS / Cloud HSM, and the HelloWorld `run_gcp.sh` script will copy any referenced credentials into `~/helloworld/.secrets/` automatically.

For more on vTPM:
-  https://cloud.google.com/blog/products/identity-security/virtual-trusted-platform-module-for-shielded-vms-security-in-plaintext
- https://docs.cloud.google.com/compute/shielded-vm/docs/quickstart

### 3) TestPyPI / PyPI (production)

```bash
# build
cd SDK/python
python -m build

# upload to TestPyPI
python -m pip install twine
twine upload --repository testpypi dist/*

# verify
pip install -i https://test.pypi.org/simple/ ephapsys

# upload to PyPI (when ready)
twine upload dist/*
pip install ephapsys
```

> Tip: keep `pyproject.toml` version in sync with your release tags.

### 4) Helper Script

At the repo root run:

```bash
Product/SCRIPTS/publish-sdk.sh --dev        # rebuild + reinstall locally
Product/SCRIPTS/publish-sdk.sh --staging    # build + upload to TestPyPI
Product/SCRIPTS/publish-sdk.sh --production # build + upload to PyPI (prompts before pushing)
```

Set `PUBLISH_FORCE=1` to skip the production confirmation prompt.

**Prerequisites for `--staging` and `--production`**
- Create accounts on both [TestPyPI](https://test.pypi.org/account/register/) and [PyPI](https://pypi.org/account/register/) and generate API tokens (reference the official [TestPyPI guide](https://packaging.python.org/en/latest/guides/using-testpypi/)).
- Store the credentials where Twine can read them (recommended: `~/.pypirc`, alternatively export `TWINE_USERNAME`/`TWINE_PASSWORD` or `TWINE_API_KEY`). The script now checks for these entries and, if missing, prompts for the values and writes them to `~/.pypirc` for you (opening the relevant signup/docs pages when needed).
- Ensure the version in `SDK/python/pyproject.toml` is bumped before publishing; PyPI/TestPyPI reject duplicate version numbers.
- After publishing, update `NEXT_PUBLIC_SDK_VERSION` in `Product/AOC/frontend/.env.local*` so the marketing site reflects the latest version badge (the publish script prints a reminder with the current version).
---

## Configuration (Environment)

The SDK and CLI read a single `.env` (or process env vars):

```ini
# AOC backend base URL (aka EPM_URL in samples)
AOC_API_BASE=http://localhost:8000
# API key for your org (mock accepts "dev")
AOC_API_KEY=dev

# Agent identity (used by TrustedAgent.from_env)
EPHAPSYS_AGENT_ID=agent_demo

# Optional/advanced
PKI_BACKEND=mock            # or ejbca (in backend)
OFFLINE_ENABLE_LEASE_SEC=0  # override in offline mode
```

Legacy alias: `EPM_URL` is treated as `AOC_API_BASE` if present.

---

## Quickstart (Python)

```python
from ephapsys import TrustedAgent, ModulatorClient, persist_ecm, init_ecm

# Canonical constructor (spec)
agent = TrustedAgent(
    agent_id="agent_demo",
    api_base="http://localhost:8000",
    api_key="dev",
)

# or convenience:
# agent = TrustedAgent.from_env()

# Fail-closed verification (certs, digests, ECM, host binding, leases)
agent.verify()
if agent.is_revoked():
    raise RuntimeError("Agent is revoked")

# Authenticated client for modulation APIs
client = ModulatorClient(base_url="http://localhost:8000", signer=agent.signer)

with agent.session(lease_seconds=1800) as sess:
    # Example: guard check + minimal prompt inference (pseudo)
    if not agent.is_enabled():
        raise RuntimeError("Agent disabled by policy")
    # ... run your model via agent.wrap(model) and your inference code ...
```

---

## Quickstart (CLI)

```bash
# auth
ephapsys login --api-base http://localhost:8000 --api-key dev

# models
ephapsys models upload ./model.pt --label "GPT-2 (Demo)" --type language --version 1.0
ephapsys models list

# modulation
ephapsys modulation start --model-id model:gpt2-demo \
  --variant multiplicative --epsilon 1.0 --lambda0 0.05 --kappa 0.0 --mu 200 --phi identity
ephapsys modulation stream --model-id model:gpt2-demo
ephapsys modulation finish --model-id model:gpt2-demo   # issues model ECM cert

# assemble agent
ephapsys agents create --label "Assistant Suite" --models model:gpt2-demo
ephapsys agents status --agent-id agent_demo
ephapsys agents enable --agent-id agent_demo

# guard
ephapsys verify --agent-id agent_demo
```

---

## Python API Reference

### `TrustedAgent`

```python
TrustedAgent(agent_id: str, api_base: str, api_key: str)
@classmethod TrustedAgent.from_env() -> TrustedAgent
```

**Core methods**

- `verify(agent_id: str | None = None, manifest_path: str | None = None) -> None`  
- `session(lease_seconds: int = 1800)` → context manager  
- `wrap(model)` → `model`  
- `is_enabled() -> bool` / `is_revoked() -> bool`  
- `agents.get(agent_id)` / `agents.list()`  
- `models.list()`  
- `certificates.list(filters)` / `certificates.get(serial)`  

**Lifecycle control**

- `enable(agent_id)` / `disable(agent_id)` / `revoke(agent_id, reason=None)`

**Provisioning**

- `provision(target: dict) -> ProvisioningResult`  

**Model & Modulation helpers**

- `register_model(...) -> str`  
- `upload_model(...) -> str`  
- `start_modulation(model_id, variant, hyperparams, kpi, dataset_path=None) -> str`  
- `stream_metrics(model_id) -> Iterator[MetricEvent]`  
- `finish_modulation(model_id) -> Certificate`  
- `assemble_agent(label, model_ids, policy) -> str`  

---

### `ModulatorClient`

```python
ModulatorClient(base_url: str, signer: Any, timeout_s: int = 15)
```

**Common calls**

- `start_job(template_id | model_id, variant, hyperparams, kpi) -> dict`
- `stream(job_id | model_id) -> Iterable[SSEvent]`
- `report_metrics(job_id, metrics: list[dict]) -> None`
- `complete_job(job_id, artifact_urls: dict, ecm_digest: str, rms_hash: str | None = None) -> dict`
- `fetch_ecm(agent_id: str, prefer="latest") -> Any`

---

### ECM & Digest Utilities

```python
from ephapsys import persist_ecm, init_ecm

digest = persist_ecm(Λ, "ecm.npy")               # → "sha256:<hex>"

Λ = init_ecm("identity", h=hidden_size, scale=lambda0)
Λ = init_ecm("transpose", W=weight_tensor, scale=lambda0)
Λ = init_ecm("random", h=hidden_size, scale=lambda0, seed=42)
Λ = init_ecm("guided_topk", W=weight_tensor, kappa=0.1, scale=lambda0)
```

---

## Publishing & Release (checklist)

1. **Bump version** in `pyproject.toml`.
2. **Pass tests & lint**.
3. **Build**: `python -m build`
4. **Smoke test** the wheel.
5. **Upload** (TestPyPI → PyPI) with `twine upload`.
6. **Tag & release** in VCS.

---
