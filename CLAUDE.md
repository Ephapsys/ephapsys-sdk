# CLAUDE.md — ephapsys-sdk

**PUBLIC REPOSITORY** — This SDK is published to PyPI as `pip install ephapsys`. This file and any changes to this repo may be visible to the public. Keep content appropriate for a public audience.

---

## What Is This?

The Ephapsys Python SDK — developer tools for building, provisioning, and running trusted AI agents with ephaptic coupling.

---

## Repository Structure

```
ephapsys-sdk/
├── sdk/python/         ← Core SDK library
├── scripts/            ← Build, test, publish automation
├── tests/              ← SDK test suite
├── LICENSE             ← Apache 2.0
└── README.md
```

> **Samples have moved** to [ephapsys-samples](https://github.com/Ephapsys/ephapsys-samples) (agents + modulators).

---

## Core SDK Classes

### TrustedAgent
Verify identity, bind to hardware, run inference.

```python
from ephapsys import TrustedAgent

agent = TrustedAgent.from_env()
agent.verify()                          # fail-closed: certs, digests, keys
agent.personalize(anchor="tpm")         # bind to TPM (or "none" for dev)
agent.prepare_runtime()                 # download & decrypt models
with agent.session(lease_seconds=1800):
    result = agent.run("query", model_kind="language")
```

### ModulatorClient
Run modulation jobs and track KPIs.

```python
from ephapsys.modulation import ModulatorClient

mod = ModulatorClient.from_env()
job = mod.start_job(
    model_template_id="gpt2",
    variant="ec-ann",
    search={"lr": [1e-4, 5e-5]},
    kpi={"accuracy": "max"}
)
mod.evaluate_and_report(job_id, model_path, metrics=["mmlu"], stage="modulated")
```

---

## CLI

```bash
ephapsys login --username <user>
ephapsys model register --provider huggingface --ids google/gemma-2b
ephapsys agent create-template --label "Suite" --models '[...]'
ephapsys agent verify --agent-id <id>
ephapsys tune start --model-template-id <id> --variant ec-ann --search '{"lr":[1e-4]}'
```

---

## Installation

```bash
pip install ephapsys                          # default
pip install "ephapsys[audio]"                 # + audio models
pip install "ephapsys[vision]"                # + vision models
pip install "ephapsys[eval]"                  # + evaluation stack
pip install "ephapsys[all]"                   # everything
pip install "ephapsys[tpm]"                   # + TPM support (Linux)
```

---

## Hardware Anchors

| Anchor | Platform | Notes |
|--------|---------|-------|
| `tpm` | Linux | TPM 2.0 via tpm2-pytss |
| `tee` | ARM | TrustZone / TEE |
| `dsim` | Cellular | UICC-based |
| `hsm` | Enterprise | Hardware Security Module |
| `none` | Dev/test | Skip hardware binding |

Set via `PERSONALIZE_ANCHOR` env var or `agent.personalize(anchor=...)`.

---

## Development

### Setup
```bash
cd sdk/python
pip install -r requirements-dev.txt
```

### Tests
```bash
pytest tests/
```

### Build & Publish
```bash
# Build
./scripts/build.sh

# Publish to TestPyPI (staging)
./scripts/publish_testpypi.sh

# Publish to PyPI (production)
./scripts/publish_pypi.sh
```

### Samples

Samples have moved to [ephapsys-samples](https://github.com/Ephapsys/ephapsys-samples).

```bash
# Clone separately:
git clone https://github.com/Ephapsys/ephapsys-samples.git
cd ephapsys-samples/agents/helloworld
cp .env.example .env   # configure AOC URL, agent credentials
./run.sh
```

---

## License

Apache 2.0 — permissive with patent protection clause. See `LICENSE`.

---

## Notes for Contributors

- Python 3.9+ compatibility required
- Type hints required for all public APIs
- Keep SDK dependencies minimal (torch, transformers, cryptography, requests, click)
- Hardware-specific deps (tpm2-pytss) go in optional feature groups
- Do not add internal platform implementation details to this repo

---

*Last updated: 2026-04-09*
