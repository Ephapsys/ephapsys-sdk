# What is Ephapsys SDK?

Ephapsys SDK helps you deploy trustworthy AI agents enhanced and secured using ephaptic coupling with built-in cryptography at the neural level. Simply put, think of it as the built-in performance boost and kill switch for your AI Agents.

With the Ephapsys SDK, you can:
- Cryptographically seal your models to guarantee provenance and integrity during fine-tuning. 
- Leverage ephaptic coupling during and after fine-tuning to boost your model performance.
- Deploy, monitor, audit, and instantly override agent behaviors via your Agents Ops Center.

## Prerequisites
- Python 3.9+
- pip 23+ (recommended)
- Supported OS: Linux (full TPM/TEE flows), macOS (CLI-only mode; TPM-disabled), Windows WSL2 (CLI-only)
- Optional: `build` + `twine` if you publish wheels; `docker` if you use the samples’ compose files.
- Registered [Ephapsys](https://ephapsys.com) organization account

## Installation

Latest release from PyPI:
```bash
pip install ephapsys
```

## Quickstart (Python)
```python
from ephapsys import TrustedAgent, ModulatorClient

agent = TrustedAgent(
    agent_id="agent_demo",
    api_base="https://api.ephapsys.com",
    api_key="YOUR_API_KEY",
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
# Authenticate
ephapsys login

# Upload and list models
ephapsys models upload ./model.pt --label "Demo Model" --type language --version 1.0
ephapsys models list

# Start / stream / finish a modulation run
ephapsys modulation start --model-id model:demo --variant multiplicative --epsilon 1.0 --lambda0 0.05
ephapsys modulation stream --model-id model:demo
ephapsys modulation finish --model-id model:demo

# Assemble and verify an agent
ephapsys agents create --label "Assistant Suite" --models model:demo
ephapsys verify --agent-id agent_demo
```

## Configuration
The SDK/CLI read environment variables or a `.env` file:
```
AOC_API_BASE=https://api.ephapsys.com   # backend base URL
AOC_API_KEY=YOUR_API_KEY               # org API key
EPHAPSYS_AGENT_ID=agent_demo           # agent identity for TrustedAgent.from_env
PERSONALIZE_ANCHOR=none                # or tpm depending on hardware anchors
```

## Samples
- See `samples/agents/helloworld` for a minimal chatbot agent.
- See `samples/agents/robot` for a full multi-modal reference agent.

## Contributions, Support and Security
- **Contributions:** Open issues/PRs with clear repro steps, expected vs. actual results, and environment (OS, Python, SDK version). Run lint/tests (`python -m pip install -r requirements-dev.txt && pytest`) before submitting.
- **Support:** For usage help, file a “question” issue or email support@ephapsys.com with logs, SDK version, and API base URL. We aim to respond within 1 business day.
- **Security:** Report vulnerabilities privately to security@ephapsys.com (optionally PGP). Include affected versions, impact, and minimal reproduction steps. We follow coordinated disclosure and confirm receipt within 24 hours.
