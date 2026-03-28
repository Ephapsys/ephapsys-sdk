# Experimental World Model Modulator

This folder is the starter modulation scaffold for the new `world` model kind.

It is intended for temporal world models such as:

- `facebook/vjepa2-vitl-fpc64-256`

## Current scope

This sample is intentionally **scaffold-only** for now.

It supports:

- creating or referencing a `world` model template in AOC
- starting a modulation job with world-model defaults
- downloading the registered model snapshot locally
- writing summary artifacts into `./artifacts_world/`

It does **not** yet implement the actual temporal evaluation loop, because that
needs a dedicated adapter for:

- frame-window sampling
- video clip preprocessing
- embedding/state extraction
- world-model-specific KPIs

## Create Model Template and Open Modulator

1. Create a Model Template in AOC:
   - Source: External repository
   - Provider: Hugging Face
   - Repository ID: `facebook/vjepa2-vitl-fpc64-256`
   - Model Kind: `world`
   - Revision: `main`
   - Register immediately

2. Use this sample:
   ```bash
   cp .env.example .env
   ```

3. Set:
   - `AOC_BASE_URL`
   - `AOC_MODULATION_TOKEN`
   - `MODEL_TEMPLATE_ID`

4. Run:
   ```bash
   ./modulate.sh
   ```

## Output

The sample writes:

- `summary.json`
- `README.txt`

into a timestamped folder under `./artifacts_world/`.

These artifacts document the registered model snapshot and the current scaffold
state until the full world-model modulation/evaluation loop is implemented.
