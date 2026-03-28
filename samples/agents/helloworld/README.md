# HelloWorld Agent (Sample with Ephapsys SDK)

This sample demonstrates the simplest possible agent using the Ephapsys SDK.  
It verifies, personalizes (if needed), and runs a language model to output **"Hello World"**.

## Fastest Local Path

The default one-command path is now:

```bash
cd ephapsys-sdk/samples/agents/helloworld
./quickstart.sh
```

`quickstart.sh` does:
- `cp .env.example .env` if `.env` is missing, then stops and tells you to fill in the required values first
- checks AOC for existing HelloWorld starter model/agent templates and writes their IDs into `.env` when found
- falls back to `./push.sh --local` only if those starter templates are not available yet
- `./run.sh --local`

For the default HelloWorld flow, the only values you normally need to set in `.env` before `./quickstart.sh` are:
- `AOC_BASE_URL`
- `AOC_ORG_ID`
- `AOC_PROVISIONING_TOKEN`
- `AOC_MODULATION_TOKEN`

Leave `MODEL_TEMPLATE_ID` and `AGENT_TEMPLATE_ID` blank on first run. `./quickstart.sh` will first try to reuse existing HelloWorld starter templates, and only call `./push.sh` if it cannot find them.
Leave `HF_TOKEN` blank unless you switch away from the default public repo (`Qwen/Qwen3.5-0.8B`) to a gated or private model.
`AOC_PROVISIONING_TOKEN` is a secret copied from the AOC UI. `./push.sh` can bootstrap the model and agent template IDs, but it does not create or refresh provisioning credentials.

If you want to bootstrap everything from this sample instead of manually creating a model template, running modulation, and then creating an agent template, use:

```bash
cd ephapsys-sdk/samples/agents/helloworld
cp .env.example .env
./push.sh --local
./run.sh --local
```

For GCP-based modulation instead of local modulation:

```bash
./push.sh --gcp --gpu t4
```

`push.sh` reuses the existing language modulator sample and automates this sequence:
- resolve or register the canonical HelloWorld language model template (`Qwen/Qwen3.5-0.8B` by default)
- idempotently publish it by default, or run full modulation locally or on GCP when requested
- create or reuse a HelloWorld agent template bound to that model
- write `MODEL_TEMPLATE_ID` and `AGENT_TEMPLATE_ID` back into `.env`

By default, `push.sh` now uses idempotent publish mode for the HelloWorld starter path. Use `--no-idempotent` if you explicitly want a full local or GCP modulation run.

If you already have:
- an Ephapsys org ID
- a provisioning token
- an agent template ID wired to a modulated language model

then the shortest path is:

```bash
cd ephapsys-sdk/samples/agents/helloworld
cp .env.example .env
./run.sh --local
```

`run.sh --local` now does the local bootstrap work for you via `run_local.sh`:
- chooses `PERSONALIZE_ANCHOR=none` on macOS or machines without TPM tooling
- chooses `PERSONALIZE_ANCHOR=tpm` on Linux when `tpm2-tools` is available
- runs the backend preflight automatically before launching the sample
- uses the SDK already installed in your active Python environment by default
- supports repo-local SDK development only when you opt in with `HELLOWORLD_USE_LOCAL_SDK=1`

Before the final command, edit `.env` and set:
- `AOC_BASE_URL` (`AOC_API_URL` is still accepted as a compatibility alias)
- `AOC_ORG_ID`
- `AOC_PROVISIONING_TOKEN`
- `AOC_MODULATION_TOKEN` if you plan to use `./push.sh`
- `AGENT_TEMPLATE_ID` only if you are skipping `./push.sh` and already have an existing template

`AOC_PROVISIONING_TOKEN` must come from the AOC UI for the target environment. If it is stale or invalid, `./run.sh --local` will fail preflight even if `./push.sh` succeeded.

> ⚠️ **Important Requirements Before Running**  
> This demo still requires a real Ephapsys organization plus valid tokens.  
> The difference is that `./push.sh` can now bootstrap the default HelloWorld model template and agent template for you.
>
> If you use `./quickstart.sh` or `./push.sh`, you do **not** need to manually create the modulated language model or the agent template first.
> If you skip `./push.sh` and go straight to `./run.sh --local`, then you must already have a valid `AGENT_TEMPLATE_ID` pointing at a published language model.

---

## 🧩 Workflow

1. **Verify agent status**
   - If not yet personalized, run `personalize()` once.
   - If revoked or disabled, refuse to start.

2. **Run a single inference**
   - Send the string `"Hello World"` to the agent.
   - Print the agent's generated response.

---

## ▶️ Run

### Local First-Run Checklist

Use this order for the least friction:

1. Copy `.env.example` to `.env`.
2. Fill in `AOC_BASE_URL`, `AOC_ORG_ID`, `AOC_PROVISIONING_TOKEN`, and `AOC_MODULATION_TOKEN`.
3. Leave `MODEL_TEMPLATE_ID` and `AGENT_TEMPLATE_ID` blank on first run.
4. Run `./quickstart.sh`.
5. If you want to rerun the agent later without rebuilding assets, run `./run.sh --local`.

Keep in mind:
- `AOC_PROVISIONING_TOKEN` is a secret developer credential copied from the AOC UI.
- `./push.sh` writes `MODEL_TEMPLATE_ID` and `AGENT_TEMPLATE_ID` into `.env`, but it does not mint or rotate provisioning tokens.

If startup fails, check these first:
- `404 Agent template not found`: `AGENT_TEMPLATE_ID` is wrong or the template does not exist in that AOC environment.
- `FAIL language_model_missing`: the agent template exists, but it does not reference a language model.
- `FAIL language_model_not_ready` or `FAIL language_model_missing_artifacts`: the linked language model exists, but it is not yet modulated or published correctly.
- Runtime preparation failure: the template exists, but the language model was not modulated or published correctly.
- Personalization failure on a laptop: set `PERSONALIZE_ANCHOR=none` unless you explicitly want TPM/HSM flow.

### Before you run `run.sh --gcp`
Complete the checklist below after publishing the new SDK and redeploying the AOC backend:

1. **Modulated model + agent template** – Ensure the model you want is modulated in AOC and bound to an agent template; capture the template ID for `.env.*`.
2. **API credentials** – Keep your local `.env` and `.env.gcp` files up to date. These files are intentionally not tracked in git.
3. **gcloud access** – Run `gcloud auth login` (or `gcloud auth application-default login`) on the machine invoking `run.sh --gcp` and make sure you can `gcloud compute ssh` into the target project/zone.
4. **Compute IAM** – The user running `gcloud` needs `roles/compute.instanceAdmin.v1`. Use `./check_iam_role.sh --project <project> --member user:<you>` (or grant the role via Cloud Console).
5. **Cloud KMS key (only if `PERSONALIZE_ANCHOR=hsm`)**
   - If you already have a key, set `HSM_KMS_KEY` (and optionally `HSM_KMS_CREDENTIALS`) in `.env.gcp`.
   - Otherwise do nothing—`run.sh --gcp` will automatically call `./provision_kms_key.sh` (using the default Compute Engine service account or the one you specify via `COMPUTE_SERVICE_ACCOUNT`) and capture the resulting `HSM_KMS_KEY` for you.
6. **Vendor HSM (non-GCP)** – Skip the KMS step, set `HSM_HELPER="<your helper command>"`, and export whatever PKCS#11/KMIP env vars the helper requires; confirm it prints valid JSON when invoked with a nonce.
7. **Local `.env` sanity check** – Run `grep -v '^#' .env.gcp` on your local copy and verify no required variable is blank.

### Local

1. Recommended first run: execute `./quickstart.sh`.
2. `quickstart.sh` first looks for existing HelloWorld starter templates in AOC and writes `MODEL_TEMPLATE_ID` / `AGENT_TEMPLATE_ID` into `.env` when found.
3. If the starter templates do not exist yet, `quickstart.sh` falls back to `./push.sh --local`, then writes the resulting IDs into `.env`.
4. It then launches `./run.sh --local`.
5. Ensure the SDK is already installed in your active Python environment, for example with `pip install ephapsys` or the `scripts/use-sdk.sh` helper.
6. For GGUF/llama.cpp CPU runtime, also install either `llama-cpp-python` or a `llama-cli` binary.

Internal repo development only:

```bash
HELLOWORLD_USE_LOCAL_SDK=1 ./run.sh --local
```

That opt-in path creates `.venv` if needed and installs the SDK from the local checkout instead of the published package.

### GCP VM

1. Copy `.env.gcp.example` to `.env.gcp`.
2. Populate local `.env` with the appropriate agent credentials (`AOC_BASE_URL`, `AOC_ORG_ID`, `AOC_PROVISIONING_TOKEN`, `AGENT_TEMPLATE_ID`) and local `.env.gcp` with your GCP deployment settings.
3. Run `./check_gcp.sh`.
4. Run `./run.sh --gcp`. The script:
   - Reads the chosen `.env.*` before naming the VM so the hostname reflects the anchor (e.g., `hello-agent-<ts>-tpm`).
   - Creates a Shielded VM with vTPM enabled (required for TPM anchors).
   - Installs the SDK + deps, pins NumPy `<2`, sets up `tpm2-tools`, and starts the bot via `nohup … >> helloworld.log`.
   - *Interactive mode (default)*: after provisioning completes, the script automatically SSHs into the VM and runs `python helloworld_agent.py` so you can chat immediately.
   - *Background mode*: pass `--no-interactive` if you want to deploy without opening the chatbot. Logs stream to `~/helloworld/helloworld.log` either way; use `./reattach_gcp.sh` to tail them or run the printed `gcloud … tail` command manually.
5. By default we install the CPU-only PyTorch wheel to avoid downloading the 1.8 GB CUDA stack. If you need GPU builds, pass `--gpu` (expect slower provisioning).
6. Detailed GCP setup: [GCP.md](/Users/aidevmac/Projects/Ephapsys/Product/ephapsys-sdk/samples/agents/helloworld/GCP.md)

### HSM / Cloud KMS mode

If you need centralized key custody (`PERSONALIZE_ANCHOR=hsm`):

1. Provision a Cloud KMS asymmetric-signing key (optionally with `--protection-level=hsm`) and grant the VM’s service account `roles/cloudkms.signerVerifier` on that key. You can automate this with `./provision_kms_key.sh --project <id> --service-account <sa@project.iam.gserviceaccount.com> [--hsm]`.
2. In `.env.gcp`, set:
   ```ini
   PERSONALIZE_ANCHOR=hsm
   HSM_KMS_KEY=projects/<project>/locations/<loc>/keyRings/<ring>/cryptoKeys/<key>[/cryptoKeyVersions/<n>]
   # Optional when you want to upload a dedicated SA JSON instead of using the instance metadata credentials:
   HSM_KMS_CREDENTIALS=/absolute/path/to/service-account.json
   HSM_KMS_ENDPOINT=privatekms.googleapis.com   # optional
   ```
   Optional overrides: `HSM_KMS_LOCATION` (default `us-central1`), `HSM_KMS_KEY_RING` (default `agents`), `HSM_KMS_KEY_NAME` (default `helloworld`), `HSM_KMS_USE_HSM=1` (forces `--protection-level=hsm` when auto-provisioning), and `COMPUTE_SERVICE_ACCOUNT` if you don’t want to use the project’s default Compute Engine service account.
   > Not on Google Cloud? Leave the `HSM_KMS_*` vars empty, set `HSM_HELPER` to the command that talks to your HSM (for example, a PKCS#11-based helper), and configure whatever environment variables that helper requires. As long as it prints the evidence JSON expected by the SDK (`sig_b64`, `nonce_b64`, optional cert chain, slot, label), personalization will succeed with any vendor module.
3. Run `./run.sh --gcp`. If `HSM_KMS_KEY` is blank and no helper is configured, the script automatically provisions the Cloud KMS key (via `provision_kms_key.sh`), grants the Compute Engine service account `roles/cloudkms.signerVerifier`, and then proceeds with deployment. It also skips TPM tooling, uploads any credential JSON into `~/helloworld/.secrets/kms-credentials.json`, rewrites `.env` on the VM so both `HSM_KMS_CREDENTIALS` and `GOOGLE_APPLICATION_CREDENTIALS` point at that path, and prints a `[VM] Personalization anchor=hsm` log so it’s obvious which mode is active.
4. `helloworld_agent.py` now calls Google Cloud KMS via `google-cloud-kms` to sign the backend nonce, and `reattach_gcp.sh` can reconnect later without touching TPM-specific flow.

---

## 📂 Files

- `helloworld_agent.py` → Minimal TrustedAgent HelloWorld demo.
  - Includes optional commented GGUF/llama.cpp hints; keep default behavior unchanged unless you enable them.
  - Includes optional commented A2A hints using `A2AClient` (send/inbox/ack) for peer-agent messaging.
- `run.sh` → Public entrypoint. Use `--local` for local execution or `--gcp` for VM deployment.
- `run_local.sh` → Local helper invoked by `run.sh --local`.
- `run_gcp.sh` → GCP helper invoked by `run.sh --gcp`, with `--interactive/--no-interactive` and optional `--gpu`. Starts the bot in the background, prints the log-tail command, and (by default) opens an interactive chatbot session.
- `check_gcp.sh` → GCP preflight helper for local deployment setup.
- `GCP.md` → focused GCP setup notes for this sample.
- `reattach_gcp.sh` → Smart reconnection helper. If it detects `helloworld_agent.py` running, it loads `.env`, activates the venv, and drops you into the chatbot. Otherwise it tails `~/helloworld/helloworld.log`.
- `detach_gcp.sh` → (Legacy) left around in case you revert to tmux; not needed in the default workflow.
- `check_iam_role.sh` → Quick helper to verify whether the current (or supplied) gcloud account has `roles/compute.instanceAdmin.v1` on the project.
- `provision_kms_key.sh` → Convenience script to create a Cloud KMS key (optionally HSM‑backed) and grant a service account the `roles/cloudkms.signerVerifier` role.
- *IAM note*: The scripts call `gcloud compute ssh`, which requires `roles/compute.instanceAdmin.v1`. If you’re running in a sandbox where gcloud can’t write credentials (common in CI), grant that role once via Cloud Console or from a machine with full gcloud access.
- `check_iam_role.sh` → Quick helper to verify whether the current (or supplied) gcloud account has `roles/compute.instanceAdmin.v1` on the project.
- `.env.example` → tracked example env for local bootstrap.
- `.env.gcp.example` → tracked template for your untracked local `.env.gcp`.
