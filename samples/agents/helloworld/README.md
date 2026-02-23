# HelloWorld Agent (Sample with Ephapsys SDK)

This sample demonstrates the simplest possible agent using the Ephapsys SDK.  
It verifies, personalizes (if needed), and runs a language model to output **"Hello World"**.


> ⚠️ **Important Requirements Before Running**  
> This demo will not work out of the box unless you first prepare your Ephapsys environment:  
> 
> 1. **Ephapsys Account & Credentials**  
>    - You must have an active Ephapsys account.  
>    - Preferred: get `AOC_ORG_ID` + `AOC_BOOTSTRAP_TOKEN` from AOC (legacy `AOC_API_KEY` remains compatibility-only).  
> 
> 2. **Model Modulation**  
>    - At least a **Language model** must be modulated in the AOC (for HelloWorld).  Look into **'modulators** folder for an example.
>    - Without modulation, the TrustedAgent will not be able to fetch runtime artifacts.  
> 
> 3. **Agent Template**  
>    - You must create an Agent Template in the AOC that references the modulated Language model.  
>    - Note the template ID and configure it in your `.env` or environment variables.  
> 
> If these steps are not completed, the HelloWorld agent will fail with verification or runtime errors (e.g., 404 Agent Not Found, missing models).

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

### Before you run `run_gcp.sh`
Complete the checklist below after publishing the new SDK and redeploying the AOC backend:

1. **Modulated model + agent template** – Ensure the model you want is modulated in AOC and bound to an agent template; capture the template ID for `.env.*`.
2. **API credentials** – Keep your usual `.env`, `.env.stag`, and `.env.prod` files up to date (the script automatically prefers those; it falls back to the local copies in this folder only if the global ones are missing).
3. **gcloud access** – Run `gcloud auth login` (or `gcloud auth application-default login`) on the machine invoking `run_gcp.sh` and make sure you can `gcloud compute ssh` into the target project/zone.
4. **Compute IAM** – The user running `gcloud` needs `roles/compute.instanceAdmin.v1`. Use `./check_iam_role.sh --project <project> --member user:<you>` (or grant the role via Cloud Console).
5. **Cloud KMS key (only if `PERSONALIZE_ANCHOR=hsm`)**
   - If you already have a key, set `HSM_KMS_KEY` (and optionally `HSM_KMS_CREDENTIALS`) in `.env.stag`/`.env.prod`.
   - Otherwise do nothing—`run_gcp.sh` will automatically call `./provision_kms_key.sh` (using the default Compute Engine service account or the one you specify via `COMPUTE_SERVICE_ACCOUNT`) and capture the resulting `HSM_KMS_KEY` for you.
6. **Vendor HSM (non-GCP)** – Skip the KMS step, set `HSM_HELPER="<your helper command>"`, and export whatever PKCS#11/KMIP env vars the helper requires; confirm it prints valid JSON when invoked with a nonce.
7. **Local `.env` sanity check** – Run `grep -v '^#' .env.stag` (or `.env.prod`) and verify no required variable is blank.

### Local

1. Install runtime dependencies:
   ```bash
   pip install "ephapsys[modulation]"
   ```
2. Copy `.env.example` to `.env` and fill in your `AOC_BASE_URL`, `AOC_ORG_ID`, `AOC_BOOTSTRAP_TOKEN`, and `AGENT_TEMPLATE_ID`.
3. Execute `./run_local.sh` – it sources `.env`, defaults `PERSONALIZE_ANCHOR=tpm`, and launches `helloworld_agent.py`.

### GCP VM

1. Populate `.env.stag` or `.env.prod` with the appropriate credentials (`AOC_BASE_URL`, `AOC_ORG_ID`, `AOC_BOOTSTRAP_TOKEN`, `AGENT_TEMPLATE_ID`)—both set `PERSONALIZE_ANCHOR=tpm` by default.
2. Run `./run_gcp.sh --staging` or `./run_gcp.sh --production`. The script:
   - Reads the chosen `.env.*` before naming the VM so the hostname reflects the anchor (e.g., `hello-agent-<ts>-tpm`).
   - Creates a Shielded VM with vTPM enabled (required for TPM anchors).
   - Installs the SDK + deps, pins NumPy `<2`, sets up `tpm2-tools`, and starts the bot via `nohup … >> helloworld.log`.
   - *Interactive mode (default)*: after provisioning completes, the script automatically SSHs into the VM and runs `python helloworld_agent.py` so you can chat immediately.
   - *Background mode*: pass `--no-interactive` if you want to deploy without opening the chatbot. Logs stream to `~/helloworld/helloworld.log` either way; use `./reattach_gcp.sh` to tail them or run the printed `gcloud … tail` command manually.
3. By default we install the CPU-only PyTorch wheel to avoid downloading the 1.8 GB CUDA stack. If you need GPU builds, pass `--gpu` (expect slower provisioning).

### HSM / Cloud KMS mode

If you need centralized key custody (`PERSONALIZE_ANCHOR=hsm`):

1. Provision a Cloud KMS asymmetric-signing key (optionally with `--protection-level=hsm`) and grant the VM’s service account `roles/cloudkms.signerVerifier` on that key. You can automate this with `./provision_kms_key.sh --project <id> --service-account <sa@project.iam.gserviceaccount.com> [--hsm]`.
2. In `.env.stag`/`.env.prod`, set (or rely on the values automatically loaded from your global `.env.*` files):
   ```ini
   PERSONALIZE_ANCHOR=hsm
   HSM_KMS_KEY=projects/<project>/locations/<loc>/keyRings/<ring>/cryptoKeys/<key>[/cryptoKeyVersions/<n>]
   # Optional when you want to upload a dedicated SA JSON instead of using the instance metadata credentials:
   HSM_KMS_CREDENTIALS=/absolute/path/to/service-account.json
   HSM_KMS_ENDPOINT=privatekms.googleapis.com   # optional
   ```
   Optional overrides: `HSM_KMS_LOCATION` (default `us-central1`), `HSM_KMS_KEY_RING` (default `agents`), `HSM_KMS_KEY_NAME` (default `helloworld`), `HSM_KMS_USE_HSM=1` (forces `--protection-level=hsm` when auto-provisioning), and `COMPUTE_SERVICE_ACCOUNT` if you don’t want to use the project’s default Compute Engine service account.
   > Not on Google Cloud? Leave the `HSM_KMS_*` vars empty, set `HSM_HELPER` to the command that talks to your HSM (for example, a PKCS#11-based helper), and configure whatever environment variables that helper requires. As long as it prints the evidence JSON expected by the SDK (`sig_b64`, `nonce_b64`, optional cert chain, slot, label), personalization will succeed with any vendor module.
3. Run `./run_gcp.sh --staging` (or `--production`). If `HSM_KMS_KEY` is blank and no helper is configured, the script automatically provisions the Cloud KMS key (via `provision_kms_key.sh`), grants the Compute Engine service account `roles/cloudkms.signerVerifier`, and then proceeds with deployment. It also skips TPM tooling, uploads any credential JSON into `~/helloworld/.secrets/kms-credentials.json`, rewrites `.env` on the VM so both `HSM_KMS_CREDENTIALS` and `GOOGLE_APPLICATION_CREDENTIALS` point at that path, and prints a `[VM] Personalization anchor=hsm` log so it’s obvious which mode is active.
4. `helloworld_agent.py` now calls Google Cloud KMS via `google-cloud-kms` to sign the backend nonce, and `reattach_gcp.sh` can reconnect later without touching TPM-specific flow.

---

## 📂 Files

- `helloworld_agent.py` → Minimal TrustedAgent HelloWorld demo.
- `run_local.sh` → Local convenience wrapper.
- `run_gcp.sh` → GCP launcher with `--staging/--production`, `--interactive/--no-interactive`, and optional `--gpu`. Starts the bot in the background, prints the log-tail command, and (by default) opens an interactive chatbot session.
- `reattach_gcp.sh` → Smart reconnection helper. If it detects `helloworld_agent.py` running, it loads `.env`, activates the venv, and drops you into the chatbot. Otherwise it tails `~/helloworld/helloworld.log`.
- `detach_gcp.sh` → (Legacy) left around in case you revert to tmux; not needed in the default workflow.
- `check_iam_role.sh` → Quick helper to verify whether the current (or supplied) gcloud account has `roles/compute.instanceAdmin.v1` on the project.
- `provision_kms_key.sh` → Convenience script to create a Cloud KMS key (optionally HSM‑backed) and grant a service account the `roles/cloudkms.signerVerifier` role.
- *IAM note*: The scripts call `gcloud compute ssh`, which requires `roles/compute.instanceAdmin.v1`. If you’re running in a sandbox where gcloud can’t write credentials (common in CI), grant that role once via Cloud Console or from a machine with full gcloud access.
- `check_iam_role.sh` → Quick helper to verify whether the current (or supplied) gcloud account has `roles/compute.instanceAdmin.v1` on the project.
- `.env.example`, `.env.stag`, `.env.prod` → env templates for each target (copied during deployment).
