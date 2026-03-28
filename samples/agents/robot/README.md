# Robot Agent (Sample with Ephapsys SDK)

This sample demonstrates a simple **Robot Agent** built on the Ephapsys SDK.  
The agent can **hear, see, think, and speak** using offline models after being verified and personalized through the TrustedAgent.

The sample is structured as a small local `body + brain + face` demo:
- `body` handles microphone, camera, and speaker I/O
- `brain` is a local FastAPI service that owns runtime preparation, trusted verification, memory, and model orchestration
- `face` is the terminal UI developers interact with today and connects to the brain over localhost
- `channel` is the local event/command boundary between body and brain

There is now a second deployment shape for this sample:
- local mode: `body + brain + face` all run on the same machine
- GCP mode: only the `brain` runs remotely; the local machine keeps the `body` and terminal `face`

> ⚠️ **Important Requirements Before Running**  
> This demo will not work out of the box unless you first prepare your Ephapsys environment:  
> 
> 1. **Ephapsys Account & Credentials**  
>    - You must have an active Ephapsys account.  
>    - Get `AOC_ORG_ID` + `AOC_PROVISIONING_TOKEN` from AOC.  
> 
> 2. **Model Bootstrapping**  
>    - `./push.sh` registers the baseline robot models and idempotently publishes them by default.
>    - The starter embedding model is `sentence-transformers/all-MiniLM-L6-v2`, which does not require gated Hugging Face access.
>    - Set `ROBOT_ENABLE_WORLD_MODEL=1` if you also want to bootstrap the optional V-JEPA world model (`facebook/vjepa2-vitl-fpc64-256`).
> 
> 3. **Agent Template**  
>    - You must create an Agent Template in the AOC that references these modulated models.  
>    - Note the template ID and configure it in your `.env` or environment variables.  
> 
> If these steps are not completed, the robot agent will fail with verification or runtime errors (e.g., 404 Agent Not Found, missing models).

---

## 🧩 Workflow

1. **Verify agent status**
   - If not yet personalized, run `personalize()` once with a supported anchor (`tpm`, `hsm`, or `none` for local dev only).
   - If revoked or disabled, refuse to start.

2. **Enter main loop**
   - Re-verify agent status on every cycle (revocation-aware).
   - Capture microphone input (speech until silence) and camera input (sampled every N seconds).
   - Process/think:
     - **STT** → convert mic audio to text.
     - **Vision** → classify what the camera sees.
     - **Language Gen** → combine inputs into a response.
     - **Embedding + FAISS Memory** → compute a semantic vector, store it, and retrieve most similar past response.
   - Output action:
     - **TTS** → speak the response via audio, augmented with memory context.

---

## 🔄 Data Flow

```mermaid
flowchart TD
    subgraph Body["Body"]
        Mic[Microphone Capture]
        Cam[Camera Capture]
        Speaker[Speaker Output]
    end

    subgraph Channel["Local Channel"]
        Events[Events<br/>microphone, camera]
        Commands[Commands<br/>speak]
    end

    subgraph Brain["Brain Service"]
        V[Verify Agent Status]
        STT[Speech-to-Text]
        Vision[Vision Understanding]
        Lang[Language Generation]
        Emb[Embedding Vector + FAISS Memory]
    end

    subgraph Face["Terminal Face"]
        UI[Live Console]
    end

    Mic --> Events
    Cam --> Events
    Events --> V
    V --> STT
    V --> Vision
    STT --> Lang
    Vision --> Lang
    Lang --> Emb
    Lang --> Commands
    Commands --> Speaker
    Brain --> UI
```

---

## ▶️ Run

### 1. Install dependencies
```bash
pip install "ephapsys[audio,vision,embedding]"
pip install webrtcvad sounddevice pyaudio
```

### 2. Set environment variables (or create .env file)
```bash
export AOC_BASE_URL=https://api.ephapsys.com
export AOC_ORG_ID=org_xxxxxxxxx
export AOC_PROVISIONING_TOKEN=boot_xxxxxxxxx
export AOC_MODULATION_TOKEN=mod_xxxxxxxxx
export AGENT_TEMPLATE_NAME="Robot Agent Template"
export PERSONALIZE_ANCHOR=none
export ROBOT_ENABLE_WORLD_MODEL=0
```

### 3. First run

Use the one-command bootstrap:

```bash
./quickstart.sh
```

If `.env` is missing, this creates it from `.env.example` and stops so you can
fill in the required values. Rerun `./quickstart.sh` after editing `.env`.

### 4. Direct entrypoints

Bootstrap robot templates locally:

```bash
./push.sh
```

Bootstrap the same templates for a later GCP brain run:

```bash
./push.sh --gcp
```

By default this uses idempotent publish for the model templates. If you want the
robot stack to run full modulation instead, use:

```bash
./push.sh --no-idempotent
```

To include the optional V-JEPA world-model template in the robot stack:

```bash
ROBOT_ENABLE_WORLD_MODEL=1 ./push.sh
```

Run the robot agent locally:

```bash
./run.sh
```

Run the robot brain in GCP while keeping the local microphone, camera, speaker,
and terminal face on your machine:

```bash
cp .env.gcp.example .env.gcp
./check_gcp.sh
./run.sh --gcp
```

Notes for `--gcp`:
- it provisions a Compute Engine VM
- installs the current SDK version from PyPI by default
- starts only `robot_brain_server.py` on the VM
- opens an SSH tunnel back to `localhost`
- runs `robot_remote_agent.py` locally so your body and terminal stay local
- GCP deployment settings come from local `.env.gcp` and are not tracked
- package-source overrides are env-based:
  - `SDK_PACKAGE_SOURCE=custom` with `SDK_INDEX_URL=...`
- detailed setup: [GCP.md](/Users/aidevmac/Projects/Ephapsys/Product/ephapsys-sdk/samples/agents/robot/GCP.md)

---

## 🔒 Permissions (Local Setup)

Unlike a browser app, this Python demo will not pop up a permission request for microphone or camera access. Instead, you need to ensure your operating system has already granted access to Python (or your terminal) before running the demo:

- **macOS**
  - Go to **System Preferences → Security & Privacy → Privacy**.
  - Under **Microphone**, check the box for your terminal or Python interpreter.
  - Under **Camera**, do the same.
  - If these aren’t enabled, `sounddevice` will record silence and `cv2.VideoCapture(0)` will fail with “Camera capture failed.”

- **Windows**
  - Go to **Settings → Privacy → Microphone/Camera**.
  - Ensure “Allow apps to access” is enabled.
  - Check that Python is allowed access (sometimes listed under “Desktop apps”).

- **Linux**
  - Most distributions don’t enforce application-level mic/cam permissions.
  - Make sure your user account has access to `/dev/audio` and `/dev/video0` devices.

⚠️ If permissions aren’t set correctly:
- Microphone capture will yield empty audio.
- Camera capture will return errors or blank frames.

---

## 📂 Files

- `robot_agent.py` → Thin launcher that starts the local brain server and terminal face.
- `robot_channel.py` → Local typed event/command channel between body and brain.
- `robot_body.py` → Device/body layer for microphone, camera, and speaker I/O.
- `robot_brain.py` → Runtime/brain layer for verification, orchestration, and memory.
- `robot_brain_server.py` → Local FastAPI brain service exposing runtime state over WebSocket.
- `robot_remote_agent.py` → Local remote-body client for GCP mode.
- `robot_face.py` → Terminal face layer for live developer feedback.
- `check_gcp.sh` → GCP preflight helper for local deployment setup.
- `GCP.md` → focused GCP setup notes for this sample.
- `quickstart.sh` → Creates `.env` if needed, bootstraps robot templates, then launches the sample.
- `.env.gcp.example` → tracked template for your untracked local `.env.gcp`.
- `push.sh` → Public bootstrap entrypoint. No flag defaults to local bootstrap; `--gcp` dispatches to GCP bootstrap.
- `push_gcp.sh` → GCP wrapper for the same robot template bootstrap flow.
- `run.sh` → Public run entrypoint that dispatches to `run_local.sh` or `run_gcp.sh`.
- `run_gcp.sh` → Brain-only GCP deployer plus local remote-body launcher.
- `run_brain_server.sh` → Brain-only server launcher used by GCP mode.
- `push_local.sh` → Current local implementation for model registration/modulation and agent template creation.
- `run_local.sh` → Current local implementation for robot runtime startup.
- `requirements_brain.txt` → Minimal dependency set for brain-only remote deployment.
- `.env` → cp .env.example .env

For repo-local SDK development only, you can force the sample to install the SDK
from this checkout by running:

```bash
ROBOT_USE_LOCAL_SDK=1 ./run.sh --local
```
