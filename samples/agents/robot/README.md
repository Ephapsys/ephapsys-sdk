# Robot Agent (Sample with Ephapsys SDK)

This sample demonstrates a simple **Robot Agent** built on the Ephapsys SDK.  
The agent can **hear, see, think, and speak** using offline models after being verified and personalized through the TrustedAgent.

> ⚠️ **Important Requirements Before Running**  
> This demo will not work out of the box unless you first prepare your Ephapsys environment:  
> 
> 1. **Ephapsys Account & Credentials**  
>    - You must have an active Ephapsys account.  
>    - Get `AOC_ORG_ID` + `AOC_BOOTSTRAP_TOKEN` from AOC.  
> 
> 2. **Model Modulation**  
>    - All required models (STT, TTS, Language, Vision, Embedding) must be modulated in the AOC. Look into **'modulators** folder for  examples.
>    - Without modulation, the TrustedAgent will not be able to fetch runtime artifacts.  
> 
> 3. **Agent Template**  
>    - You must create an Agent Template in the AOC that references these modulated models.  
>    - Note the template ID and configure it in your `.env` or environment variables.  
> 
> If these steps are not completed, the robot agent will fail with verification or runtime errors (e.g., 404 Agent Not Found, missing models).

---

## 🧩 Workflow

1. **Verify agent status**
   - If not yet personalized, run `personalize()` once with the chosen anchor (`tpm`, `tee`, or `dsim`).
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
    subgraph RobotAgent["Robot Agent Loop"]
        V[Verify Agent Status]
        Mic[Microphone Audio]
        Cam[Camera Frame]
        STT[Speech-to-Text]
        Vision[Vision Classification]
        Lang[Language Generation]
        Emb[Embedding Vector + FAISS Memory]
        TTS[Text-to-Speech (with Memory Context)]

        V -->|ok| Mic
        V -->|ok| Cam
        Mic --> STT
        STT --> Lang
        Cam --> Vision
        Vision --> Lang
        Lang --> Emb
        Emb --> TTS
    end

    TTS --> Out[Speaker Output (Voice)]
```

---

## ▶️ Run

### 1. Install dependencies
```bash
pip install "ephapsys[modulation,audio,vision,embedding]"
pip install webrtcvad sounddevice pyaudio
```

### 2. Set environment variables (or create .env file)
```bash
export AOC_API_URL=http://localhost:8000
export AOC_ORG_ID=org_xxxxxxxxx
export AOC_BOOTSTRAP_TOKEN=bt_xxxxxxxxx
export AGENT_TEMPLATE_ID=agent_robot
export PERSONALIZE_ANCHOR=tpm
```

### 3. Run the robot agent
Use the provided shell wrapper:

```bash
 ./run.sh
```

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

- `robot_agent.py` → Main Python sample agent loop (mic until silence, cam frame every N seconds, FAISS memory).
- `run.sh` → Convenience wrapper to start the agent with environment setup.
- `.env` → cp .env.example .env
