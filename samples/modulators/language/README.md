# Modulator script for Language (Flan-T5) with ephaptic coupling integration

Fine-tuning GPT-2 on WikiText makes sense because both share the same goal — predicting the next word in a sequence. GPT-2 is a causal language model trained for free-form text generation, and WikiText provides high-quality, well-structured English suitable for that purpose. This alignment helps GPT-2 improve its fluency and factual style without changing its fundamental behavior, unlike FLAN-T5, which is tuned for instruction following rather than raw language modeling.

Th train_language.py is generic for the task type (“language”), butmodel-agnostic because different “language models” fall into two fundamentally different architectural classes:

## 1. Encoder–Decoder (Seq2Seq) models

Examples: T5, Flan-T5, BART, Pegasus, MT5, etc.

- Use AutoModelForSeq2SeqLM.

- Expect both an encoder and a decoder.

- Typical for translation, summarization, or instruction-following tasks.

- The code’s calls like model.get_encoder() and compute_language_metrics_stream() are designed for this structure.

## 2. Decoder-only (Causal LM) models

Examples: GPT-2, GPT-Neo, LLaMA, Falcon, etc.

- Use AutoModelForCausalLM.

- Have only a decoder (no get_encoder()).

- Trained with next-token prediction objectives (language modeling).

- We can’t call get_encoder() or use Seq2Seq-style metric pipelines.


## 1. Create Model Template and Open Modulator
1. Create a **Model Template** (via the *Create Model* page in the AOC):  
   - **Source:** External repository  
   - **Provider:** Hugging Face  
   - **Repository ID:** `openai-community/gpt2`, `google/flan-t5-small`, `google/gemma-3-270m`
   - **Model Kind:** `language`  
   - **Revision:** `main`  
   - **Hugging Face Token:** `hf_xxxxxxxx`  
   -  *Register immediately* (so a provenance certificate is issued)

2. Go to the **Modulator page** for this template:  
   - **Mode:** Automatic  
   - **MaxSteps:** number of steps/samples to evaluate  
   - **Dataset:** specify name/config/split (e.g., `wikitext`, `wikitext-103-raw-v1`, `train[:1%]`)  
   - **KPI Targets:** enable at least one relevant KPI (e.g., Accuracy, Perplexity, Loss)


**TODO Future improvements to the generated reports are mainly diagnostic:**

- Include Λ-matrix delta plots (optional but insightful),
- Confirm ECM injection actually mutates model parameters,
- Log per-step metric deltas to visualize ephaptic effect over time.
- Support other evaluation metrics for language (ROUGE for summarization tasks, BLEU for translation tasks, etc.)
---

## 2. Configure Environment

### Infra config (`gcp.env`)
This file contains **GCP infrastructure settings** (project, region, zone).  

1. Copy the example file:  
   ```bash
   cp gcp.env.example gcp.env
   ```
2. Edit `gcp.env` with your project details:  
   ```bash
   export PROJECT_ID=<your-gcp-project-id>
   export REGION=<us-central1>
   export ZONE=<us-central1-f>
   export REPO=<repo-docker>
   # optional overrides:
   # export IMAGE_NAME=modulate
   # export TAG=latest
   # export DOCKER_PLATFORM=linux/amd64
   ```
3. `deploy_and_run.sh` will automatically source this file at runtime if present.

### Runtime config (`.env`)
This file contains **modulator runtime values** (passed into the container).  

```ini
API_KEY=your_api_key_here
MODEL_TEMPLATE_ID=your_model_template_id_here
BASE_URL=http://localhost:7001   # or your AOC backend
```

---

## 3. Run Modulation on GCP (Containerized)

We now run modulation inside a GPU container on a **temporary GCP VM**.  
The VM is **auto-deleted** when the job finishes unless `AUTO_DELETE=false` is set.  

### Default run (build + push new container, then modulate on T4 GPU):
```bash
./modulate_gcp.sh t4
```

### Run on A100:
```bash
./modulate_gcp.sh a100
```

### Run on A100-4G GPU:
```bash
./modulate_gcp.sh a100-4g
```

### Skip rebuild/push and reuse last image (faster when unchanged):
```bash
./modulate_gcp.sh t4 --no-build
```

### Build locally but skip push (useful for quick testing):
```bash
./modulate_gcp.sh t4 --no-push
```

Results will be copied locally into:  
```
results/modulate_<timestamp>/artifacts/
```

---

## 4. Notes

- The script automatically:  
  - Enables the Artifact Registry API (if not already enabled).  
  - Creates the Artifact Registry repo if missing.  
  - Builds & pushes a Docker image containing `train_language.py` (unless `--no-build` or `--no-push`).  
  - Provisions a GPU VM (T4 or A100 family).  
  - Installs Docker + NVIDIA runtime on the VM if not already installed.  
  - Runs modulation inside the container with your `.env` values injected.  
  - Copies back artifacts for inspection.  
  - Cleans up the VM.  

- You can keep the VM for debugging by running with:  
  ```bash
  AUTO_DELETE=false ./modulate_gcp.sh t4
  ```

---

## 5. SDK Integration Options

Since `train_language.py` imports `from ephapsys.modulation import ModulatorClient`, the container must include the Ephapsys SDK. There are four ways to integrate it:

1. **Local source copy (current default for SAMPLES)**  
   - Dockerfile copies the SDK source tree directly:  
     ```dockerfile
     COPY sdk/python/ephapsys /workspace/ephapsys
     ```  
   - Works for internal testing without publishing the SDK.

2. **Install from private GitHub repo (pre-release option)**  
   - Add to Dockerfile:  
     ```dockerfile
     RUN pip install git+https://github.com/your-org/ephapsys-sdk.git@main
     ```  
   - Requires a GitHub token if the repo is private.

3. **Install from Artifact Registry (internal distribution)**  
   - Build and upload your SDK wheel (`.whl`) to a private PyPI repository on GCP:  
     ```bash
     gcloud artifacts repositories create ephapsys-pypi        --repository-format=python        --location=us-central1        --description="Ephapsys Python packages"
     ```
   - Then in Dockerfile:  
     ```dockerfile
     RUN pip install --extra-index-url=https://us-central1-python.pkg.dev/ephapsys-dev/ephapsys-pypi/simple ephapsys
     ```

4. **Install from PyPI (future public release)**  
   - Once the SDK is published publicly:  
     ```dockerfile
     RUN pip install ephapsys
     ```

---

👉 Summary:  
- `gcp.env` → infra config (not committed, based on `gcp.env.example`)  
- `.env` → runtime config (API keys, template IDs, etc.)  
- SDK available via: local copy (default), private GitHub, private Artifact Registry, or PyPI.  
