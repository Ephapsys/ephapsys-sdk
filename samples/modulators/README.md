# Ephapsys Modulation 

This repository provides reference modulation scripts for different **Model Kinds** supported in Ephapsys.  
Each modulation (also referred to as trainer) follows the same structure adapted for the corresponding task and KPIs.

---

## 📑 Trainers Overview

| Model Kind  | Default Model Repo                  | KPIs                        | Folder        | Notes |
|-------------|--------------------------------------|-----------------------------|---------------|-------|
| **TTS**     | `microsoft/speecht5_tts`             | `wer`, `mos`                | `tts/`        | Speech synthesis |
| **STT**     | `openai/whisper-tiny.en`             | `wer`                       | `stt/`        | Speech recognition |
| **Language**| `google/flan-t5-small`               | `accuracy`, `perplexity`, `loss` | `language/` | Text generation & classification |
| **Embedding**| `google/embeddinggemma-300m`        | `cosine_sim`, `recall_at_k` | `embedding/`  | Text embedding evaluation |
| **Vision**  | `hustvl/yolos-base`                  | `accuracy`, `fid`           | `vision/`     | Object detection & classification |
| **RL**      | `<your-rl-model-repo>`               | `reward`, `success_rate`    | `rl/`         | Reinforcement learning loop (Gym/Isaac/etc.) |
| **Audio**   | `superb/wav2vec2-base-superb-ks`     | `accuracy`                  | `audio/`      | Audio classification |

---

## 🌀 Ephapsys Modulation Flow

The **modulation process** controls fine-tuning / training inside Ephapsys.  
Instead of directly running `train.py`, all runs are orchestrated by the **Agent Operations Center (AOC)** using the **Modulator**.

```mermaid
flowchart TD
    subgraph AOC [Agent Operations Center]
        UI[Modulator Page UI]
        CLI[CLI / SDK]
    end

    UI -->|User sets epsilon λ₀ φ ECM| M(Modulator Config)
    CLI -->|Register model template| T(Template Registry)

    T -->|Fetch template + dataset config| Trainer
    M -->|Pass modulation params| Trainer

    subgraph Trainer [Trainer Script (per kind)]
        L[Load Model from Hugging Face Repo]
        D[Load Dataset via HF Datasets]
        F[Forward Pass / Inference Loop]
        K[Compute KPIs (e.g. WER, Accuracy, etc.)]
        R[Aggregate Metrics]
    end

    Trainer -->|Report metrics| AOC

    AOC -->|Store provenance, KPIs| Ledger[(Provenance Ledger)]
```

---

## 🔑 Key Points

1. **Model Templates**  
   - Created once via the UI or CLI.  
   - Define repo (`SourceRepo`), kind (`ModelKind`), dataset, and revision.  
   - Ephapsys issues a provenance certificate on registration.

2. **Modulator Config**  
   - Defines how ephaptic parameters (`ε`, `λ₀`, `φ`, `ECM_init`) affect training.  
   - These parameters are applied **at runtime**, without hardcoding in the trainer.

3. **Trainer Scripts**  
   - Minimal CLI args: `--base_url`, `--api_key`, `--model_template_id`, `--outdir`.  
   - Load model + dataset automatically from the registered template.  
   - Run forward passes and compute KPIs.  
   - Report metrics back to Ephapsys.

4. **KPIs**  
   - Each trainer aligns with the KPI definitions used in the Modulator dashboard.  
   - Examples: `WER` (STT/TTS), `Accuracy` (Language, Vision, Audio), `Reward` (RL), `Cosine Similarity` (Embedding).

5. **Artifacts**  
   - All outputs are stored under `./artifacts_<kind>/` with timestamped subfolders.  
   - Ephapsys uses these to trace training provenance and governance.

---

✅ With this structure, we can easily plug in new Hugging Face models or custom repos, while keeping a uniform workflow across **Language, Vision, Speech, RL, Embeddings, and Audio**.

