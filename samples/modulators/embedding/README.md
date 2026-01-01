# Trainer script for Embedding (Gemma-300M) with ephaptic coupling integration.

## Create Model Template and Open Modulator

1. Create a Model Template (via the Create Model page):
   - Source: External repository
   - Provider: Hugging Face
   - Repository ID: google/embeddinggemma-300m
   - Model Kind: embedding
   - Revision: main
   - Hugging Face Token: hf_xxxxxxxx
   - Register immediately (so a provenance certificate is issued)

2. Go to the Modulator page for this template:
   - Select Manual
   - Variant: additive or multiplicative
   - Hyperparameters: epsilon (ε), lambda0 (λ₀), phi (activation), ecm_init
   - MaxSteps: number of samples/steps to evaluate
   - KPI Targets: enable embedding KPIs (Cosine Similarity, Recall@k)

## Run Terminal
```bash
 ./modulate.sh
```
