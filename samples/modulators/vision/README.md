# Trainer script for Vision (YOLOS) with ephaptic coupling integration.

## Create Model Template and Open Modulator

1. Create a Model Template (via the Create Model page):
   - Source: External repository
   - Provider: Hugging Face
   - Repository ID: hustvl/yolos-base
   - Model Kind: vision
   - Revision: main
   - Hugging Face Token: hf_xxxxxxxx
   - Register immediately (so a provenance certificate is issued)

2. Go to the Modulator page for this template:
   - Select Manual
   - Variant: additive or multiplicative
   - Hyperparameters: epsilon (ε), lambda0 (λ₀), phi (activation), ecm_init
   - MaxSteps: number of samples/steps to evaluate
   - Dataset: name (e.g., cifar10), split (e.g., test[:1%])
   - KPI Targets: enable at least one KPI relevant to vision (e.g., Accuracy, FID)

## Run Terminal

```bash
 ./modulate.sh
```


