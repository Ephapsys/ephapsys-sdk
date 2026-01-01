
# Trainer script for TTS with ephaptic coupling integration.

## Create Model Template and Open Modulator

1. Create a Model Template (via the Create Model page):
   - Source: External repository
   - Provider: Hugging Face
   - Repository ID: microsoft/speecht5_tts
   - Model Kind: TTS
   - Revision: main
   - Hugging Face Token: hf_xxxxxxxx
   - Register immediately (so a provenance certificate is issued)

2. Go to the Modulator page for this template:
   - Select Manual
   - Variant: additive or multiplicative
   - Hyperparameters: epsilon (ε), lambda0 (λ₀), phi (activation), ecm_init
   - MaxSteps: number of samples/steps to evaluate
   - Dataset: name (e.g., librispeech_asr), config (e.g., clean), split (e.g., validation[:1%])
   - KPI Targets: enable at least one KPI relevant to TTS (e.g., WER, MOS)

## Run Terminal
```bash
 ./modulate.sh
```



