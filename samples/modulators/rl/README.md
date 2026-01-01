# Trainer script for Reinforcement Learning (RL) with ephaptic coupling integration.

## Create Model Template and Open Modulator

1. Create a Model Template (via the Create Model page):
   - Source: Custom or External
   - Provider: Hugging Face or custom repo
   - Repository ID: <your-rl-model-repo>
   - Model Kind: rl
   - Register immediately (so a provenance certificate is issued)

2. Go to the Modulator page for this template:
   - Select Manual
   - Variant: additive or multiplicative
   - Hyperparameters: epsilon (ε), lambda0 (λ₀), phi (activation), ecm_init
   - MaxSteps: number of episodes/steps to evaluate
   - KPI Targets: enable RL KPIs (Reward, Success Rate)

## Run Terminal
```bash
 ./modulate.sh
```
