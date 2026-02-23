#!/usr/bin/env python3
"""
Trainer script for Reinforcement Learning (RL) with ephaptic coupling integration.

This trainer streams per-episode metrics (reward, success_rate) back to the AOC,
so the frontend UI can render live charts during training.

Usage flow:
- Minimal CLI args: --base_url, --api_key, --model_template_id, --outdir
- All hyperparameters, environment config, and model_id are fetched dynamically
  from the backend template created in the UI.
- The trainer does not accept manual tuning flags for variant, epsilon, dataset, etc.;
  these must be specified in the Modulation config of the template.

Before starting a job in the UI:

1. Create a Model Template (via the Create Model page):
   - Source: Custom or External
   - Provider: Hugging Face or custom repo
   - Repository ID: <your-rl-model-repo>
   - Model Kind: rl
   - Register immediately (so a provenance certificate is issued)

2. Go to the Modulator page for this template:
   - Variant: additive or multiplicative
   - Hyperparameters: epsilon (ε), lambda0 (λ₀), phi (activation), ecm_init
   - MaxSteps: number of episodes to evaluate
   - KPI Targets: enable RL KPIs (reward, success_rate)
"""


import os, sys, json, datetime, argparse
from ephapsys.modulation import ModulatorClient

GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", type=str, default=os.getenv("BASE_URL", "http://localhost:7001"))
    parser.add_argument("--api_key", type=str, default=os.getenv("API_KEY", os.getenv("AOC_BOOTSTRAP_TOKEN", "")))
    parser.add_argument("--model_template_id", type=str, required=True)   # <- still required
    parser.add_argument("--outdir", type=str, default="./out")
    args = parser.parse_args()

    if not args.api_key:
        raise RuntimeError("API token missing. Provide --api_key or set API_KEY/AOC_BOOTSTRAP_TOKEN in the environment")


    # --- Create base outdir + timestamped run subdir ---
    os.makedirs(args.outdir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.outdir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"[INFO] Run directory created: {run_dir}")

    # --- Setup client + wait for job ---
    mc = ModulatorClient(args.base_url, args.api_key)
    tpl, job_id = mc.wait_for_job_id(args.model_template_id)
    recipe = tpl.get("DesiredModulation") or {}

    # --- Extract config from recipe ---
    mode = recipe.get("mode") or "manual"
    variant = recipe.get("variant")
    episodes = int((recipe.get("kpi") or {}).get("maxSteps") or 10)
    if not variant:
        raise ValueError("Trainer requires 'variant' in recipe (additive or multiplicative).")

    print("=== JOB CONFIG FROM BACKEND ===")
    print(f"Job ID:      {job_id}")
    print(f"Mode:        {mode}")
    print(f"Variant:     {variant}")
    print(f"Episodes:    {episodes}")
    print(f"Run Dir:     {run_dir}")
    print("================================")

    summary = {
        "job_id": job_id,
        "mode": mode,
        "variant": variant,
        "episodes": episodes,
        "run_dir": run_dir,
    }

    # --- Run evaluation with streaming metrics ---
    last = None
    for update in mc.compute_rl_metrics_stream(
        args.model_template_id,
        episodes=episodes
    ):
        last = update

    print(f"{GREEN}Final aggregated metrics: {last}{RESET}")

    # --- Report back to backend ---
    mc.finalize_and_certify(
        run_dir,
        None,       # RL may not have a Hugging Face model artifact
        None,       # no tokenizer/processor needed
        last,
        variant,
        job_id,
        args.model_template_id
    )
    print(f"{GREEN}Reported metrics to backend and certified results.{RESET}")

    # --- Always write summary.json ---
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Summary written to {summary_path}")


if __name__ == "__main__":
    main()
