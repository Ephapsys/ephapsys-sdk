#!/usr/bin/env python3
"""
Trainer script for Audio models with ephaptic coupling integration.

Usage flow:
- Minimal CLI args: --base_url, --api_key, --model_template_id, --outdir
- All training hyperparameters, dataset config, and model_id are fetched dynamically
  from the backend template created in the UI.
- The trainer does not accept manual tuning flags for variant, epsilon, dataset split, etc.;
  these must be specified in the Modulation config of the template.

Before starting a job in the UI:

1. Create a Model Template (via the Create Model page):
   - Source: External repository
   - Provider: Hugging Face
   - Repository ID: superb/wav2vec2-base-superb-ks
   - Model Kind: audio
   - Revision: main
   - Hugging Face Token: hf_xxxxxxxx
   - Register immediately (so a provenance certificate is issued)

2. Go to the Modulator page for this template:
   - Variant: additive or multiplicative
   - Hyperparameters: epsilon (ε), lambda0 (λ₀), phi (activation), ecm_init
   - MaxSteps: number of samples/steps to evaluate
   - Dataset: name (e.g., superb, config=ks), split (e.g., test[:1%])
   - KPI Targets: enable audio KPI (Accuracy)
"""

import os, sys, json, datetime, argparse
import torch
from transformers import AutoProcessor, AutoModelForAudioClassification
from datasets import load_dataset
import evaluate

from ephapsys.modulation import ModulatorClient

# ------------------------------
# ANSI colors
# ------------------------------
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"


# ------------------------------
# Main entry
# ------------------------------
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

    # --- Download model snapshot into run_dir ---
    local_model_dir = mc.download_and_extract_model(args.model_template_id, run_dir)

    # --- Load Audio classification model from local snapshot ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = recipe.get("SourceRepo") or "superb/wav2vec2-base-superb-ks"
    processor = AutoProcessor.from_pretrained(local_model_dir, local_files_only=True)
    model = AutoModelForAudioClassification.from_pretrained(local_model_dir, local_files_only=True).to(device)

    # --- Extract config from recipe ---
    mode = recipe.get("mode") or "manual"
    variant = recipe.get("variant")
    steps = int((recipe.get("kpi") or {}).get("maxSteps") or 0)
    dataset_cfg = recipe.get("dataset", {})
    ds_name, ds_config, ds_split = dataset_cfg.get("name"), dataset_cfg.get("config"), dataset_cfg.get("split")

    if not variant:
        raise ValueError("Trainer requires 'variant' in recipe (additive or multiplicative).")

    print("=== JOB CONFIG FROM BACKEND ===")
    print(f"Job ID:      {job_id}")
    print(f"Mode:        {mode}")
    print(f"Variant:     {variant}")
    print(f"Steps:       {steps}")
    print(f"Dataset:     {ds_name}/{ds_config}/{ds_split}")
    print(f"Run Dir:     {run_dir}")
    print("================================")

    summary = {
        "job_id": job_id,
        "mode": mode,
        "variant": variant,
        "dataset": f"{ds_name}/{ds_config}/{ds_split}",
        "steps": steps,
        "run_dir": run_dir,
    }

    # --- Dataset load ---
    print(f"{YELLOW}Loading dataset: {ds_name}/{ds_config}/{ds_split}{RESET}")
    ds = load_dataset(ds_name, ds_config, split=ds_split)

    acc_metric = evaluate.load("accuracy")

    all_metrics = []
    last = None
    for i, sample in enumerate(ds):
        audio = sample["audio"]
        inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1).item()

        acc = acc_metric.compute(predictions=[pred], references=[sample["label"]])
        last = {"step": i + 1, "accuracy": acc["accuracy"]}
        all_metrics.append(last)

        if steps and (i + 1) >= steps:
            break

    # Aggregate metrics
    if all_metrics:
        agg = {"accuracy": sum(d["accuracy"] for d in all_metrics) / len(all_metrics)}
    else:
        agg = {"accuracy": 0.0}
    print(f"{GREEN}Aggregated metrics: {agg}{RESET}")

    # --- Report back to backend ---
    mc.finalize_and_certify(
        run_dir,
        model,
        processor,
        agg,
        variant,
        job_id,
        args.model_template_id,
        all_metrics=all_metrics
    )
    print(f"{GREEN}Reported metrics to backend and certified results.{RESET}")

    # --- Always write summary.json ---
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Summary written to {summary_path}")


if __name__ == "__main__":
    main()
