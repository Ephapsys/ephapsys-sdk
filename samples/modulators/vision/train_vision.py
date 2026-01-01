#!/usr/bin/env python3
"""
Trainer script for Vision models (YOLOS) with ephaptic coupling integration.

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
   - Repository ID: hustvl/yolos-base
   - Model Kind: vision
   - Revision: main
   - Hugging Face Token: hf_xxxxxxxx
   - Register immediately (so a provenance certificate is issued)

2. Go to the Modulator page for this template:
   - Variant: additive or multiplicative
   - Hyperparameters: epsilon (ε), lambda0 (λ₀), phi (activation), ecm_init
   - MaxSteps: number of samples/steps to evaluate
   - Dataset: name (e.g., cifar10, imagenet-1k), split (e.g., test[:1%], train[:1000])
   - KPI Targets: enable at least one KPI relevant to Vision (e.g., Accuracy, FID)
"""

import os, sys, json, datetime, argparse
import torch
from transformers import AutoImageProcessor, YolosForObjectDetection

from ephapsys.modulation import ModulatorClient

GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", type=str, default=os.getenv("BASE_URL", "http://localhost:7001"))
    parser.add_argument("--api_key", type=str, default=os.getenv("API_KEY", os.getenv("AOC_API_KEY", "")))
    parser.add_argument("--model_template_id", type=str, required=True)   # <- still required
    parser.add_argument("--outdir", type=str, default="./out")
    parser.add_argument("--auto_start", type=int, default=int(os.getenv("AUTO_START", "1")),
        help="1=auto-call /modulation/start (default), 0=manual mode (UI must start job)")
    args = parser.parse_args()

    if not args.api_key:
        raise RuntimeError("API key missing. Provide --api_key or set API_KEY/AOC_API_KEY in the environment")

    # --- Create base outdir + timestamped run subdir ---
    os.makedirs(args.outdir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.outdir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"[INFO] Run directory created: {run_dir}")

    # --- Setup client ---
    mc = ModulatorClient(args.base_url, args.api_key)

    if args.auto_start:
        print("[INFO] Auto-starting modulation job with full config...")

        # --- Check for existing running job ---
        tpl_existing = mc.get_template_or_die(args.model_template_id)
        mod = tpl_existing.get("Modulation") or {}

        print(f"[DEBUG] ---> Current Modulation state: {json.dumps(mod, indent=2)}") 

        if mod.get("status") == "running":
            old_job = mod.get("job_id")
            print(f"[WARN] Previous job still running (job_id={old_job}). Stopping it first...")
            try:
                mc.stop_job(job_id=old_job, model_template_id=args.model_template_id)
                print("---> 1  [INFO] Stopped previous job successfully. ")

                tpl_existing = mc.get_template_or_die(args.model_template_id)
                print("---> 2 [INFO] Stopped previous job successfully.")
                mod = tpl_existing.get("Modulation")

                print(f"[DEBUG] ---> New Modulation state after stopping: {json.dumps(mod, indent=2)}") 
            except Exception as e:
                print(f"[WARN] Failed to stop old job cleanly: {e}")
                exit(1)

        # --- Define dataset, KPIs, search config ---
        dataset = {
            "kind": "repo",
            "source": "external",
            "name": "cifar10",
            "config": "plain_text",
            "split": "train[:1%]",
        }
        kpi = {
            "targets": [
                {"name": "mAP", "direction": "max", "weight": 1},
                {"name": "recall", "direction": "max", "weight": 1},
            ],
            "maxSteps": 10,
        }
        search = {
            "algo": "bayes",
            "budget": 1,
            "parallel": 1,
            "multi_objective": True,
            "space": {
                "epsilon": {"low": 0.0, "high": 2.0},
                "lambda0": {"low": 0.0, "high": 0.5},
                "phi": ["identity", "relu", "tanh", "silu", "gelu"],
                "ecm_init": ["transpose", "identity", "random"],
                "variant": ["additive", "multiplicative"],
            },
        }

        # --- Start a fresh job ---
        mc.start_job(
            args.model_template_id,
            variant="additive",
            kpi=kpi,
            mode="auto",
            dataset=dataset,
            search=search,
        )
    else:
        print("[INFO] AUTO_START=0 → skipping /modulation/start, waiting for UI job...")

    # --- Block until job_id is available ---
    tpl, job_id = mc.wait_for_job_id(args.model_template_id)
    recipe = tpl.get("DesiredModulation") or {}

    # --- Download model snapshot into run_dir ---
    local_model_dir = mc.download_and_extract_model(args.model_template_id, run_dir)

    # --- Load YOLOS from local snapshot ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = recipe.get("SourceRepo") or "hustvl/yolos-base"
    processor = AutoImageProcessor.from_pretrained(local_model_dir, local_files_only=True)
    model = YolosForObjectDetection.from_pretrained(local_model_dir, local_files_only=True).to(device)

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

    # --- Run evaluation with streaming metrics ---
    last = None
    for update in mc.compute_vision_metrics_stream(
        model,
        processor,
        args.model_template_id,
        ds_name=ds_name,
        ds_config=ds_config,
        ds_split=ds_split,
        steps=steps,
        provider="huggingface",
        provider_token="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"
    ):
        last = update

    print(f"{GREEN}Final aggregated metrics: {last}{RESET}")

    # --- Report back to backend ---
    mc.finalize_and_certify(
        run_dir,
        model,
        processor,
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
