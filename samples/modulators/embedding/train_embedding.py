#!/usr/bin/env python3
"""
Trainer script for Embedding models (e.g., google/embeddinggemma-300m) with ephaptic coupling integration.

This trainer streams per-step metrics (cosine similarity, recall@k) back to the AOC,
so the frontend UI can render live charts during evaluation.

Usage flow:
- Minimal CLI args: --base_url, --api_key, --model_template_id, --outdir
- All training hyperparameters, dataset config, and model_id are fetched dynamically
  from the backend template created in the UI.
- The trainer does not accept manual tuning flags; these must be specified in the Modulation config.

Before starting a job in the UI:

1. Create a Model Template (via the Create Model page):
   - Source: External repository
   - Provider: Hugging Face
   - Repository ID: google/embeddinggemma-300m
   - Model Kind: embedding
   - Revision: main
   - Hugging Face Token: hf_xxxxxxxx
   - Register immediately (so a provenance certificate is issued)

2. Go to the Modulator page for this template:
   - Variant: additive or multiplicative
   - Hyperparameters: epsilon (ε), lambda0 (λ₀), phi (activation), ecm_init
   - MaxSteps: number of samples/steps to evaluate
   - KPI Targets: enable at least one KPI relevant to Embeddings (cosine_sim, recall_at_k)
"""

import os, sys, json, datetime, argparse
import torch
from transformers import AutoTokenizer, AutoModel

from ephapsys.modulation import ModulatorClient

GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", type=str, default=os.getenv("BASE_URL", "http://localhost:7001"))
    parser.add_argument("--api_key", type=str, default=os.getenv("API_KEY", os.getenv("AOC_BOOTSTRAP_TOKEN", "")))
    parser.add_argument("--model_template_id", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="./out")
    parser.add_argument("--auto_start", type=int, default=int(os.getenv("AUTO_START", "1")),
        help="1=auto-call /modulation/start (default), 0=manual mode (UI must start job)")
    args = parser.parse_args()

    if not args.api_key:
        raise RuntimeError("API token missing. Provide --api_key or set API_KEY/AOC_BOOTSTRAP_TOKEN in the environment")

    os.makedirs(args.outdir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.outdir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"[INFO] Run directory created: {run_dir}")

    mc = ModulatorClient(args.base_url, args.api_key)

    if args.auto_start:
        print("[INFO] Auto-starting modulation job with full config...")
        tpl_existing = mc.get_template_or_die(args.model_template_id)
        mod = tpl_existing.get("Modulation") or {}
        print(f"[DEBUG] ---> Current Modulation state: {json.dumps(mod, indent=2)}")

        if mod.get("status") == "running":
            old_job = mod.get("job_id")
            print(f"[WARN] Previous job still running (job_id={old_job}). Stopping it first...")
            try:
                mc.stop_job(job_id=old_job, model_template_id=args.model_template_id)
                tpl_existing = mc.get_template_or_die(args.model_template_id)
                mod = tpl_existing.get("Modulation")
            except Exception as e:
                print(f"[WARN] Failed to stop old job cleanly: {e}")
                exit(1)

        dataset = {
            "kind": "repo",
            "source": "external",
            "name": "sentence-transformers/all-nli",
            "config": "pair-score",
            "split": "train[:1%]",
        }
        kpi = {
            "targets": [
                {"name": "cosine_sim", "direction": "max", "weight": 1},
                {"name": "recall_at_k", "direction": "max", "weight": 1},
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

    tpl, job_id = mc.wait_for_job_id(args.model_template_id)
    recipe = tpl.get("DesiredModulation") or {}

    local_model_dir = mc.download_and_extract_model(args.model_template_id, run_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = recipe.get("SourceRepo") or "google/embeddinggemma-300m"
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(local_model_dir, local_files_only=True).to(device)

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

    if mode == "manual":
        print("[INFO] Running in manual mode")
        trial_cfg = {
            "variant": variant,
            "epsilon": recipe.get("epsilon"),
            "lambda0": recipe.get("lambda0"),
            "phi": recipe.get("phi"),
            "ecm_init": recipe.get("ecm_init"),
            "maxSteps": steps,
        }
        mc.inject_ecm_from_trial(
            job_id, model,
            hidden_dim=model.config.hidden_size,
            last_cfg=trial_cfg, last_score=None
        )

        last = None
        for update in mc.compute_embedding_metrics_stream(
            model, tokenizer, args.model_template_id,
            ds_name=ds_name, ds_config=ds_config, ds_split=ds_split, steps=steps
        ):
            last = update

        if not last:
            print("[WARN] No metrics were produced for this run — assigning defaults.")
            last = {"cosine_sim": 0.0, "recall_at_k": 0.0}

        mc.finalize_and_certify(
            run_dir, model, tokenizer,
            last, trial_cfg["variant"],
            job_id, args.model_template_id
        )
        print(f"[DONE] Manual mode finished successfully.")

    else:
        print("[INFO] Running in auto mode")
        best_score, best_metrics, best_variant = None, None, None
        last_cfg, last_score = None, None
        trial_num = 0
        budget = int((recipe.get("search") or {}).get("budget", 0) or 20)

        while trial_num < budget:
            trial_cfg = mc.inject_ecm_from_trial(
                job_id, model,
                hidden_dim=model.config.hidden_size,
                last_cfg=last_cfg, last_score=last_score
            )
            if not trial_cfg:
                print("\n[INFO] No more trials. Auto mode loop finished early.")
                break

            trial_num += 1
            print(f"\n[TRIAL {trial_num}/{budget}] Config → {trial_cfg}")

            last = None
            for update in mc.compute_embedding_metrics_stream(
                model, tokenizer, args.model_template_id,
                ds_name=ds_name, ds_config=ds_config, 
                ds_split=ds_split, 
                steps=steps,
                provider="huggingface",
                provider_token="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"
            ):
                last = update

            if not last:
                print("[WARN] No metrics were produced in this trial — skipping.")
                continue

            score = last["cosine_sim"]  # heuristic
            last_cfg, last_score = trial_cfg, score
            print(f"[RESULT] Trial {trial_num}/{budget} score={score:.3f}, metrics={last}")

            if best_score is None or score > best_score:
                best_score, best_metrics, best_variant = score, last, trial_cfg
                print(f"{GREEN}[BEST] Updated best score={best_score:.3f}, config={best_variant}{RESET}")

        if best_metrics:
            mc.finalize_and_certify(
                run_dir, model, tokenizer,
                best_metrics, best_variant.get("variant"),
                job_id, args.model_template_id
            )
            summary["best_score"] = best_score
            summary["best_metrics"] = best_metrics
            summary["best_variant"] = best_variant
            print(f"{GREEN}[DONE] Best trial finalized with score={best_score:.3f}{RESET}")
        else:
            print("[WARN] No valid trials executed in auto mode.")

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Summary written to {os.path.join(run_dir, 'summary.json')}")

if __name__ == "__main__":
    main()
