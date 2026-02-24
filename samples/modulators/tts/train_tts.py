#!/usr/bin/env python3
"""
Trainer script for TTS with ephaptic coupling integration.

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
   - Repository ID: microsoft/speecht5_tts
   - Model Kind: TTS
   - Revision: main
   - Hugging Face Token: hf_xxxxxxxx
   - Register immediately (so a provenance certificate is issued)

2. Go to the Modulator page for this template:
   - Variant: additive or multiplicative
   - Hyperparameters: epsilon (ε), lambda0 (λ₀), phi (activation), ecm_init
   - MaxSteps: number of samples/steps to evaluate
   - Dataset: name (e.g., librispeech_asr), config (e.g., clean), split (e.g., validation[:1%])
   - KPI Targets: enable at least one KPI relevant to TTS (e.g., WER, MOS)
"""

import os, sys, json, datetime
import argparse
import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech

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
    parser.add_argument("--base_url", type=str, default=os.getenv("AOC_BASE_URL", os.getenv("BASE_URL", "http://localhost:7001")))
    parser.add_argument("--api_key", type=str, default=os.getenv("AOC_MODULATION_TOKEN", ""))
    parser.add_argument("--model_template_id", type=str, required=True)   # <- still required
    parser.add_argument("--outdir", type=str, default="./out")
    parser.add_argument("--auto_start", type=int, default=int(os.getenv("AUTO_START", "1")),
        help="1=auto-call /modulation/start (default), 0=manual mode (UI must start job)")
    args = parser.parse_args()

    if not args.api_key:
        raise RuntimeError("API token missing. Provide --api_key or set AOC_MODULATION_TOKEN in the environment")

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
            "name": "librispeech_asr",
            "config": "clean",
            "split": "validation[:1%]",
        }
        kpi = {
            "targets": [
                {"name": "wer", "direction": "min", "weight": 1},
                {"name": "mos", "direction": "max", "weight": 1},
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
    recipe = tpl.get("DesiredModulation")

    # --- Download model snapshot into run_dir ---
    local_model_dir = mc.download_and_extract_model(args.model_template_id, run_dir)

    # --- Load SpeechT5 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = SpeechT5Processor.from_pretrained(local_model_dir, local_files_only=True)
    model = SpeechT5ForTextToSpeech.from_pretrained(local_model_dir, local_files_only=True).to(device)

    # --- Extract config ---
    mode = recipe.get("mode") or "manual"
    variant = recipe.get("variant")
    steps = int((recipe.get("kpi") or {}).get("maxSteps"))
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
        # -------- Manual run (single trial) --------
        print("[INFO] Running in manual mode")
        trial_cfg = {
            "variant": variant,
            "epsilon": recipe.get("epsilon"),
            "lambda0": recipe.get("lambda0"),
            "phi": recipe.get("phi"),
            "ecm_init": recipe.get("ecm_init"),
            "maxSteps": steps,
        }
        mc.inject_ecm_from_trial(job_id, model.speecht5.encoder,
                                 hidden_dim=model.config.hidden_size,
                                 last_cfg=trial_cfg, last_score=None)
        print(f"[TRIAL] Manual config → {trial_cfg}")

        all_metrics, last = [], None
        for last in mc.compute_tts_metrics_stream(
            model, processor, args.model_template_id,
            ds_name=ds_name, ds_config=ds_config, ds_split=ds_split, steps=steps
        ):
            all_metrics.append(last)
            score = last["mos"] - last["wer"]
            color = GREEN if score >= (summary.get("best_score") or -999) else YELLOW
            bar = "#" * int(30 * last["step"] / steps) + "-" * (30 - int(30 * last["step"] / steps))
            sys.stdout.write(
                f"\r[STEP {last['step']}/{steps}] {color}[{bar}]{RESET} "
                f"WER={last['wer']:.3f}, MOS={last['mos']:.2f}, score={score:.3f}"
            )
            sys.stdout.flush()
            summary.update(best_score=score, best_metrics=last, best_variant=trial_cfg)
        print()  # newline

        print(f"[RESULT] Manual run metrics: {last}")
        mc.finalize_and_certify(
            run_dir, model, processor, last, trial_cfg["variant"], job_id,
            args.model_template_id, all_metrics=all_metrics
        )
        print(f"[INFO] Reports saved under: {run_dir}")
        print("[DONE] Manual mode finished successfully.")
    else:
        # -------- Auto mode (multi-trial loop) --------
        print("[INFO] Running in auto mode")
        best_score, best_metrics, best_variant, best_all_metrics = None, None, None, None
        last_cfg, last_score, trial_num = None, None, 0
        budget = int((recipe.get("search") or {}).get("budget", 0) or 20)

        while True:
            trial_cfg = mc.inject_ecm_from_trial(job_id, model.speecht5.encoder,
                                                 hidden_dim=model.config.hidden_size,
                                                 last_cfg=last_cfg, last_score=last_score)
            if not trial_cfg: break
            trial_num += 1
            print(f"\n[TRIAL {trial_num}/{budget}] Config → {trial_cfg}")

            all_metrics, last = [], None
            for last in mc.compute_tts_metrics_stream(
                model, processor, args.model_template_id,
                ds_name=ds_name, ds_config=ds_config, ds_split=ds_split,
                steps=int(trial_cfg.get("maxSteps") or steps)
            ):
                all_metrics.append(last)
                score = last["mos"] - last["wer"]
                color = GREEN if (best_score is None or score > best_score) else YELLOW
                bar = "#" * int(30 * last["step"] / steps) + "-" * (30 - int(30 * last["step"] / steps))
                sys.stdout.write(
                    f"\r[STEP {last['step']}/{steps}] {color}[{bar}]{RESET} "
                    f"WER={last['wer']:.3f}, MOS={last['mos']:.2f}, score={score:.3f}"
                )
                sys.stdout.flush()
            print()  # newline

            score = last["mos"] - last["wer"]
            last_cfg, last_score = trial_cfg, score
            print(f"[RESULT] Trial {trial_num}/{budget} score={score:.3f}, metrics={last}")

            if best_score is None or score > best_score:
                best_score, best_metrics, best_variant, best_all_metrics = score, last, trial_cfg, list(all_metrics)
                print(f"{GREEN}[BEST] Updated best score={best_score:.3f}, config={best_variant}{RESET}")

        if best_metrics:
            mc.finalize_and_certify(
                run_dir, model, processor, best_metrics,
                best_variant.get("variant"), job_id, args.model_template_id,
                all_metrics=best_all_metrics
            )
            summary.update(best_score=best_score, best_metrics=best_metrics, best_variant=best_variant)
            print(f"[INFO] Reports saved under: {run_dir}")
            print(f"{GREEN}[DONE] Best trial finalized with score={best_score:.3f}, metrics={best_metrics}{RESET}")
        else:
            print("[WARN] No valid trials executed in auto mode.")

    # --- Always write summary.json ---
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w") as f: json.dump(summary, f, indent=2)
    print(f"[INFO] Summary written to {summary_path}")

if __name__ == "__main__":
    main()
