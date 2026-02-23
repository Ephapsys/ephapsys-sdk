#!/usr/bin/env python3
"""
Trainer script for Language models (e.g., Flan-T5) with ephaptic coupling integration.

This trainer streams per-step metrics (accuracy, loss, perplexity) back to the AOC,
so the frontend UI can render live charts during evaluation.
It supports both manual and auto modes:

- Manual mode: inject ECM once with config from the Modulator page, then evaluate.
- Auto mode: run multiple trials with configs suggested by the backend (Bayesian search),
  scoring each trial and finalizing the best one.

Usage flow:
- Minimal CLI args: --base_url, --api_key, --model_template_id, --outdir
- All training hyperparameters, dataset config, and model_id are fetched dynamically
  from the backend template created in the UI.
- The trainer does not accept manual tuning flags; these must be specified in the Modulation config.

Before starting a job in the UI:

1. Create a Model Template (via the Create Model page):
   - Source: External repository
   - Provider: Hugging Face
   - Repository ID: google/flan-t5-small, google/gemma-3-270m, openai-community/gpt2
   - Model Kind: language
   - Revision: main
   - Hugging Face Token: hf_xxxxxxxx
   - Register immediately (so a provenance certificate is issued)

2. Go to the Modulator page for this template:
   - Variant: additive or multiplicative
   - Hyperparameters: epsilon (ε), lambda0 (λ₀), phi (activation), ecm_init
   - MaxSteps: number of samples/steps to evaluate
   - Dataset: name (e.g., wiki), config (e.g., wikitext-103-raw-v1), split (e.g., train[:1%])
   - KPI Targets: enable at least one KPI relevant to Language (accuracy, loss, perplexity)
"""

import os, sys, json, datetime, argparse, time
import math
from ephapsys.modulation import ModulatorClient

GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def evaluate_baseline(mc, model, tokenizer, model_template_id, ds_name, ds_config, ds_split, steps):
    """Run a baseline (unmodulated) evaluation for comparison."""
    print(f"{YELLOW}[BASELINE] Running standard evaluation (no ECM injected)...{RESET}")
    baseline_stream = []

    # === Use full compute_language_metrics_stream (includes ROUGE/BLEU/BERTScore) ===
    for update in mc.compute_language_metrics_stream(
        model, tokenizer, model_template_id,
        ds_name=ds_name, ds_config=ds_config, ds_split=ds_split, steps=steps
    ):
        baseline_stream.append(update)

    baseline = baseline_stream[-1] if baseline_stream else {}
    print(f"{YELLOW}[BASELINE] Results: {baseline}{RESET}")

    # === Upload baseline to backend so AOC baseline matches DOCX & UI ===
    try:
        print(f"[BASELINE] Uploading baseline metrics for {model_template_id} to AOC...")

        # Dynamically detect all numeric KPI keys (auto-expands for ROUGE/BLEU/BERTScore)
        metric_keys = tuple(
            k for k, v in baseline.items()
            if isinstance(v, (int, float)) and k not in ("step", "total")
        )

        if not metric_keys:
            metric_keys = (
                "accuracy", "loss", "perplexity",
                "rouge1", "rouge2", "rougeL", "bleu", "bertscore_f1"
            )

        mc.upload_baseline_metrics(
            model_template_id,
            baseline_stream,
            kpis=metric_keys,
        )

        # --- Trigger immediate baseline re-emit so frontend refreshes dashed lines ---
        import requests
        resp = requests.post(
            f"{mc.base_url}/modulation/baseline_emit",
            headers={"Authorization": f"Bearer {mc.api_key}"},
            json={"model_template_id": model_template_id},
            timeout=10,
        )
        if resp.ok:
            print(f"[BASELINE] Baseline (Standard) re-emitted for {model_template_id}")
        else:
            print(f"[WARN] Baseline re-emit failed: {resp.status_code} {resp.text}")

        # --- Log summary of uploaded KPI keys ---
        uploaded_keys = [
            k for k in baseline.keys()
            if isinstance(baseline.get(k), (int, float))
        ]
        print(f"[BASELINE] Uploaded baseline curves: {uploaded_keys}")

    except Exception as e:
        print(f"[WARN] Baseline upload error: {e}")
    # ================================================================

    return baseline, baseline_stream

# Inspect the current ephaptic coupling matrix (Λ)
def inspect_lambda(model, label="Λ"):
    """Print diagnostics for the ephaptic coupling matrix if it exists."""
    import torch

    for name, p in model.named_parameters():
        if "lambda_ecm" in name:
            with torch.no_grad():
                norm = torch.linalg.norm(p).item()
                minv, maxv, meanv = p.min().item(), p.max().item(), p.mean().item()
                print(f"[{label}] Norm={norm:.6f}, min={minv:.6f}, max={maxv:.6f}, mean={meanv:.6f}")
            return p.detach().clone()
    print(f"[WARN] No ephaptic Λ found in model during {label} inspection.")
    return None

def main():
    import torch
    from transformers import (
        AutoConfig,
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        AutoModelForCausalLM,
    )

    best_score, best_metrics, best_variant, best_stream = None, {}, {"variant": "additive"}, []
    start_time = time.time()  # Track total runtime
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", type=str, default=os.getenv("BASE_URL", "http://localhost:7001"))
    parser.add_argument("--api_key", type=str, default=os.getenv("API_KEY", os.getenv("AOC_BOOTSTRAP_TOKEN", "")))
    parser.add_argument("--model_template_id", type=str, required=True)   # <- still required
    parser.add_argument("--outdir", type=str, default="./out")
    # --- Train is an option/flag (not a mode) ---
    parser.add_argument("--train", action="store_true", help="Enable gradient updates during per-step loop")
    # Backward compatibility with old flag name
    parser.add_argument("--train_mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--auto_start", type=int, default=int(os.getenv("AUTO_START", "1")),
        help="1=auto-call /modulation/start (default), 0=manual mode (UI must start job)")
    args = parser.parse_args()
    # Merge old flag into the new one
    args.train = bool(args.train or args.train_mode)

    if not args.api_key:
        raise RuntimeError("API token missing. Provide --api_key or set API_KEY/AOC_BOOTSTRAP_TOKEN in the environment")

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
            "name": "wikitext",
            "config": "wikitext-103-raw-v1",
            "split": "train[:1%]",
        }

        kpi = {
            "targets": [
                {"name": "accuracy", "direction": "max", "weight": 1},
                {"name": "loss", "direction": "min", "weight": 1},
                {"name": "perplexity", "direction": "min", "weight": 1},
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
        # NOTE: mode is an AOC/UI concept (manual vs auto). Training is now a *flag* here, not a mode.
        mc.start_job(
            args.model_template_id,
            variant="additive",
            kpi=kpi,
            mode="auto",  # carry search behavior; training is controlled locally by --train
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

    # --- Load model (Seq2Seq or Causal) from local snapshot ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = recipe.get("SourceRepo") 

    config = AutoConfig.from_pretrained(local_model_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir, local_files_only=True)

    # Detect model type
    if config.model_type in ["t5", "bart", "mbart", "pegasus", "mt5"]:
        print(f"[INFO] Detected Seq2Seq model: {config.model_type}")
        model = AutoModelForSeq2SeqLM.from_pretrained(local_model_dir, local_files_only=True).to(device)
        encoder = model.get_encoder()
        is_seq2seq = True
    else:
        print(f"[INFO] Detected Causal LM model: {config.model_type}")
        model = AutoModelForCausalLM.from_pretrained(local_model_dir, local_files_only=True).to(device)
        encoder = model  # causal LMs have no separate encoder
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        is_seq2seq = False

    # Clone a baseline copy before ECM injection
    from copy import deepcopy
    baseline_model = deepcopy(model)

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

    # --- Initialize baseline placeholders to avoid UnboundLocalError ---
    baseline_metrics, baseline_stream = {}, []

    # =========================
    #  Base baseline (for both manual and auto flows)
    # =========================
    baseline_metrics, baseline_stream = evaluate_baseline(
        mc, baseline_model, tokenizer, args.model_template_id, ds_name, ds_config, ds_split, steps
    )
    summary = {
        "job_id": job_id,
        "mode": mode,
        "variant": variant,
        "dataset": f"{ds_name}/{ds_config}/{ds_split}",
        "steps": steps,
        "run_dir": run_dir,
    }

    # Helper to build training dataset when training is enabled
    def build_training_ds(ds_name, ds_config, ds_split):
        from datasets import load_dataset
        ds = load_dataset(ds_name, ds_config, split=ds_split)
        def sample_at(i):
            return ds[int(i % len(ds))]
        return ds, sample_at

    # =========================
    # MANUAL MODE
    # =========================
    if mode == "manual":
        print("[INFO] Running in manual mode")

        # Inject ECM once with config from recipe
        trial_cfg = {
            "variant": variant,
            "epsilon": recipe.get("epsilon"),
            "lambda0": recipe.get("lambda0"),
            "phi": recipe.get("phi"),
            "ecm_init": recipe.get("ecm_init"),
            "maxSteps": steps,
        }

        trial_cfg = mc.inject_ecm_from_trial(job_id, encoder, last_cfg=trial_cfg, last_score=None)
        if not trial_cfg:
            raise RuntimeError("[MANUAL] inject_ecm_from_trial returned no config")

        # Inspect Λ before modulation/training
        lambda_before = inspect_lambda(model, label="Λ (before)")

        metrics_stream = []

        if args.train:
            # --- Optimizer + (optional) loss (seq2seq uses model.loss; causal uses CE through labels) ---
            print("[TRAIN] Training enabled in manual mode — running gradient updates per step.")
            ds, sample_at = build_training_ds(ds_name, ds_config, ds_split)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            model.train()

            for step_idx in range(steps):
                sample = sample_at(step_idx)
                text = (sample.get("text") or "").strip() or "Hello world."

                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128,
                    padding=True
                ).to(device)

                if is_seq2seq:
                    labels = inputs["input_ids"]  # define labels first
                    outputs = model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        labels=labels
                    )

                    loss = outputs.loss
                    logits = outputs.logits
                    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100
                    mask = (labels != pad_id)
                    preds = logits.argmax(dim=-1)
                    correct = (preds.eq(labels) & mask).sum().item()
                    total = mask.sum().item()

                else:
                    labels = inputs["input_ids"]
                    outputs = model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        labels=labels
                    )

                    loss = outputs.loss
                    logits = outputs.logits
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    with torch.no_grad():
                        pred_ids = shift_logits.argmax(dim=-1)
                        mask = (shift_labels != (tokenizer.pad_token_id or -100))
                        correct = (pred_ids.eq(shift_labels) & mask).sum().item()
                        total = mask.sum().item()

                acc = (correct / total) if total > 0 else 0.0
                ppl = math.exp(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                metric = {
                    "step": step_idx + 1,
                    "total": steps,
                    "accuracy": float(acc),
                    "loss": float(loss.item()),
                    "perplexity": float(ppl),
                }
                metrics_stream.append(metric)

                # Live inline progress (colored, same line)
                acc_color = GREEN if acc >= 0.5 else YELLOW
                loss_color = "\033[91m" if loss.item() > 5 else GREEN  # red if high loss
                sys.stdout.write(
                    f"\r{YELLOW}[MANUAL]{RESET} {GREEN}Step {step_idx + 1:03d}/{steps}{RESET} "
                    f"| acc={acc_color}{acc:.4f}{RESET} | loss={loss_color}{loss.item():.4f}{RESET} | ppl={ppl:.2f}   "
                )
                sys.stdout.flush()


                # newline only after final step for clean formatting
                if step_idx + 1 == steps:
                    sys.stdout.write("\n")

                # stream live to AOC
                mc._report_model_metrics(
                    args.model_template_id,
                    {"accuracy": acc, "loss": loss.item(), "perplexity": ppl},
                    step=step_idx + 1,
                )

        else:
            # --- Run evaluation with streaming metrics (includes language-quality KPIs) ---
            for update in mc.compute_language_metrics_stream(
                model, tokenizer, args.model_template_id,
                ds_name=ds_name, ds_config=ds_config, ds_split=ds_split, steps=steps
            ):
                metrics_stream.append(update)
                step = update.get("step") or len(metrics_stream)
                acc = update.get("accuracy", 0)
                loss = update.get("loss", 0)
                ppl = update.get("perplexity", 0)

                # --- Report live core metrics ---
                mc._report_model_metrics(
                    args.model_template_id,
                    {"accuracy": acc, "loss": loss, "perplexity": ppl},
                    step=step,
                )

                # --- Report language-quality metrics in real time ---
                quality_metrics = {
                    k: update.get(k, 0.0)
                    for k in ("rouge1", "rouge2", "rougeL", "bleu", "bertscore_f1")
                    if k in update
                }
                if any(v != 0 for v in quality_metrics.values()):
                    mc._report_model_metrics(args.model_template_id, quality_metrics, step=step)

        # Inspect Λ after modulation/training
        lambda_after = inspect_lambda(model, label="Λ (after)")
        if lambda_before is not None and lambda_after is not None:
            delta = torch.linalg.norm(lambda_after - lambda_before).item()
            print(f"[ΔΛ] Frobenius difference: {delta:.6f}")

        last = metrics_stream[-1] if metrics_stream else {}
        print(f"[RESULT] Manual run metrics: {last}")

        total_runtime = time.time() - start_time
        summary["runtime_secs"] = round(total_runtime, 2)

        # --- Finalize with rich metrics stream ---
        print("[DIAGNOSTIC] Final Λ state before certification:")
        inspect_lambda(model, label="Λ (final)")

        # --- Preserve language-quality metrics from last stream (no extra eval) ---
        if metrics_stream and any(k in metrics_stream[-1] for k in ("rouge1", "rouge2", "rougeL", "bleu", "bertscore_f1")):
            last.update({
                k: metrics_stream[-1][k]
                for k in ("rouge1", "rouge2", "rougeL", "bleu", "bertscore_f1")
                if k in metrics_stream[-1]
            })
            print("[EVALUATE] Added language-quality metrics to final report:",
                  {k: last.get(k) for k in ("rouge1","rouge2","rougeL","bleu","bertscore_f1")})

        # --- Report final language-quality metrics so dashboard sees them ---
        for k in ("rouge1", "rouge2", "rougeL", "bleu", "bertscore_f1"):
            if k in last:
                mc._report_model_metrics(args.model_template_id, {k: last[k]}, step=steps)

        mc.finalize_and_certify(
            run_dir,
            model,
            tokenizer,
            last,
            trial_cfg["variant"],
            job_id,
            args.model_template_id,
            all_metrics=metrics_stream,  # ✅ include per-step data for PNG/CSV
            baseline_metrics=baseline_metrics,  # ✅ include comparison table
            exp_config={**trial_cfg, "runtime": total_runtime},  # ✅ pass ephaptic config + runtime
        )
        print(f"[INFO] Reports saved under: {run_dir}")
        print("[DONE] Manual mode finished successfully.")

    # =========================
    # AUTO MODE
    # =========================
    else:
        print("[INFO] Running in auto mode")
        best_score, best_metrics, best_variant, best_stream = None, None, None, None
        last_cfg, last_score = None, None
        trial_num = 0
        budget = int((recipe.get("search") or {}).get("budget", 0) or 20)

        while True:
            trial_cfg = mc.inject_ecm_from_trial(
                job_id, encoder,
                last_cfg=last_cfg, last_score=last_score
            )
            if not trial_cfg:
                print("\n[INFO] No more trials. Auto mode loop finished.")
                break

            trial_num += 1
            print(f"\n[TRIAL {trial_num}/{budget}] Config → {trial_cfg}")

            # For training-enabled trials, we can (optionally) isolate updates by copying the model.
            # This avoids cross-trial contamination of weights.
            model_trial = deepcopy(model) if args.train else model
            encoder_trial = model_trial.get_encoder() if is_seq2seq else model_trial

            # Inspect Λ before modulation/training
            lambda_before = inspect_lambda(model_trial, label=f"Λ (trial {trial_num} before)")

            metrics_stream = []

            if args.train:
                # --- Show how many trials we expect overall (once, before the first) ---
                if trial_num == 1:
                    print(f"{YELLOW}[INFO] Preparing to run {budget} ephaptic trials (auto mode){RESET}")

                print(f"[TRAIN] Training enabled for trial {trial_num} — running gradient updates per step.")
                ds, sample_at = build_training_ds(ds_name, ds_config, ds_split)
                optimizer = torch.optim.AdamW(model_trial.parameters(), lr=1e-4)
                model_trial.train()

                for step_idx in range(steps):
                    sample = sample_at(step_idx)
                    text = (sample.get("text") or "").strip() or "Hello world."

                    inputs = tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=128,
                        padding=True
                    ).to(device)

                    # --- forward + loss ---
                    if is_seq2seq:
                        outputs = model_trial(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            labels=inputs["input_ids"]
                        )
                        loss = outputs.loss
                        logits = outputs.logits
                        labels = inputs["input_ids"]
                        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100
                        mask = (labels != pad_id)
                        preds = logits.argmax(dim=-1)
                        correct = (preds.eq(labels) & mask).sum().item()
                        total = mask.sum().item()
                    else:
                        labels = inputs["input_ids"]
                        outputs = model_trial(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            labels=labels
                        )
                        loss = outputs.loss
                        logits = outputs.logits
                        shift_logits = logits[:, :-1, :].contiguous()
                        shift_labels = labels[:, 1:].contiguous()
                        with torch.no_grad():
                            pred_ids = shift_logits.argmax(dim=-1)
                            mask = (shift_labels != (tokenizer.pad_token_id or -100))
                            correct = (pred_ids.eq(shift_labels) & mask).sum().item()
                            total = mask.sum().item()

                    acc = (correct / total) if total > 0 else 0.0
                    ppl = math.exp(loss.item())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    metric = {
                        "step": step_idx + 1,
                        "total": steps,
                        "accuracy": float(acc),
                        "loss": float(loss.item()),
                        "perplexity": float(ppl),
                    }
                    metrics_stream.append(metric)

                    # Live inline colored progress (single line)
                    acc_color = GREEN if acc >= 0.5 else YELLOW
                    loss_color = "\033[91m" if loss.item() > 5 else GREEN
                    sys.stdout.write(
                        f"\r{YELLOW}[TRIAL {trial_num}/{budget}] {GREEN}Step {step_idx + 1:03d}/{steps}{RESET} "
                        f"| acc={acc_color}{acc:.4f}{RESET} | loss={loss_color}{loss.item():.4f}{RESET} | ppl={ppl:.2f}   "
                    )
                    sys.stdout.flush()

                    # newline only after the last step so the next print starts cleanly
                    if step_idx + 1 == steps:
                        sys.stdout.write("\n")

                    # stream live to AOC
                    mc._report_model_metrics(
                        args.model_template_id,
                        {"accuracy": acc, "loss": loss.item(), "perplexity": ppl},
                        step=step_idx + 1,
                    )

            else:
                # --- Run evaluation with streaming metrics (includes language-quality KPIs) ---
                for update in mc.compute_language_metrics_stream(
                    model_trial, tokenizer, args.model_template_id,
                    ds_name=ds_name, ds_config=ds_config, ds_split=ds_split, steps=steps
                ):
                    metrics_stream.append(update)
                    step = update.get("step") or len(metrics_stream)
                    acc = update.get("accuracy", 0)
                    loss = update.get("loss", 0)
                    ppl = update.get("perplexity", 0)

                    # --- Report live core metrics ---
                    mc._report_model_metrics(
                        args.model_template_id,
                        {"accuracy": acc, "loss": loss, "perplexity": ppl},
                        step=step,
                    )

                    # --- 🆕 Report language-quality metrics in real time ---
                    quality_metrics = {
                        k: update.get(k, 0.0)
                        for k in ("rouge1", "rouge2", "rougeL", "bleu", "bertscore_f1")
                        if k in update
                    }
                    if any(v != 0 for v in quality_metrics.values()):
                        mc._report_model_metrics(args.model_template_id, quality_metrics, step=step)

                    # Optional: log partial quality metrics inline
                    print(
                        f"[TRIAL {trial_num}] Step {step}/{steps} "
                        f"| acc={acc:.4f} | loss={loss:.4f} | ppl={ppl:.2f} "
                        f"| rouge1={update.get('rouge1',0):.4f} | bleu={update.get('bleu',0):.4f}"
                    )


            # Inspect Λ after modulation/training
            lambda_after = inspect_lambda(model_trial, label=f"Λ (trial {trial_num} after)")
            if lambda_before is not None and lambda_after is not None:
                delta = torch.linalg.norm(lambda_after - lambda_before).item()
                print(f"[ΔΛ] Change during trial {trial_num}: {delta:.6f}")

            last = metrics_stream[-1] if metrics_stream else {}
            score = last.get("accuracy", 0.0) - last.get("loss", 0.0)
            last_cfg, last_score = trial_cfg, score
            print(f"[RESULT] Trial {trial_num}/{budget} score={score:.3f}, metrics={last}")

            if best_score is None or score > best_score:
                best_score, best_metrics, best_variant = score, last, trial_cfg
                best_stream = list(metrics_stream)
                print(f"{GREEN}[BEST] Updated best score={best_score:.3f}, config={best_variant}{RESET}")

        if best_metrics:
            total_runtime = time.time() - start_time
            summary["runtime_secs"] = round(total_runtime, 2)

            print("[DIAGNOSTIC] Final Λ (best variant):")
            inspect_lambda(model if not args.train else model, label="Λ (final best)")

            #  Build a strict exp_config for provenance (no None fields)
            exp_cfg = {
                "variant": best_variant.get("variant"),
                "epsilon": float(best_variant.get("epsilon")),
                "lambda0": float(best_variant.get("lambda0")),
                "phi": best_variant.get("phi"),
                "ecm_init": best_variant.get("ecm_init"),
                "runtime": total_runtime,
                "maxSteps": steps,
            }
            print(f"[INFO] Final exp_config for report: {json.dumps(exp_cfg, indent=2)}")

            # Normalize naming for report compatibility
            # --- Guard against None trial_cfg on faster GCP runs ---
            if best_variant is not None:
                best_variant["maxSteps"] = best_variant.get("maxSteps", steps)
                best_variant.pop("timesteps", None)
            else:
                print("[DEBUG] best_variant is None at summary stage — skipping patch")


            # --- Preserve language-quality metrics from last stream (no extra eval) ---
            if best_stream and any(k in best_stream[-1] for k in ("rouge1", "rouge2", "rougeL", "bleu", "bertscore_f1")):
                best_metrics.update({
                    k: best_stream[-1][k]
                    for k in ("rouge1", "rouge2", "rougeL", "bleu", "bertscore_f1")
                    if k in best_stream[-1]
                })
                print("[EVALUATE] Added language-quality metrics to final best metrics:",
                      {k: best_metrics.get(k) for k in ("rouge1","rouge2","rougeL","bleu","bertscore_f1")})

            for k in ("rouge1", "rouge2", "rougeL", "bleu", "bertscore_f1"):
                if k in best_metrics:
                    mc._report_model_metrics(args.model_template_id, {k: best_metrics[k]}, step=steps)

            mc.finalize_and_certify(
                run_dir,
                model,            # keep main model artifact; Λ digests are uploaded separately
                tokenizer,
                best_metrics,
                exp_cfg["variant"],
                job_id,
                args.model_template_id,
                all_metrics=best_stream,
                baseline_metrics=baseline_metrics,
                exp_config=exp_cfg,
            )

            summary["best_variant"] = exp_cfg  # keep summary.json consistent
            summary["best_score"] = best_score
            summary["best_metrics"] = best_metrics
            summary["timesteps"] = steps
            print(f"[INFO] Reports saved under: {run_dir}")
            print(f"{GREEN}[DONE] Best trial finalized with score={best_score:.3f}, metrics={best_metrics}{RESET}")
        else:
            print("[WARN] No valid trials executed in auto mode.")

    # --- Always write summary.json ---
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Summary written to {summary_path}")


if __name__ == "__main__":
    main()
