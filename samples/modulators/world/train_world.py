#!/usr/bin/env python3
"""
Experimental trainer scaffold for world models (e.g. V-JEPA 2).

This sample exists so the new `world` model kind has a concrete starter path in
the SDK repo. It intentionally stops short of claiming a full evaluation loop,
because V-JEPA-style temporal world models require a dedicated adapter around:

- frame-window sampling
- temporal batching
- embedding/state decoding
- world-model-specific KPIs

Current behavior:
- reads modulation config from AOC
- optionally auto-starts the modulation job
- downloads the registered model snapshot into the run directory
- writes a summary artifact explaining that this sample is a scaffold

This keeps the folder structure and DX consistent while the actual world-model
runtime/evaluation adapter is implemented in the SDK and robot sample.
"""

import argparse
import datetime
import json
import os
from typing import Any, Dict

from ephapsys.modulation import ModulatorClient


def _truthy(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", type=str, default=os.getenv("AOC_BASE_URL", os.getenv("BASE_URL", "http://localhost:7001")))
    parser.add_argument("--api_key", type=str, default=os.getenv("AOC_MODULATION_TOKEN", ""))
    parser.add_argument("--model_template_id", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="./artifacts_world")
    parser.add_argument("--auto_start", type=int, default=int(os.getenv("AUTO_START", "1")))
    args = parser.parse_args()

    if not args.api_key:
        raise RuntimeError("API token missing. Provide --api_key or set AOC_MODULATION_TOKEN in the environment")

    os.makedirs(args.outdir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.outdir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"[INFO] Run directory created: {run_dir}")

    mc = ModulatorClient(args.base_url, args.api_key)

    if args.auto_start:
        print("[INFO] Auto-starting experimental world-model modulation job...")
        dataset = {
            "kind": "repo",
            "source": "external",
            "name": "kinetics700",
            "config": "default",
            "split": "train[:256]",
        }
        kpi = {
            "targets": [
                {"name": "accuracy", "direction": "max", "weight": 1},
                {"name": "loss", "direction": "min", "weight": 1},
            ],
            "maxSteps": 16,
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
        print("[INFO] AUTO_START=0 -> waiting for an already-started world-model job")

    tpl, job_id = mc.wait_for_job_id(args.model_template_id)
    recipe: Dict[str, Any] = tpl.get("DesiredModulation") or {}
    local_model_dir = mc.download_and_extract_model(args.model_template_id, run_dir)

    source_repo = recipe.get("SourceRepo") or tpl.get("source_repo") or "facebook/vjepa2-vitl-fpc64-256"
    summary = {
        "job_id": job_id,
        "model_template_id": args.model_template_id,
        "model_kind": "world",
        "source_repo": source_repo,
        "local_model_dir": local_model_dir,
        "status": "scaffold_only",
        "note": (
            "World-model evaluation is not wired into the generic modulator yet. "
            "Use this scaffold as the registration/download/bootstrap path for V-JEPA "
            "until the dedicated temporal evaluation adapter lands."
        ),
        "recipe": recipe,
    }

    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    readme_path = os.path.join(run_dir, "README.txt")
    with open(readme_path, "w") as f:
        f.write(
            "Experimental world-model scaffold\n"
            "================================\n\n"
            f"Model template: {args.model_template_id}\n"
            f"Source repo: {source_repo}\n"
            f"Local snapshot: {local_model_dir}\n\n"
            "No world-model KPI loop ran in this sample. The next step is a dedicated\n"
            "temporal adapter for frame-window preparation and V-JEPA-specific metrics.\n"
        )

    print("[INFO] Downloaded world-model snapshot and wrote scaffold summary artifacts.")
    print(f"[INFO] Summary written to {summary_path}")


if __name__ == "__main__":
    main()
