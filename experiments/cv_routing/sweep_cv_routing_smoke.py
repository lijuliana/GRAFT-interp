"""Stage 0 smoke test: one GRAFT run of 50 steps with the canonical toy recipe.
    python -u sweep.py --name cvr_smoke --config sweeps/sweep_cv_routing_smoke.py --no_baseline --no_wandb --no_mps
"""
import os

from sweeps.respmon_repro_coh128_lam1_3seed import _new, _shared

CONFIG_DIR = os.environ.get("CVR_CONFIG_DIR", "configs/cv_routing")
STEM = "object_qa__r-actually-exactly-quite__h-indeed"
STEPS = int(os.environ.get("CVR_SMOKE_STEPS", "50"))

runs = [{
    **_shared, **_new,
    "config": f"{CONFIG_DIR}/{STEM}.yaml",
    "max_steps": STEPS,
    "save_steps": STEPS,
    "unconditional_hackable": False,
    "hack_frac": 0.5,
    "rh_detector_recall": 1.0,
    "seed": 1,
    "run_name": f"cvr.{STEM}.graft.s1",
}]
per_gpu = 1
no_baseline = True
