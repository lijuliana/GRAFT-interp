"""Stage 2: the six 2-vs-2 splits of object_qa's hack words. NOT YET EXECUTED.

Same recipe as the pilot. The no-routing control does not depend on the split
(its monitor is never consulted), so stage 1's control runs are reused and only
the GRAFT arm is trained here.
    python -u sweep.py --name cvr_folds --config sweeps/sweep_cv_routing_folds.py --no_baseline --no_wandb
"""
import os

from sweeps.respmon_repro_coh128_lam1_3seed import _new, _shared

CONFIG_DIR = os.environ.get("CVR_CONFIG_DIR", "configs/cv_routing")
# Sampler precision. Upstream default is bfloat16, which a T4 cannot run; set CVR_VLLM_DTYPE=float32 there.
VLLM_DTYPE = os.environ.get("CVR_VLLM_DTYPE", "bfloat16")
ENV = os.environ.get("CVR_ENV", "object_qa")
STEPS = {"object_qa": 1000, "persona_qa": 1000, "cities_qa": 2000, "addition_v2": 2000}
STEMS = sorted(f[:-5] for f in os.listdir(CONFIG_DIR)
               if f.startswith(ENV + "__") and f.endswith(".yaml")
               and len(f.split("__")[1].split("-")) - 1 == len(f[:-5].split("__")[2].split("-")) - 1)
SEEDS = [1, 2, 3]

runs = [{
    **_shared, **_new,
    "config": f"{CONFIG_DIR}/{stem}.yaml",
    "max_steps": STEPS[ENV],
    "save_steps": STEPS[ENV],
    "unconditional_hackable": False,
    "hack_frac": 0.5,
    "vllm_dtype": VLLM_DTYPE,
    "rh_detector_recall": 1.0,
    "seed": seed,
    "run_name": f"cvr.{stem}.graft.s{seed}",
} for stem in STEMS for seed in SEEDS]

per_gpu = 2
no_baseline = True
