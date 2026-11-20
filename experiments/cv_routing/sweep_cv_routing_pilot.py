"""Stage 1 pilot. NOT YET EXECUTED.

Copy into the upstream checkout's rl/sweeps/ and launch from rl/:
    python -u sweep.py --name cvr_pilot --config sweeps/sweep_cv_routing_pilot.py --no_baseline --no_wandb

Hyperparameters are upstream's canonical toy recipe, imported from
sweeps/respmon_repro_coh128_lam1_3seed.py so they cannot drift from it.
Arms: GRAFT, and GRAFT without routing (rh_detector_recall 0.0), which upstream
documents as its no-routing baseline. Run names: cvr.<config stem>.<arm>.s<seed>.
"""
import os

from sweeps.respmon_repro_coh128_lam1_3seed import _new, _shared

CONFIG_DIR = os.environ.get("CVR_CONFIG_DIR", "configs/cv_routing")
STEMS = ["object_qa__r-actually-exactly-quite__h-indeed"]
STEPS = {"object_qa": 1000, "persona_qa": 1000, "cities_qa": 2000, "addition_v2": 2000}
ARMS = {"graft": 1.0, "noroute": 0.0}
SEEDS = [1, 2, 3]

runs = []
for stem in STEMS:
    env = stem.split("__")[0]
    for arm, recall in ARMS.items():
        for seed in SEEDS:
            runs.append({
                **_shared, **_new,
                "config": f"{CONFIG_DIR}/{stem}.yaml",
                "max_steps": STEPS[env],
                "save_steps": STEPS[env],
                "unconditional_hackable": False,
                "hack_frac": 0.5,
                "rh_detector_recall": recall,
                "seed": seed,
                "run_name": f"cvr.{stem}.{arm}.s{seed}",
            })

assert len(runs) == len(STEMS) * len(ARMS) * len(SEEDS)
per_gpu = 2
no_baseline = True
