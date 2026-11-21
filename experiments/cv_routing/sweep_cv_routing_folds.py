"""Stage 2 and the coverage sweep: GRAFT with the monitor flagging 1, 2 or 3 of an
environment's hack words. Same recipe as the pilot. Only the GRAFT arm is trained here:
the no-routing and baseline arms do not depend on the split and come from the pilot.

    CVR_ROUTED_SIZES=1+2+3 CVR_SHARD=0/3 python -u sweep.py --name cvr_folds_0 \
        --config sweeps/sweep_cv_routing_folds.py --no_wandb

Needs configs from `make_fold_configs.py --all-splits`.
  CVR_ENV           environment (default object_qa)
  CVR_ROUTED_SIZES  how many words the monitor flags, joined by + (default 2: the six 2-vs-2 splits)
  CVR_SKIP          config stems to leave out (default: the pilot's split)
  CVR_SHARD         i/n, take every n-th config starting at i, to spread a sweep over machines
"""
import os

from sweeps.respmon_repro_coh128_lam1_3seed import _new, _shared

CONFIG_DIR = os.environ.get("CVR_CONFIG_DIR", "configs/cv_routing")
# Sampler precision. Upstream default is bfloat16, which a T4 cannot run; set CVR_VLLM_DTYPE=float32 there.
VLLM_DTYPE = os.environ.get("CVR_VLLM_DTYPE", "bfloat16")
ENV = os.environ.get("CVR_ENV", "object_qa")
SIZES = {int(x) for x in os.environ.get("CVR_ROUTED_SIZES", "2").replace("+", ",").split(",")}
SKIP = set(os.environ.get("CVR_SKIP", "object_qa__r-actually-exactly-quite__h-indeed").split(","))
SHARD, NSHARD = (int(x) for x in os.environ.get("CVR_SHARD", "0/1").split("/"))
STEPS = {"object_qa": 1000, "persona_qa": 1000, "cities_qa": 2000, "addition_v2": 2000}


def n_routed(stem):
    return len(stem.split("__")[1].split("-")) - 1


def select(names, env=ENV, sizes=SIZES, skip=SKIP, shard=SHARD, nshard=NSHARD):
    stems = sorted(f[:-5] for f in names if f.startswith(env + "__") and f.endswith(".yaml"))
    stems = [s for s in stems if n_routed(s) in sizes and s not in skip]
    return [s for i, s in enumerate(stems) if i % nshard == shard]


STEMS = select(os.listdir(CONFIG_DIR)) if os.path.isdir(CONFIG_DIR) else []
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
