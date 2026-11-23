"""Filtering and reward-penalty baselines for the pilot's split, driven directly.

    python -u sweep.py --name cvr_pilot_bl --config sweeps/sweep_cv_routing_pilot_baselines.py --no_wandb

Upstream's sweep.py adds these two baselines to a GRAFT sweep automatically. At the pinned commit
(6da923d) its FILTER_BASELINE_STRIP leaves renormalization_mode='balanced' in place, the trainer
rejects that when routing is off, and the automatic copies stop at start-up. This file builds the
same runs: a pilot GRAFT run's parameters minus FILTER_BASELINE_STRIP, minus the two keys upstream
already strips from its no-intervention baseline. Run names match the automatic ones.
"""
from sweeps.sweep_cv_routing_pilot import runs as _graft

# sweep.py FILTER_BASELINE_STRIP at 6da923d
STRIP = {"routing_mode", "coherence", "coherence_every", "coherence_gen", "coherence_rh_mode", "coherence_rh_penalty",
         "coh_samples_per_rollout", "retain_mode", "retain_penalty", "run_name",
         "rh_detector_verifies_retain_samples", "rh_detector_retain_recall",
         "interlaced_coh_opt_batch_mode", "rollout_forget_scale_mode"}
# the omission: both are in upstream's ROUTING_ONLY_PARAMS
STRIP |= {"renormalization_mode", "split_moment"}

assert _graft and all(".graft.s" in r["run_name"] for r in _graft), "launch with CVR_ARMS unset or set to graft"


def _baseline(run, kind):
    stem = run["run_name"].split(".")[1]
    return {**{k: v for k, v in run.items() if k not in STRIP}, "routing_mode": "none", f"{kind}_baseline": True,
            "run_name": f"{kind}_cvr_{stem}_s{run['seed']}"}


runs = [_baseline(r, kind) for kind in ("filter", "reward_penalty") for r in _graft]
per_gpu = 2
no_baseline = True
