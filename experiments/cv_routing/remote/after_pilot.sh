#!/usr/bin/env bash
# Follow-up on the pilot machine. Start it in a second tmux window; it waits for run_sweep.sh to finish.
#   1. Score the no-routing checkpoints under every other object_qa word split. A no-routing run never
#      uses the split, so one set of three seeds is the control for every split's GRAFT runs.
#   2. Train the filtering and reward-penalty baselines (see sweep_cv_routing_pilot_baselines.py).
# Then schedule a shutdown so the machine never idles.
set -uo pipefail
TAG=cvr_pilot; G=$HOME/GRAFT-interp/experiments/cv_routing
PILOT_STEM=object_qa__r-actually-exactly-quite__h-indeed
while [ ! -f "$HOME/$TAG.done" ]; do sleep 60; done
sleep 30; sudo shutdown -c 2>/dev/null; sudo shutdown -h +"${SAFETY_MIN:-720}"
cd "$HOME/GRAFT/rl"; source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH RH_REPO_PATH=$HOME/rl-rewardhacking WANDB_MODE=disabled
python "$G/make_fold_configs.py" --out configs/cv_routing --all-splits
mkdir -p "$HOME/scores/noroute_folds"
for d in output/cvr_pilot_noroute/cvr.*/; do
  [ -d "$d" ] || continue
  seed=$(basename "$d"); seed=${seed##*.}; ck=$(ls -d "$d"checkpoint-* 2>/dev/null | sort -V | tail -1)
  [ -n "$ck" ] || continue
  for cfg in configs/cv_routing/object_qa__*.yaml; do
    stem=$(basename "$cfg" .yaml); [ "$stem" = "$PILOT_STEM" ] && continue
    python "$G/shadow_score.py" --checkpoint "$ck" --config "$cfg" \
        --out "$HOME/scores/noroute_folds/cvr.$stem.noroute.$seed.npz" >> "$HOME/noroute_folds.score.log" 2>&1
  done
done
date -u +%s > "$HOME/noroute_folds.done"
python -u sweep.py --name cvr_pilot_bl --config sweeps/sweep_cv_routing_pilot_baselines.py --no_wandb --per_gpu 2 \
    > "$HOME/cvr_pilot_bl.sweep.log" 2>&1
date -u +%s > "$HOME/cvr_pilot_bl.done"
sudo shutdown -c 2>/dev/null; sudo shutdown -h +"${IDLE_MIN:-45}"
