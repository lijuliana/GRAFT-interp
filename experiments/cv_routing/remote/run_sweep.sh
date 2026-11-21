#!/usr/bin/env bash
# Train one or more sweeps, score every GRAFT / no-routing run with the shadow adapter, analyze,
# then schedule a shutdown so the machine never idles.
# Usage (inside tmux on the GPU machine):
#   bash run_sweep.sh <tag> <runs per GPU> <sweep spec> [<sweep spec> ...]
# A sweep spec is  name=<sweep name>,file=<file in rl/sweeps/>[,VAR=value ...]  (extra VAR=value are exported).
set -uo pipefail
TAG=$1; PER=$2; shift 2
G=$HOME/GRAFT-interp/experiments/cv_routing
cd "$HOME/GRAFT/rl"; source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH RH_REPO_PATH=$HOME/rl-rewardhacking WANDB_MODE=disabled
date -u +%s > "$HOME/$TAG.start"
NAMES=()
for spec in "$@"; do
  name=""; file=""; envs=()
  IFS=, read -ra parts <<< "$spec"
  for kv in "${parts[@]}"; do
    case "$kv" in name=*) name=${kv#name=};; file=*) file=${kv#file=};; *) envs+=("$kv");; esac
  done
  NAMES+=("$name")
  env "${envs[@]}" python -u sweep.py --name "$name" --config "sweeps/$file" --no_wandb --per_gpu "$PER" > "$HOME/$name.sweep.log" 2>&1
done
mkdir -p "$HOME/scores/$TAG"
for name in "${NAMES[@]}"; do
  for d in output/$name/cvr.*/; do
    [ -d "$d" ] || continue
    n=$(basename "$d"); stem=$(echo "$n" | cut -d. -f2); ck=$(ls -d "$d"checkpoint-* 2>/dev/null | sort -V | tail -1)
    [ -n "$ck" ] && python "$G/shadow_score.py" --checkpoint "$ck" --config "configs/cv_routing/$stem.yaml" \
        --out "$HOME/scores/$TAG/$n.npz" >> "$HOME/$TAG.score.log" 2>&1
  done
done
python "$G/analyze.py" "$HOME"/scores/$TAG/*.npz --out "$HOME/scores/$TAG/analysis.json" > "$HOME/$TAG.analysis.log" 2>&1
date -u +%s > "$HOME/$TAG.done"
sudo shutdown -c 2>/dev/null; sudo shutdown -h +"${IDLE_MIN:-45}"
