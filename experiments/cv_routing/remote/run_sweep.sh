#!/usr/bin/env bash
# Train a sweep, score every GRAFT/no-routing run with the shadow adapter, analyze, then
# schedule a shutdown so the machine never idles.
# Usage (inside tmux, on the GPU machine): bash run_sweep.sh <name> <sweep file in rl/sweeps/> [runs per GPU]
set -uo pipefail
NAME=$1; SWEEP=$2; PER=${3:-2}
G=$HOME/GRAFT-interp/experiments/cv_routing
cd "$HOME/GRAFT/rl"; source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH RH_REPO_PATH=$HOME/rl-rewardhacking WANDB_MODE=disabled
date -u +%s > "$HOME/$NAME.start"
python -u sweep.py --name "$NAME" --config "sweeps/$SWEEP" --no_wandb --per_gpu "$PER" > "$HOME/$NAME.sweep.log" 2>&1
mkdir -p "$HOME/scores/$NAME"
for d in output/$NAME/cvr.*/; do
  n=$(basename "$d"); stem=$(echo "$n" | cut -d. -f2); ck=$(ls -d "$d"checkpoint-* 2>/dev/null | sort -V | tail -1)
  [ -n "$ck" ] && python "$G/shadow_score.py" --checkpoint "$ck" --config "configs/cv_routing/$stem.yaml" \
      --out "$HOME/scores/$NAME/$n.npz" >> "$HOME/$NAME.score.log" 2>&1
done
python "$G/analyze.py" "$HOME"/scores/$NAME/*.npz --out "$HOME/scores/$NAME/analysis.json" > "$HOME/$NAME.analysis.log" 2>&1
date -u +%s > "$HOME/$NAME.done"
sudo shutdown -c 2>/dev/null; sudo shutdown -h +"${IDLE_MIN:-45}"
