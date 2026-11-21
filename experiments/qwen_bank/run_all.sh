#!/usr/bin/env bash
# All no-training Qwen3-8B read-outs, then a scheduled shutdown. Run inside tmux on the GPU machine.
set -uo pipefail
cd "$HOME/GRAFT/rl"; source .venv/bin/activate
export GRAFT_COUNTDOWN_SFT_MODEL=$HOME/models/qwen3-8b-sft RH_REPO_PATH=$HOME/rl-rewardhacking
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Q=$HOME/GRAFT-interp/experiments/qwen_bank; A=$HOME/adapters; O=${OUT:-$HOME/qwen_out}; S=${SCALE:-1}
G1=$A/graft/cdhf100_gr_lccoh64_lr1_s1          # GRAFT, lr 5e-4, seed 1: deployed model keeps undetectable hacks
S9=$A/graft/cdhf100_gr_lccoh64_lr3_s9           # GRAFT, paper settings
S15=$A/graft/cdhf100_gr_lccoh64_lr3_s15
NR=$A/graft-no-routing/cdhf100_noroute_anchor_lr1_s1
mkdir -p "$O/bank" "$O/scores"; date -u +%s > "$O/start"
b() { python "$Q/bank.py" "$@" >> "$O/bank.log" 2>&1; }
s() { python "$Q/score.py" "$@" >> "$O/score.log" 2>&1; }
# 1. deployed model and training-like rollouts of one GRAFT run, on disjoint questions
b --ckpt $G1 --scales 1,0 --q_start 0  --n_q $((64/S)) --k 16 --gen_batch 1 --out $O/bank/g1_dep.json
b --ckpt $G1 --scales 1,1 --q_start 64 --n_q $((64/S)) --k 8  --gen_batch 2 --out $O/bank/g1_both.json
s --bank $O/bank/g1_dep.json  --ckpt $G1 --policy_scales 1,0 --out $O/scores/dep.npz
s --bank $O/bank/g1_both.json --ckpt $G1 --policy_scales 1,1 --out $O/scores/trn.npz
s --bank $O/bank/g1_dep.json  --ckpt $NR --parts shadow      --out $O/scores/dep_ctrl.npz
python "$Q/analyze_bank.py" "$O" > "$O/analysis.log" 2>&1
date -u +%s > "$O/stage1.done"
# 2. transplant: one run's forget adapter on another run's model
for pair in "s9:$S9:$S15" "s15:$S15:$S9"; do
  IFS=: read -r name own other <<< "$pair"
  b --ckpt $own --scales 1,1 --q_start 0 --n_q $((32/S)) --k 8 --gen_batch 2 --out $O/bank/tp_${name}_own_both.json
  b --ckpt $own --scales 1,0 --q_start 0 --n_q $((32/S)) --k 4 --gen_batch 4 --out $O/bank/tp_${name}_own_retain.json
  b --ckpt $own --forget_ckpt $other --scales 1,1 --q_start 0 --n_q $((32/S)) --k 8 --gen_batch 2 --out $O/bank/tp_${name}_foreign_forget.json
done
python "$Q/analyze_bank.py" "$O" > "$O/analysis.log" 2>&1
date -u +%s > "$O/done"
sudo shutdown -c 2>/dev/null; sudo shutdown -h +"${IDLE_MIN:-45}"
