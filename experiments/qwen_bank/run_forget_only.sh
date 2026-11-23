#!/usr/bin/env bash
# Control for the transplant (experiment 18), added after its result. Run inside tmux on the GPU machine.
# The forget adapter of S9 and of S15 on the frozen base with the retain adapter switched off, same
# 32 questions and sampling as the transplant arms. If the hack rate matches the both-adapter rate, the
# forget adapter does not need a retain adapter, and swapping retain adapters could not have changed it.
set -uo pipefail
sudo shutdown -c 2>/dev/null; sudo shutdown -h +"${SAFETY_MIN:-120}"
cd "$HOME/GRAFT/rl"; source .venv/bin/activate
export GRAFT_COUNTDOWN_SFT_MODEL=$HOME/models/qwen3-8b-sft RH_REPO_PATH=$HOME/rl-rewardhacking
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Q=$HOME/GRAFT-interp/experiments/qwen_bank; A=$HOME/adapters; O=${OUT:-$HOME/qwen_out}; S=${SCALE:-1}
for name in s9 s15; do
  python "$Q/bank.py" --ckpt "$A/graft/cdhf100_gr_lccoh64_lr3_$name" --scales 0,1 --q_start 0 --n_q $((32/S)) --k 8 \
      --gen_batch 2 --out "$O/bank/tp_${name}_forget_only.json" >> "$O/bank.log" 2>&1
done
python "$Q/analyze_bank.py" "$O" > "$O/analysis.log" 2>&1
date -u +%s > "$O/forget_only.done"
sudo shutdown -c 2>/dev/null; sudo shutdown -h +"${IDLE_MIN:-30}"
