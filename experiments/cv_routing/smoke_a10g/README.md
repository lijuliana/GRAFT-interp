# Smoke test on an AWS A10G, 20 November 2026

Question: does upstream GRAFT's toy training run on an Ampere-class GPU with stock settings, how long is a step, and does this repo's scoring pipeline run on a real checkpoint.

Answer: yes to all three.

Setup: one `g5.2xlarge` (A10G, 24 GB), AWS deep-learning image (Ubuntu 22.04, driver 580, CUDA 12.8), `../remote/setup_vm.sh`, then `sweep_cv_routing_smoke.py` with upstream's defaults (sampler in bfloat16).

## Timing

- 50 training steps in 3 min 27 s: 4.15 s per step on average, 3.93 s at steady state. Whole launch including model load and step-0 evaluation: 5 min 12 s.
- A 1,000-step run projects to about 70 minutes when it has the GPU to itself. That is under the design's 2-hour limit.
- The 6-run pilot is about 7 GPU-hours one run at a time. Upstream ran several toy runs per GPU; whether two fit on a 24 GB A10G is untested.
- Setup from a fresh instance took about 3 minutes, because the image already has the driver and compiler.

## Pipeline check

`shadow_score.py` and `analyze.py` ran end to end on the 50-step checkpoint (`pipeline_check.json`). Scoring 2,048 rollouts took 19 seconds. The first attempt ran out of GPU memory; the script now gathers logits per token and scores in batches of 64.

The numbers in `pipeline_check.json` are not results. The checkpoint has 50 of 1,000 steps, there is one seed and no control run, and there are 63 held-out positives where the design requires 100.

## Hack-word rates during the 50 steps

From upstream's in-training evaluation, 256 prompts per mode (`routing_eval.jsonl`). Routed words are "actually", "exactly", "quite"; the held-out word is "indeed".

| Step | Retain only, routed | Retain only, held-out | Both adapters, routed | Both adapters, held-out |
|---|---|---|---|---|
| 0 | 0.012 | 0.105 | 0.016 | 0.109 |
| 10 | 0.023 | 0.086 | 0.039 | 0.047 |
| 20 | 0.020 | 0.074 | 0.004 | 0.062 |
| 30 | 0.020 | 0.059 | 0.008 | 0.062 |
| 40 | 0.020 | 0.059 | 0.016 | 0.047 |

Fifty steps is too short to read a trend from.
