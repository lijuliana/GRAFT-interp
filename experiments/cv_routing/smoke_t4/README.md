# Smoke test on an Azure T4, 20 November 2026

Question: does upstream GRAFT's toy training run on a 16 GB T4, and how long is a step.

Answer: no. The sampler can be made to start, and the trainer cannot run. No training step completed, so there is no timing.

Setup: one `Standard_NC8as_T4_v3` machine, Ubuntu 22.04, NVIDIA driver 580, CUDA toolkit 12.8, upstream `jnward/GRAFT` at commit `6da923d` installed with `uv sync` and its vLLM patches (`../remote/setup_vm.sh`). One run of `sweep_cv_routing_smoke.py` per attempt.

| Attempt | Change | Outcome |
|---|---|---|
| 1 | Stock settings | vLLM refuses to start: bfloat16 needs GPU compute capability 8.0, a T4 has 7.5 |
| 2 | `--vllm_dtype float32` on the command line | Same error. Upstream reads the sampler precision from each run's settings, and the flag is ignored |
| 3 | `vllm_dtype: float32` in the run settings | vLLM's adapter kernel asserts the input is float16 or bfloat16 |
| 4 | `vllm_dtype: float16` | vLLM's attention backend compiles a kernel at start-up and needs `nvcc`, which the driver package lacks |
| 5 | float16, CUDA toolkit installed | Sampler starts and the step-0 evaluation runs. The first training step fails: "FlashAttention only supports Ampere GPUs or newer" |

Upstream's `train.py` loads the model with `attn_implementation="flash_attention_2"` and its routed loss asserts the packed FlashAttention path, with no switch for either. Running on a T4 would mean editing upstream's trainer and sampling in float16, which are two departures from the published recipe. The design's rule for this case is to move to the next provider. The trainer needs an Ampere or newer GPU (A10G, L4, L40S, A100, H100).

Two things that carry over to any machine:

- The CUDA toolkit is a required install, and the setup script now includes it.
- The sampler precision is set per run. The sweep files read it from `CVR_VLLM_DTYPE` and default to upstream's bfloat16.

## Step-0 evaluation

The one evaluation that ran is the untrained model (adapters start at zero, so the three adapter modes are three samples of the same policy). Files: `step0_routing_eval.jsonl`, `step0_eval_samples.jsonl` (256 prompts per mode, sampler in float16; upstream stores scores for every row and the response text for 24 of them).

| Mode | Any hack word | Routed words (actually, exactly, quite) | Held-out word only (indeed) |
|---|---|---|---|
| both | 0.105 | 0.020 | 0.086 |
| retain_only | 0.117 | 0.047 | 0.070 |
| forget_only | 0.102 | 0.016 | 0.086 |

Pooled over the 768 samples, 83 contain a hack word, 21 a routed word and 62 only the held-out word. The base model already writes "indeed" in about 8% of responses before any training. The pilot's positives therefore include ordinary uses of the word, and the analysis should report the trained model's rate against this base rate.
