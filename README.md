# GRAFT-interp

Follow-up work on GRAFT ([Ward et al., 2026](https://arxiv.org/abs/2610.17062)), a gradient-routing method that removes reward hacking in RL by sending monitor-flagged updates to a deletable "forget" adapter. Two questions: what the forget adapter encodes, and whether GRAFT works as a test of misbehavior and on behaviors other than reward hacking.

## Status

No training runs yet. Results so far come from released adapter weights and evaluation files for Qwen3-8B on Countdown-Code (Samuel Knoche's reproduction; the GRAFT authors' weights are not public).

- The forget adapter's output weights are close to orthogonal to an activation-difference hack direction (3.0x a random direction, under 0.1% of energy, 4 seeds).
- Its input weights do respond to that direction (3.2x) and the retain adapter's do not (1.0x).
- With the forget adapter on, the model produces only the hack type the monitor flags: 0 unflagged hacks in 2,304 rollouts, against 73 with it off.
- After GRAFT, 3.7% of rollouts still hack and the monitor can see 36% of those (100% before routing).

## Layout

| Path | Contents |
|---|---|
| `docs/lit_review.md` | Literature review, assessment of ideas, proposed experiments |
| `docs/first_checks.md` | Results of the weight-level checks, with caveats |
| `docs/refs.bib` | Bibliography for both |
| `checks/` | Scripts for the checks. They read every file over HTTP into memory |
| `checks/results/` | Result files the write-ups cite |

## Reproducing the checks

Python 3 with numpy, about 3 GB of free memory, about 2.7 GB of network transfer, no GPU.

```bash
cd checks
python3 probe_headers.py results/base_header.json
python3 weights_check.py results results/base_header.json
python3 summarize_weights.py results/weights_check.json
python3 monitor_proxy.py results results/base_header.json
python3 token_readout.py results results/base_header.json
python3 eval_hack_types.py results
```
