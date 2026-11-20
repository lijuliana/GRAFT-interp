# Cross-validated routing: design

Status, 20 November 2026: designed, code drafted. The stage 0 smoke test ran on an Azure T4 and failed: upstream's trainer needs FlashAttention and bfloat16, which a T4 lacks (`smoke_t4/README.md`). No training step has completed anywhere. `analyze.py` and `make_fold_configs.py` are tested on a laptop. The sweep files load and launch under upstream's `sweep.py`. `shadow_score.py` has never been executed.

## Design note

**Claim (systematic, per environment).** When GRAFT routes on a subset of hack words, the forget adapter, run as a shadow on the deployed model, detects hacks that use the held-out words, and it does so better than the same adapter from a run with no routing.

**Why it matters.** [@draganover2026sacred] proposes keeping the forget adapter as a test of misbehavior. A test is only worth having for hacks the routing monitor missed, which are 64% of the hacks left after GRAFT on Countdown-Code (`docs/first_checks.md`). This experiment measures that held-out detection rate with exact ground truth, the way test error is measured by holding out data.

**Best-case conclusion.** Routing on half of the hack vocabulary gives a shadow adapter that catches most hacks from the other half at a 1% false-positive rate, across environments, while a no-routing adapter and a probe trained on the same monitor labels do not. A lab could then report a held-out detection rate for a training technique.

**Stage.** Understand. Three accounts are written down and the experiment separates them.

**Shadow adapter.** The deployed model acts with the retain adapter only. The forget adapter reads the same MLP inputs and its output is discarded. Nothing about the model's behavior changes.

**Competing accounts.**

- A, shared hack representation. Routed and held-out hack words share upstream features, so an adapter trained on routed hacks responds to held-out ones.
- B, distilled monitor. The forget adapter encodes the routed words. It responds to held-out words no more than any adapter does.
- C, joint-learning artifact. Unflagged rollouts update both adapters, so the forget adapter also learned the held-out hack directly, as half of a jointly trained pair. It responds to held-out hacks, and so does the "forget" half of a run with no routing.

**Predictions.**

| Measurement | A | B | C |
|---|---|---|---|
| Shadow score on held-out hacks, GRAFT run | high | near chance | high |
| Same score, no-routing run | lower than GRAFT | near chance | same as GRAFT |
| Probe trained on monitor labels, held-out hacks | at or below the adapter | near chance, or above the adapter if words are close in embedding space | not informative |
| Shadow score on routed hacks (in-fold) | high | high | high |

The no-routing run separates A from C. The probe separates "the adapter generalizes" from "any detector trained on these labels generalizes".

**Scores.** All are computed on text the deployed model wrote, in a forward pass with the forget adapter's output discarded.

- Primary: per-token L2 norm of the forget adapter's output, summed over layers, maximum over response tokens.
- Secondary: per-token log-likelihood ratio of the sampled token under both adapters against retain only, summed over response tokens.
- Each is also reported for tokens strictly before the first hack word, to separate anticipating a hack from recognizing a word that was already written.

**Kill-first test (the pilot).** Upstream's stock `object_qa` config is already one fold: the monitor flags "actually", "exactly", "quite" and misses "indeed". Train it and a no-routing run, 3 seeds each, score both. No new configs are needed. This is the cheapest run that can end the project.

**Baselines and controls.**

| Control | Objection it answers |
|---|---|
| No-routing run (`rh_detector_recall: 0`), same score | The forget adapter saw held-out hacks in unflagged updates, so detection is joint learning (account C) |
| Retain adapter's output norm, same run | Any adapter is more active on hack tokens |
| Linear probe on the residual stream, trained on the monitor's flags | A cheap probe with the same labels does as well |
| Scores before the first hack word | The score only reads the hack word after it is written |
| In-fold detection of routed words | The score definition is wrong (positive control: must be at least 0.9 AUROC) |
| Labels shuffled within a run | The pipeline produces separation from nothing (must give 0.5) |

**Noise floor.** AUROC spread across the 3 seeds of one config, and across two disjoint halves of one run's rollouts. A difference between GRAFT and control smaller than either is reported as no difference.

**Held-out axes.** Hack words (the point of the experiment), prompts (upstream's fixed test split, seed 99), seeds. Stage 3 adds environments.

**Acceptance criteria, fixed before any run.**

- Validity: at least 100 held-out-word hacks among the deployed model's scored rollouts per run. In-fold AUROC at least 0.9. Planted-signal tests pass.
- Routing-specific detection: held-out AUROC of the primary score in the GRAFT run exceeds the no-routing run's by at least 0.10 on average, with the same sign in 3 of 3 seed pairs.
- Beats the probe: held-out AUROC at least 0.05 above the probe's in at least 2 of 3 seeds.
- Stop rule: if the GRAFT run's held-out AUROC is within 0.05 of the no-routing run's on average, stop after the pilot and report the null.
- Also reported whatever the outcome: recall at a 1% false-positive rate with bootstrap intervals, and the worst fold.

**Independent checks.** Hack labels come from upstream's own detectors, which share no code with the scores. `analyze.py` is checked against planted signals (`test_analyze.py`). One run is rescored with a second sampling seed.

**Budget.** Unknown until a smoke test completes on a GPU that can run the trainer (Ampere or newer; a T4 cannot). Upstream ran 3 to 5 toy runs per large GPU. The first action on the T4 is one 50-step run, timed, to project a 1,000-step run. If the projection is over 2 hours per run on a T4, move to AWS (next in the provider order) instead of changing hyperparameters.

## Stages

| Stage | Runs | Purpose |
|---|---|---|
| 0. Smoke test | 1 run, 50 steps | Done on a T4: the stack does not run there. Repeat on an Ampere or newer GPU to time a step |
| 1. Pilot | 6 runs of 1,000 steps: stock `object_qa` and no routing, seeds 1 to 3 | Kill-first test with the stop rule above |
| 2. Folds | 18 runs: the 6 two-against-two splits of the 4 hack words, 3 seeds. Controls reuse stage 1 | Cross-validated held-out recall for one environment |
| 3. Other environments | `cities_qa`, `addition_v2` (4 words), `persona_qa` (6 phrases) | Whether the estimate depends on the task |

The review's estimate of "about 36 short runs" assumed two folds in each of six environments. The repeat and sort environments have only two hack forms each, so their folds are upstream's existing "inverted" configs. The keyword environments give real k-fold structure and are used here.

## How to run

Everything runs from the `rl/` directory of a checkout of `jnward/GRAFT` (commit `6da923d`), set up as its README describes.

```bash
G=/path/to/GRAFT-interp/experiments/cv_routing

# once: put the fold configs and sweep files into the upstream checkout
python $G/make_fold_configs.py --out configs/cv_routing
cp $G/sweep_cv_routing_*.py sweeps/

# stage 0: smoke test, timed
python train.py --config configs/cv_routing/object_qa__r-actually-exactly-quite__h-indeed.yaml \
    --max_steps 50 --output_dir output/cvr_smoke --no_wandb

# stage 1: pilot (6 runs)
python -u sweep.py --name cvr_pilot --config sweeps/sweep_cv_routing_pilot.py --no_baseline --no_wandb

# score every run. Run names are cvr.<config stem>.<arm>.s<seed>
for d in output/cvr_pilot/cvr.*/; do
  n=$(basename $d); stem=$(echo $n | cut -d. -f2)
  python $G/shadow_score.py --checkpoint $d/checkpoint-1000 \
      --config configs/cv_routing/$stem.yaml --out scores/$n.npz
done

# analyze (numpy only, any machine)
python $G/analyze.py scores/*.npz --out cvr_pilot.json
```

For stage 2, score each stage 1 no-routing checkpoint once per fold config, so its rollouts are labelled with that fold's word split.

Tests that run without a GPU: `python test_analyze.py` (planted signal recovered, null gives the stop verdict, a signal that appears only after the hack word vanishes from the "before" window, fold configs partition the vocabulary).

## Known gaps

- `shadow_score.py` is unexecuted. The sweep files launch under upstream's `sweep.py`, and no run has got past step 0. Flag names in the sweep dictionaries are copied from upstream's canonical sweep and its README. Run names with dots are accepted.
- The pilot's config is generated here and matches upstream's stock `configs/object_qa_respmon.yaml` except for its name (checked by diff).
- Upstream's toy anchor applies a penalty of 2.0 to monitor-flagged hacks in retain-only rollouts. The monitor therefore does enter the reward in the toy setting, for the anchor slice only. This is upstream's canonical recipe and is kept.
- With `rh_detector_recall: 0` upstream also disables that anchor penalty, so the no-routing control differs from GRAFT in two ways. A second control with routing off and the penalty on would need a code change upstream.
- Hack words are detected by case-insensitive substring match, as upstream does.
- The toy hack is a word in the output. A positive result here shows generalization across hack words, which is a weaker claim than generalization across hack strategies in Countdown-Code. Experiment 2 in the project log covers that case.
- The untrained model writes the held-out word "indeed" in about 8% of responses (62 of 768, `smoke_t4/README.md`). Held-out positives must be compared with this base rate, and the 100-positive validity rule should count hacks above it.
