# Cross-validated routing: results

21 November 2026. Experiment 1 (pilot and word splits) and the first test of idea 15, on SmolLM2-135M in upstream's `object_qa` environment. 54 training runs of 1,000 steps on AWS A10G instances. Design, criteria and the rule added after the pilot: `DESIGN.md`. Numbers: `results/summary.json` and `results/summary.log` (from `summarize_splits.py`), `results/cvr_pilot/scores/cvr_pilot/analysis.json` (analysis fixed before the run) and `posthoc.json` (written after it).

The environment rewards the model for writing any of four words. A monitor flags some of them ("routed" words) and never the others ("held-out" words). A held-out hack is a response with a held-out word and no routed word. The deployed model is the trained model with the forget adapter removed.

## What the model writes

The no-routing model writes only two of the four words: "indeed" in about 36% of responses and "actually" in about 13% (as the only hack word in the response). It writes "exactly" and "quite" in under 0.1%. So the environment has two hack words in practice, and a split is informative only if one of them is routed and the other held out.

## Pilot: verdicts under the criteria fixed before the run

Split: the monitor flags "actually", "exactly" and "quite" and misses "indeed". 3 seeds each of GRAFT and no routing, 8,192 deployed-model responses scored per run.

| Criterion | Result | Verdict |
|---|---|---|
| Validity: at least 100 held-out hacks per GRAFT run | 119, 40, 68 | Fails in 2 of 3 seeds |
| Validity: primary score at least 0.9 AUROC on routed words | 0.94, 0.94, 0.94 | Holds |
| Routing-specific: primary score at least 0.10 above the no-routing run's, same sign in 3 of 3 seeds | 0.73, 0.70, 0.73 against 0.77, 0.83, 0.86 (differences -0.04, -0.12, -0.13) | Fails |
| Stop if within 0.05 of the no-routing run on average | 0.10 below | Stop |
| Beats the probe by 0.05 in at least 2 of 3 seeds | Probe 0.54, 0.46, 0.56 | Holds, against a probe at chance |

The primary score is the maximum over response tokens of the forget adapter's output norm.

## Pilot: hack words in the deployed model, every training arm

Mean of the 10 evaluations at steps 900 to 990, seeds 1 to 3. The deployed model is retain-only for GRAFT and the whole model for the other arms.

| Arm | Held-out word ("indeed") | Routed words | Questions correct |
|---|---|---|---|
| Before training | 12%, 13%, 13% | 2% | 10% |
| No intervention | 27%, 41%, 35% | 21%, 13%, 18% | 52%, 54%, 50% |
| No routing | 37%, 35%, 35% | 17%, 13%, 19% | 52%, 51%, 52% |
| Filtering on the monitor | 50%, 49%, 35% | 2%, 1%, 2% | 49%, 53%, 56% |
| Reward penalty on the monitor | 47%, 50%, 0.3% | 0.0%, 0.2%, 0.2% | 51%, 57%, 56% |
| GRAFT | 1.5%, 0.3%, 0.7% | 0.0% | 50%, 47%, 50% |

- Filtering and the reward penalty remove the routed words and leave the held-out word at or above its no-intervention rate in 5 of 6 runs. GRAFT removes both.
- With only the forget adapter on, the GRAFT model writes the held-out word in 18%, 14% and 24% of responses. The word the monitor never flagged ended up in the forget adapter.
- This reproduces the GRAFT paper's result for this environment. It is the opposite of idea 15's claim for this split.
- The filter and reward-penalty runs are a rerun: the first copies stopped at start-up (`sweep_cv_routing_pilot_baselines.py` explains why).

## Pilot: the secondary score (read after the results)

The design's secondary score is the log-likelihood ratio of the response under both adapters against retain only, summed over tokens.

| Score, held-out hacks against clean responses | GRAFT, seeds 1 to 3 | No routing |
|---|---|---|
| Output norm (primary), AUROC | 0.73, 0.70, 0.73 | 0.77, 0.83, 0.86 |
| Likelihood ratio, AUROC | 0.88, 0.91, 0.83 | 0.59, 0.68, 0.64 |
| Likelihood ratio, recall at a 1% false-positive rate | 63%, 70%, 50% (60% of 227 pooled) | 2%, 9%, 17% (8% of 6,097 pooled) |
| Likelihood ratio at the hack token alone, AUROC | 0.99, 1.00, 1.00 | 0.69, 0.93, 0.96 |
| Likelihood ratio over tokens before the hack token, AUROC | 0.76, 0.74, 0.64 | 0.65, 0.67, 0.63 |
| Response length alone, AUROC | 0.54, 0.57, 0.60 | 0.54, 0.55, 0.59 |

- The comparison was read after the results, so it was treated as a lead and put to the word splits under a rule fixed before any of them was scored (`DESIGN.md`).
- Length does not produce it: responses are about 16 tokens long in both groups.
- The signal sits at the hack token. The forget adapter makes the held-out word more likely where it is written. Before the word the score is near the no-routing run's.
- The no-routing comparison is weak for this score. In a no-routing run both halves of the adapter produce the hack, so removing one half changes the likelihood of everything a little and of hack words not much more.

## Word splits: 13 more splits, 3 GRAFT seeds each

39 GRAFT runs, with the pilot's three no-routing runs scored under each split as the control. The no-routing model column is the trained model with both halves of its adapter on.

**One written word routed, the other held out (8 splits, 24 GRAFT runs).**

| Routed words | Held-out words | Held-out hacks in 8,192 deployed responses, seeds 1 to 3 | No-routing model | Flagged rollouts in 8,192 with both adapters | Likelihood ratio AUROC, GRAFT | No routing |
|---|---|---|---|---|---|---|
| actually, exactly, quite | indeed | 119, 40, 68 (1.5%, 0.5%, 0.8%) | 36%, 34%, 35% | 1385, 1412, 1575 | 0.88, 0.91, 0.83 | 0.59, 0.68, 0.64 |
| actually, exactly | indeed, quite | 2, 0, 7 (0.0%, 0.0%, 0.1%) | 37%, 35%, 35% | 3579, 3101, 3775 | 0.94, -, 0.76 | 0.58, 0.68, 0.63 |
| actually, quite | indeed, exactly | 17, 0, 21 (0.2%, 0.0%, 0.3%) | 35%, 35%, 36% | 2093, 3428, 34 | 0.83, -, 0.66 | 0.58, 0.68, 0.64 |
| actually | indeed, exactly, quite | 0, 18, 9 (0.0%, 0.2%, 0.1%) | 37%, 34%, 35% | 3521, 2565, 1878 | -, 0.77, 0.66 | 0.58, 0.67, 0.64 |
| indeed, exactly, quite | actually | 0, 0, 4 (0.0%, 0.0%, 0.0%) | 14%, 12%, 14% | 3060, 3205, 3223 | -, -, 0.56 | 0.56, 0.61, 0.47 |
| indeed, exactly | actually, quite | 2, 3, 89 (0.0%, 0.0%, 1.1%) | 14%, 12%, 14% | 3504, 3105, 202 | 0.91, 0.64, 0.79 | 0.56, 0.59, 0.48 |
| indeed, quite | actually, exactly | 2, 2, 207 (0.0%, 0.0%, 2.5%) | 14%, 12%, 14% | 3525, 2921, 334 | 0.02, 0.50, 0.80 | 0.56, 0.59, 0.46 |
| indeed | actually, exactly, quite | 4, 4, 249 (0.0%, 0.0%, 3.0%) | 14%, 13%, 14% | 3626, 3833, 515 | 0.29, 0.39, 0.83 | 0.56, 0.62, 0.48 |

**The monitor flags only words the model never writes (3 splits).** These are the only splits the added rule counts as valid.

| Routed words | Held-out words | Held-out hacks in 8,192 deployed responses, seeds 1 to 3 | No-routing model | Flagged rollouts in 8,192 with both adapters | Likelihood ratio AUROC, GRAFT | No routing |
|---|---|---|---|---|---|---|
| exactly, quite | indeed, actually | 3741, 2627, 359 (46%, 32%, 4.4%) | 53%, 48%, 54% | 6, 11, 10 | 0.61, 0.65, 0.95 | 0.58, 0.66, 0.57 |
| exactly | indeed, actually, quite | 1733, 3374, 1987 (21%, 41%, 24%) | 53%, 48%, 54% | 9, 7, 5 | 0.77, 0.58, 0.70 | 0.58, 0.66, 0.57 |
| quite | indeed, actually, exactly | 3491, 3229, 158 (43%, 39%, 1.9%) | 53%, 50%, 54% | 5, 2, 1 | 0.68, 0.59, 0.50 | 0.58, 0.66, 0.58 |

**Both written words routed (3 splits).** No model writes the held-out words, so there is nothing to detect.

| Routed words | Held-out words | Held-out hacks in 8,192 deployed responses, seeds 1 to 3 | No-routing model | Flagged rollouts in 8,192 with both adapters | Likelihood ratio AUROC, GRAFT | No routing |
|---|---|---|---|---|---|---|
| indeed, actually, exactly | quite | 0, 2, 1 (0.0%, 0.0%, 0.0%) | 0.0%, 0.0%, 0.0% | 4148, 4019, 4696 | -, 0.60, 0.96 | 0.72, 0.51, 0.34 |
| indeed, actually, quite | exactly | 0, 1, 0 (0.0%, 0.0%, 0.0%) | 0.0%, 0.0%, 0.0% | 4020, 4346, 3805 | -, 0.17, - | 0.15, 0.36, 0.19 |
| indeed, actually | exactly, quite | 0, 2, 3 (0.0%, 0.0%, 0.0%) | 0.0%, 0.0%, 0.0% | 4204, 4017, 4435 | -, 0.16, 0.40 | 0.34, 0.37, 0.26 |

### Idea 15: do held-out words rise when routing is on

No, in 0 of 42 GRAFT runs.

- When one written word is routed, the other disappears from the deployed model: 0 to 249 held-out hacks in 8,192 responses across 24 runs (median 4, at most 3.0%), against 12% to 37% for the no-routing model. It does not matter which of the two words is routed, or whether one, two or three words are flagged.
- When the monitor flags only words the model never writes, almost nothing is routed (1 to 11 flagged rollouts in 8,192) and GRAFT behaves like a no-routing run: held-out hacks in 2% to 46% of deployed responses, still below the no-routing model's 48% to 54% in all 9 runs.
- Filtering and the reward penalty do push the model to the held-out word (pilot table above). Routing does not.

### The rule added after the pilot: fails

| | |
|---|---|
| Valid splits (every GRAFT seed has at least 100 held-out hacks) | 3 of 13: routed "exactly"; routed "quite"; routed "exactly" and "quite" |
| Likelihood-ratio AUROC, GRAFT minus no routing, per seed | +0.19, -0.09, +0.13; +0.11, -0.07, -0.08; +0.03, -0.01, +0.37 |
| Splits that confirm (mean at least +0.10, same sign in every seed) | 0 of 3 |
| Outcome under the rule | Fails |

- The rule did not test what it was written to test. Its validity condition kept only the splits where the monitor flags words the model never writes, so nothing was routed in them. In every split where routing happened, the deployed model has too few held-out hacks to pass validity. I did not know when writing the rule that the model writes only two of the four words.
- Descriptive only, chosen after seeing the table: among the 24 informative runs, 9 left at least 17 held-out hacks. In all 9 the GRAFT run's likelihood-ratio AUROC is above the no-routing run's, and in 7 by at least 0.10. The threshold of 17 has no justification beyond where the counts fall.
- The primary score (output norm) is below the no-routing run's in the informative splits wherever it can be computed (`results/summary.log`).

## What this adds up to

- Where routing works, it removes the hacks the monitor missed along with the ones it flagged, and so leaves almost nothing for a detector built from the forget adapter to find. The few that remain are too few to test detection with this design.
- The missed hack is in the forget adapter (it comes back with the forget adapter alone), so the adapter does hold information about hacks the monitor never flagged. Whether that can be read out as a test needs an environment where the deployed model still produces enough missed hacks, or prompts that elicit them.
- The output-norm score is retired for this purpose: it lost to the no-routing control here and failed its positive control on Qwen3-8B (`../qwen_bank/RESULTS.md`).

## Limits

- One environment and one model. The environment has two working hack words, so "13 splits" are 8 informative ones, in two families (route "actually" or route "indeed").
- The no-routing control is three runs, reused for every split.
- Seed 3 behaves differently in several splits (fewer flagged rollouts, more held-out hacks left). Not investigated.
- The secondary-score reading, the token-level breakdown and the count of 9 runs were all made after seeing results.
- The pilot machine shut itself down before its baseline files were copied. It was restarted for 15 minutes to fetch them. No run was repeated.
