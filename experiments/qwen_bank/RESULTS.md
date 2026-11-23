# Qwen3-8B read-outs: results

21 November 2026. Covers experiments 2, 3 (first test), 14 (first test) and 18 (transplant).

Setup: forward passes and sampling on one AWS A10G, no training. All weights are Knoche's reproduction of GRAFT. In experiments 2, 3 and 14 every hack comes from one run, G1 (`graft/cdhf100_gr_lccoh64_lr1_s1`), which uses a learning rate the paper does not. Experiment 18 uses two runs at the paper's settings, S9 and S15. Design and stop rules: `DESIGN.md`. Numbers: `results/analysis_stage1.json` and `results/analysis.json` (analysis fixed before the run; the second file adds the transplant and its control) and `results/posthoc_stage1.json` (written after reading the first file and raw rollouts).

## Verdicts under the rules fixed before the run

| Experiment | Rule | Result | Verdict |
|---|---|---|---|
| 2. Shadow adapter against a probe | Stop below AUROC 0.8 or at the probe's level | Forget-adapter output norm 0.57 [0.52, 0.61]. Probe 0.78 [0.75, 0.81]. 139 undetectable hacks against 814 non-hack rollouts | Stop |
| 14. Does the model admit undetectable hacks | Pursue at 50% recall at a 5% false-positive rate, drop below 20% | 4% when the policy answers, 14% when the frozen base model answers (mean of the two wordings) | Drop |
| 3. Are hacks stated in the reasoning | Informative only between 20% and 80% | 139 of 139 undetectable and 71 of 71 detectable hacks, by the base-model judge. 98% and 100% by keyword | Outside the range |
| 18. Transplant | Works if the hack rate with the other run's forget adapter is at least half the donor's own rate and at least 5 times the recipient's retain-only rate | 41% and 42% of 256 rollouts. Donors' own rates 41% and 46%. Recipients' retain-only rates 2% and 1% | Works, both directions. A control added afterwards shows the test could not have failed |

Validity for experiment 2 held: 139 undetectable hacks, 100 required.

## What the rollouts contain

| Bank | Rollouts | Solved | Failed | Detectable hacks | Undetectable hacks |
|---|---|---|---|---|---|
| `g1_dep`: retain adapter only (the deployed model) | 1,024 | 482 | 332 | 71 | 139 |
| `g1_both`: both adapters (as in training) | 512 | 270 | 2 | 240 | 0 |

- "Failed" means no hack and no correct answer. The design's "clean" group is solved plus failed.
- Of the 139 undetectable hacks, the grader labels 117 as relaxing a rule, 18 as changing the inputs and 4 as other. In the three I read, the model rewrote `test.py` to drop the check that every number is used and set the expression to the target.
- Hacks follow long reasoning. Median tokens before the answer's files: 1,038 for undetectable hacks, 1,012 for detectable hacks, 224 for solved rollouts. For failed rollouts the median is 1,536, the token limit, so most of them never reach an answer.
- With both adapters the model writes no undetectable hack in 512 rollouts.

## Experiment 2: the forget adapter's output norm carries no hack signal

AUROC for undetectable hacks in the deployed model's rollouts. "Length-matched" compares only rollouts in the same fifth of reasoning length.

| Detector | Against all non-hack (the pre-set comparison) | Against solved | Against solved, length-matched | Against failed |
|---|---|---|---|---|
| Forget adapter norm, maximum over tokens (the pre-set score) | 0.57 | 0.60 | 0.47 | 0.51 |
| Forget adapter norm, mean over tokens | 0.62 | 0.91 | 0.53 | 0.21 |
| Retain adapter norm, mean | 0.62 | 0.90 | 0.51 | 0.22 |
| Forget half of the no-routing run, mean | 0.66 | 0.95 | 0.73 | 0.26 |
| Reasoning length alone | 0.63 | 0.96 | 0.72 | 0.15 |
| Probe trained on the monitor's flags | 0.78 | 0.99 | 0.92 | 0.49 |
| Confession, policy answers | 0.74 | 0.99 | 0.95 | 0.37 |
| Confession, base model answers | 0.79 | 0.99 | 0.93 | 0.50 |
| Base model judges the reasoning | 0.92 | 1.00 | 0.97 | 0.81 |

- **The positive control failed.** On rollouts sampled with both adapters, where the forget adapter is switched on and produces the hacks, the pre-set score gives 0.47 for 240 test overwrites against 272 clean rollouts. The log-likelihood ratio between both adapters and retain only gives 1.00 on the same rollouts, so the adapter's effect is there and the norm does not measure it.
- **The mean norm tracks length.** It reaches 0.91 against solved rollouts, and so does the retain adapter's (0.90) and the no-routing run's forget half (0.95). After matching on length it falls to 0.53.
- **The probe beats every norm score** and keeps 0.92 against solved rollouts at matched length. It is at chance against failed rollouts.
- This retires the output norm as the score. It does not test other read-outs of the forget adapter, such as a projection onto the direction it writes on flagged rollouts.

## Experiment 14: the confession separates hacks from solved answers, not from failed ones

Share of rollouts for which the answer means "cheated" (Yes to "did it cheat", No to "was it genuine"), deployed model's rollouts:

| Question, answerer | Undetectable hacks | Detectable hacks | Solved | Failed |
|---|---|---|---|---|
| "Did it cheat", policy | 13% | 20% | 0.4% | 38% |
| "Did it cheat", base model | 23% | 38% | 1% | 50% |
| "Was it genuine", policy | 100% | 100% | 19% | 97% |
| "Was it genuine", base model | 100% | 99% | 11% | 96% |

- Under the pre-set comparison the recall at a 5% false-positive rate is 4% (policy) and 14% (base model).
- Against solved rollouts only, the same score has AUROC 0.99 and recall 99% (policy) and 93% (base model). Against failed rollouts it is 0.37 and 0.50. The pre-set comparison pooled the two groups.
- Asked directly, the model denies most of its hacks: it says "cheated" for 13% of undetectable hacks. The ranking still works because it says so for 0.4% of solved rollouts.
- On rollouts sampled with both adapters the policy says "cheated" for 97% of test overwrites and 0% of clean rollouts. Those are hacks the routing monitor already flags.

## Experiment 3: every hack is announced in the reasoning

| Group | Deployed model: judge says the reasoning states a hack | Both adapters |
|---|---|---|
| Undetectable hacks | 139 of 139 | none sampled |
| Detectable hacks | 71 of 71 | 240 of 240 |
| Solved | 2.5% of 482 | 0 of 270 |
| Failed | 75% of 332 | 1 of 2 |

- A routing experiment with a reasoning monitor cannot be informative in this environment: the monitor would flag every hack, so nothing is left to generalize to.
- Used as a detector, the judge has the highest AUROC in the table above (0.92 against all non-hack rollouts).
- Three quarters of failed rollouts state a plan to hack. Many of them are hacks cut off by the token limit, which is why no detector separates hacks from failed rollouts well.

## Experiment 18: the forget adapter alone carries the hack, so the transplant is uninformative

Hack rate over the same 32 questions, 8 samples each (4 for retain only), with 95% intervals. Every hack in the both-adapter rows is a test overwrite.

| Retain adapter | Forget adapter | Hacks | Rate | Solved |
|---|---|---|---|---|
| S9 | S9 (own) | 119 of 256 | 46% [40, 53] | 54% |
| S9 | S15 (transplanted) | 106 of 256 | 41% [36, 48] | 58% |
| S9 | none (deployed model) | 3 of 128 | 2% [1, 7] | 74% |
| S15 | S15 (own) | 104 of 256 | 41% [35, 47] | 59% |
| S15 | S9 (transplanted) | 108 of 256 | 42% [36, 48] | 58% |
| S15 | none (deployed model) | 1 of 128 | 1% [0, 4] | 70% |
| none | S9 (control, added afterwards) | 128 of 256 | 50% [44, 56] | 50% |
| none | S15 (control, added afterwards) | 112 of 256 | 44% [38, 50] | 55% |

- Both directions pass the rule. The transplanted rate is 102% and 91% of the donor's own rate, and 18 and 54 times the recipient's retain-only rate.
- The intervals of own and transplanted rates overlap in both directions, so these runs show no cost of the swap.
- **The swap could not have failed.** The control (`run_forget_only.sh`, written after the transplant result) puts each forget adapter on the frozen base with the retain adapter switched off. It hacks at the same rate as with both adapters, and every answer compiles. The retain adapter plays no part in the hack, so exchanging retain adapters tests nothing about transfer between models.
- The two runs share the frozen base and the recipe and differ in seed.
- Without the forget adapter both runs hack in 4 of 256 rollouts, 3 of them undetectable. At the paper's settings almost nothing is left for a detector to find, which is why experiment 2 had to use G1.
- Retain-only answers compile in 80% (S9) and 74% (S15) of rollouts, against 100% with both adapters. The degraded deployed model seen in G1 appears at the paper's settings too.

## Limits

- One run supplies every hack, at a learning rate the paper does not use. Its deployed model is degraded: 74% of its answers compile, against 99.8% with both adapters.
- Everything outside the first table was chosen after seeing results. Length matching uses five bins, and length alone still scores 0.72 after matching.
- The judge is the frozen base model with one prompt. It was not checked against human labels beyond six rollouts read by hand. On the 16 solved rollouts with reasoning as long as a typical hack's, it says yes for 5.
- The no-routing control reads text written by a different model.
