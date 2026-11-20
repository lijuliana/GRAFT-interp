# First checks on released GRAFT adapters

19 November 2026. Weights and released evaluation files only. No model was run, because the laptop has under 0.4 GB of free disk and the base model is 16.4 GB.

## Summary

1. The forget adapter does not write along the activation-difference hack direction. Its write energy along that direction is 3.0 times that of a random direction in layers 16 to 20 (4 seeds, SEM 0.4), which is under 0.1% of its write energy, and adapters trained with no routing show the same value.
2. The forget adapter reads along that direction and the retain adapter does not. Read energy in layers 16 to 20 is 3.2 times random for the forget adapter (SEM 0.2) and 1.0 for the GRAFT retain adapter (SEM 0.04). Adapters trained with no routing read at 2.5 to 3.3.
3. With the forget adapter on, the model produces only the monitored hack: 0 unmonitored hacks in 2,304 released rollouts, against 73 in 2,304 with it off. This weakens direction 6.3 of the review and leaves 6.2 undecided.

Checks that need a forward pass (1b, 1c, 2 and 3 in the review) are not done. Section "Remaining checks" lists what each needs.

## Materials

The GRAFT authors did not publish adapter weights. All weights here come from the reproduction released with [@knoche2026loreft]: Qwen3-8B after the Countdown-Code supervised stage, with 64-neuron retain and forget adapters on every layer, 200 training steps.

| Group | Checkpoints | Role |
|---|---|---|
| GRAFT | `graft/cdhf100_gr_lccoh64_lr3`, seeds 1, 9, 15, 16 | Paper configuration with anchor |
| No routing | `graft-no-routing/cdhf100_noroute_anchor_lr1`, same seeds | Same two adapters, every gradient goes to both. Learning rate is 3 times higher |
| No intervention | `no-intervention/cdhf100_dn_lccoh64_lr3`, same seeds | One 128-neuron adapter, no routing, same learning rate as GRAFT |

Two hack directions come from the same release. The steering direction is the difference between mean residual activations of hacking and clean responses under prompts that request or forbid the hack, one vector per layer. The natural direction is the same difference at layer 20 for 553 hacking and 1,399 clean rollouts under the neutral prompt. Both follow the difference-of-means recipe of [@goodfire2026monitoring]. Their cosine is 0.79, which matches the value the release reports.

All hacks behind these directions are test overwrites, the hack type the monitor flags.

## Write directions and the hack direction

Each adapter neuron adds a fixed vector to the residual stream, scaled by its activation. The measure is the energy ratio E: the squared projection of the 64 write vectors onto a unit direction, divided by its expected value for a random direction. E equals 1 for a random direction.

| Layers 16 to 20, mean over 4 seeds (SEM) | E along steering direction | E along rank-1 forget directions |
|---|---|---|
| GRAFT forget | 3.0 (0.4) | 11.6 (1.4) |
| GRAFT retain | 1.7 (0.3) | 8.2 (1.4) |
| No routing, "forget" half | 3.0 (0.6) | 5.0 (0.6) |
| No routing, "retain" half | 2.9 (0.5) | 4.3 (0.6) |
| No intervention | 3.5 (0.7) | 5.3 (0.6) |

The 95th percentile of E over random directions is 2.1 to 2.6 for the forget adapter in these layers, so 3.0 is at the edge of chance. The largest absolute cosine between any single forget write vector and the steering direction is 0.05 to 0.06 in layers 16 to 22 (4-seed means), against 0.04 to 0.06 for the retain adapter and 0.04 for the largest of 64 random vectors. The cosine between the forget adapter's leading write direction and the natural direction is 0.010 to 0.027.

The release of [@knoche2026loreft] reports the same for rank-1 forget adapters at layer 20: cosines of -0.03 to 0.02 with the steering direction. This check reproduces that (0.03 to 0.07 in absolute value against the natural direction) and extends it to the paper's 64-neuron adapter at every layer.

The second column tests whether the 64-neuron forget adapter shares a direction with the eight rank-1 forget adapters. It does at 11.6 times random in layers 16 to 20, but the retain adapter reaches 8.2 and adapters trained with no routing reach 4 to 5. The shared component is a property of the layer 16 to 20 write space and is not specific to hacking.

Answer to the question in the idea list: the rank-1 forget direction and an activation-difference direction are different objects. They are close to orthogonal.

## Read directions and the hack direction

Each neuron reads the residual stream through two weight vectors (gate and up). The same energy ratio is computed for the gate vectors after multiplying by the layer's normalization gain, against the steering direction of the previous layer.

| Mean over 4 seeds (SEM) | Layers 11 to 15 | 16 to 20 | 21 to 25 | 26 to 35 |
|---|---|---|---|---|
| GRAFT forget | 1.6 (0.2) | 3.2 (0.2) | 3.1 (0.4) | 6.0 (0.5) |
| GRAFT retain | 0.67 (0.02) | 1.01 (0.04) | 0.98 (0.05) | 1.7 (0.04) |
| No routing, "forget" half | 1.9 (0.2) | 3.1 (0.4) | 2.9 (0.4) | 5.6 (0.6) |
| No routing, "retain" half | 2.1 (0.3) | 3.3 (0.3) | 3.2 (0.5) | 5.7 (0.4) |
| No intervention | 1.4 (0.1) | 2.5 (0.2) | 1.6 (0.1) | 2.7 (0.3) |

The 95th percentile for random directions is 1.3. The up vectors give the same pattern in every group (GRAFT forget 3.0, GRAFT retain 0.99 in layers 16 to 20).

Every adapter that learned the hack has input weights that respond to the hack direction. The GRAFT retain adapter is the one exception and sits at the value of random weights. This is a weight-level view of the localization the paper claims [@ward2026graft].

Limits of this result:

- E of 3 means 0.07% of read energy. The claim is the contrast between adapters, and it says nothing about how much of the forget adapter's activation the direction explains.
- Gate and up weights start random. A value of 1.0 for the retain adapter can mean its input weights did not move along this direction.
- The release's own analysis finds that the steering direction partly tracks whether the verifier code appears unchanged in the text, so the forget adapter may be reading that feature.
- The measure ignores the attention output of the same layer and the normalization scale.

## Agreement across seeds

The cosine between leading write directions of two seeds, layers 16 to 21, is 0.24 to 0.26 for forget adapters, 0.26 to 0.31 for GRAFT retain adapters, 0.27 to 0.31 with no intervention, and 0.16 to 0.22 with no routing. Chance is 0.016. Forget adapters are no more consistent across seeds than retain adapters, so the weights give no seed-independent hack write direction.

Within one run, the top eight forget and retain write directions overlap at 0.07 to 0.09 (chance 0.002). That is more than forget adapters overlap across seeds (0.03 to 0.04).

The forget adapter's write norm is about twice the retain adapter's at every layer and does not peak in layers 16 to 20, where the release's layer ablation places the behavioral effect. Its leading direction holds 38% to 50% of write energy in layers 16 to 21.

## Token readout

The leading two write directions of each adapter at layers 14 to 22 were projected through the final normalization gain and the unembedding matrix.

- Positive control: the steering direction at layer 30 gives "trivial", "stub", "always", "dummy", "bypass", "override" as its top tokens.
- Forget and retain directions give no coherent token sets. The kurtosis of their logit vectors is 3.2 to 3.3, the same as 20 random directions (mean 3.27, maximum 3.34).

Single seeds show suggestive tokens ("expect" for seed 15, "avoid" for seed 16 at layer 20), with no repeat across seeds. Vocabulary projection of the write weights does not read out the hack.

## Hack types with the forget adapter on and off

The release includes final evaluations of 256 rollouts per run at forget scale 0 (retain only) and 1 (both adapters) for nine GRAFT-family runs: four at the higher learning rate, four with no anchor, one with a rank-1 forget adapter.

| 9 runs, 2,304 rollouts per row | Monitored hacks | Unmonitored hacks |
|---|---|---|
| Forget adapter off | 38 | 73 |
| Forget adapter on | 866 | 0 |

Unmonitored hacks are constraint relaxation (most of the 73) and input modification. Seven of the nine runs have at least one with the forget adapter off. No run has one with it on. Four runs with no routing show at most 1 unmonitored hack in 256.

The same files give the hacks that remain in the deployed model and the share the monitor can see.

| Model | Rollouts | Hacks | Unmonitored | Monitor recall on hacks |
|---|---|---|---|---|
| No intervention, 12 runs | 3,072 | 1,522 | 0 | 100% |
| No routing, both halves on, 4 runs | 1,024 | 630 | 0 | 100% |
| GRAFT deployed, paper configuration, 4 runs | 1,024 | 11 | 5 | 55% |
| GRAFT deployed, 9 variant runs | 2,304 | 111 | 73 | 34% |

GRAFT cuts hacking from about 50% of rollouts to 3.7% (122 of 3,328 over the 13 runs). Of the 122 hacks that remain, the monitor can see 44 (36%). Unmonitored hacks appear only in GRAFT-trained models: 78 in 3,328 rollouts against 0 in 4,096 for models trained with no routing. The paper configuration has few (5 in 1,024). The four runs at the higher learning rate have 49 of the 78 and the four with no anchor have 24.

Two readings of the on and off contrast fit. Under displacement, the prompts where the retain-only model relaxes a constraint are the prompts where the full model overwrites the test. Under suppression, the forget adapter lowers the rate of the other hack types. The released files do not separate them, because they hold rates and no per-prompt outcomes.

Consequences for the review:

- Direction 6.3 proposed finding unmonitored hacks from differences between the both-adapters policy and the retain-only policy. On Countdown-Code the forget adapter adds only the monitored hack, so that difference would recover what the monitor already flags. The unmonitored hacks are in the retain adapter, as [@taylor2026headstart] reports (5.7% at deployment).
- Direction 6.2 needs the forget adapter to respond when it reads an unmonitored hack written by another policy. Its behavior gives no support for that. The read-weight result above is weak support. The teacher-forced test is still required.
- The review planned to take unmonitored hacks from the reward-penalty baseline. No penalty-baseline weights are released. The run named `countdown_code_hack_reward_penalty_amountmissing` is the no-intervention arm and has 0 unmonitored hacks. The released source is retain-only GRAFT at the higher learning rate, seed 1 (34 of 256).

## A proxy that failed

The layer-21 adapters were run on the released rollout-mean layer-20 activations to see whether the forget adapter's output norm separates hacking from clean rollouts. AUROC is 0.97 to 1.00 for forget adapters and 0.95 to 1.00 for retain adapters. With the input coordinates shuffled, the same weights give AUROCs anywhere from 0.00 to 1.00 over 50 shuffles. Rollout means separate under nearly any nonlinear feature (a held-out difference-of-means probe reaches 0.9999), so this test cannot distinguish the adapters. Per-token activations are needed.

## Remaining checks

All need the base model on a GPU with at least 24 GB of memory. None needs training. The model is 16.4 GB at full precision, so it does not fit on a 16 GB T4.

| Check | What to run | Informative outcome |
|---|---|---|
| 1b | Forget-neuron activations on cheating opportunities in formats unlike Countdown-Code | Activation above the retain adapter's and above matched prompts with no opportunity |
| 1c | Emergent-misalignment questions with the forget adapter on and off | A shift in misaligned-answer rate with the adapter on |
| 2 | Generate about 1,000 rollouts from retain-only GRAFT (higher learning rate, seed 1), grade them, replay through the full GRAFT model, score each token by forget output norm | AUROC on unmonitored hacks above 0.8, compared with a difference-of-means probe |
| 3 | Rate at which hacking rollouts state the hack in the reasoning block | A rate between 20% and 80% |

For check 3, the release notes that hacking rollouts write the plan to patch the test about 70 tokens before the end of the reasoning block. It gives no rate. If nearly all hacks are stated, a reasoning monitor would flag nearly all of them and experiment 6.1 would need a different environment.

## Reproduction

Scripts are in `checks/` and read every file into memory over HTTP. Requirements: Python 3 with numpy, about 3 GB of free memory, about 2.7 GB of network transfer.

```bash
cd checks
python3 probe_headers.py results/base_header.json
python3 weights_check.py results results/base_header.json
python3 summarize_weights.py results/weights_check.json
python3 monitor_proxy.py results results/base_header.json
python3 token_readout.py results results/base_header.json
python3 eval_hack_types.py results
```

`weights_check.py` writes `forget_top_dirs.npz` (29 MB), which `token_readout.py` reads. Result files are in `checks/results/`.

## Caveats

- Four seeds per group, one reproduction, one environment. The weights are not the GRAFT authors' own.
- The no-routing group uses a learning rate 3 times that of the GRAFT group. The no-intervention group matches the GRAFT learning rate and shows the same read-side pattern at lower strength.
- SEMs are across seeds after averaging layers within a window.
- Statements about what the release contains come from its README files and result tables, which I read. I did not rerun its experiments.
