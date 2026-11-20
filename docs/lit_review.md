# GRAFT follow-up directions: literature review and idea assessment

Date: 18 November 2026. Citation keys refer to `refs.bib` in this folder. Blog posts are marked as such in the bibliography and are not peer reviewed.

## Summary

1. Applying GRAFT to sycophancy has been done in supervised fine-tuning with perfect labels, as a baseline in [@africa2026neologism]. The forget adapter failed to hold sycophancy on one of two models. Nobody has published GRAFT on sycophancy in RL with a weak monitor, but a GRAFT co-author released the RL environment that would be used for it [@wong2026designing].
2. Nobody has published an interpretability study of the forget adapter. The GRAFT authors list it as follow-up work and one of them has promised a post on using the adapter as a probe [@hu2026thoughts].
3. The strongest open directions test GRAFT's central claim under conditions the paper did not cover: a monitor that reads the chain of thought, a monitor with false positives, and a behavior the base model already has. Section 6 lists five.

## 1. Scope

The question is which follow-ups to GRAFT [@ward2026graft] are open, and which are worth doing. In scope: gradient routing in language models, methods that train with a removable carrier of an unwanted trait, readouts of adapters and weight changes, and activation monitors for reward hacking. Out of scope: post-hoc unlearning and general reward-hacking benchmarks.

## 2. GRAFT and its measurement methods

GRAFT adds two parallel SwiGLU branches to every MLP of a frozen model: a retain adapter and a forget adapter (64 neurons per layer each for Qwen3-8B, 16 for SmolLM2-135M). Trajectories that a monitor flags update only the forget adapter, with the gradient doubled. All other trajectories update both. The forget adapter is deleted at deployment. An anchor environment trains the retain adapter alone. The monitor never changes the reward, so the policy is not rewarded for evading it [@ward2026graft].

Two corrections to the framing in the project notes. The paper is a MATS project with authors at MATS, Stanford, FAR AI, and Anthropic; one of seven authors is at Anthropic. The forget adapter is a gated nonlinear branch and is linear only in its read and write directions. The near-linear variant is the rank-1 LoReFT forget adapter at layer 20 from [@knoche2026loreft], and the Hugging Face repository `samuel500/graft-loreft-adapters` holds the weights for that post, so the two links refer to the same work.

Methods worth reusing:

- **Ground truth for detectable and undetectable hacks.** Each environment defines several forms of one hack and a monitor that flags only some of them. In `object_qa`, four intensifier words earn extra reward and the monitor misses "indeed". In Countdown-Code the monitor flags test functions replaced with a trivial pass and misses constraint removal and problem substitution. A perfect checker scores both kinds at evaluation [@ward2026graft].
- **Three evaluation configurations.** Both adapters on, retain only (deployment), and forget only. The gap between the first two measures how much of the behavior the forget adapter holds.
- **Hackable and unhackable prompts.** Half of the toy prompts admit no hack, which lets the target task keep being learned and gives a place to measure capability loss.
- **Share and concentration metrics.** [@taylor2026headstart] defines each adapter's hack share (its fraction of hacking above the base rate), learning share (its fraction of the KL shift from the base model), and per-layer ablation curves. They also train each adapter alone on unhackable prompts to rule out incoherence as the cause of a hack-rate drop.
- **Transfer metric.** [@drori2026unrouted] defines transfer as (ceiling − narrow) / (ceiling − floor), where narrow routes only a visible subset of hacks and the rate is measured on the invisible subset.
- **Swapped environments.** [@zhong2026baiting] flips which hack forms the monitor flags so that the common form is missed. GRAFT then leaves 21% to 70% hacking in the deployed model. These are ready-made failure cases.

## 3. Prior work

### 3.1 GRAFT follow-ups (October to November 2026)

| Work | What it did | Result |
|---|---|---|
| [@knoche2026loreft] | Replaced the forget adapter with a rank-1 LoReFT at layer 20 in Countdown-Code | 6 of 8 seeds with no hacking, at 0.029% of the forget parameters. Weights and a re-extracted steering vector are released |
| [@drori2026unrouted] | SFT on Terminal Wrench with Qwen3-32B, routing only a visible category of hacks | Transfer not significantly below 1 in 8 of 12 splits. Poor transfer from code tasks to ops tasks and for serious grader tampering. Routing on hacks with explicit hack reasoning gave silent hacking in about 15% of rollouts, which the author suspects is a bug |
| [@taylor2026headstart] | Reproduced the toy environments and tested what decides where hacking concentrates | Without routing, hacking does not reliably concentrate in one adapter. The adapter that learns faster takes the hack. No single layer of the forget adapter carries it |
| [@zhong2026baiting] | Rewards monitorable hacks on prompts with a trigger phrase so that GRAFT can route them | Deployed hack rate at most 9% in 10 of 11 environments. Worse on swapped Sort |
| [@hu2026thoughts] | A GRAFT author's list of open questions | Lists interpretability of the adapter, steering-like scaling, token-level routing, graded monitor confidence, and whether routing differs for newly learned capabilities and elicited propensities |
| [@wassname2026steering] | Predates GRAFT. Used steering vectors as a label-free routing signal with two LoRA adapters | Steering vectors are poor classifiers of on-policy hacks. Seeding the adapters with opposite signs of a hack direction cut held-out hacking by 63% |

Comments on the GRAFT announcement add two points. Knoche found that the no-intervention baseline used a learning rate three times higher than GRAFT, and matching it closes much of the task-performance gap. No GRAFT experiment has varied the monitor's false-positive rate; every experiment varies recall only [@wassname2026steering].

### 3.2 GRAFT and related methods on other behaviors

[@africa2026neologism] ran GRAFT as a baseline in SFT on three models of 27B to 32B parameters, flagging every harmful example. On emergent-misalignment tasks (harmful sports, finance, and medical advice), the forget adapter held the trait in all pairs and GRAFT leaked less than their method in 11 of 12 pairs. It kept much less of the desired trait on Gemma (13% against 89% on sports). On a sycophancy task (affirming a user's wrong GCD answer), the forget adapter did not hold the trait on Qwen: the sycophancy rate was 0.44 with the adapter and 0.53 without it. On OLMo it did (0.31 to 0.12). On harassment it contained the trait (0.04 against 0.20) with coherence loss on Gemma. GRAFT diverged or went incoherent in 3 of 17 further task and model pairs.

[@riche2026inoculation] trains the carrier adapter on the unwanted trait first, freezes it during task training, and discards it. They cover sycophancy, hate speech, a new capability, and emergent misalignment across nine setups. The GRAFT paper reports that GRAFT beats this method in its SFT experiment [@ward2026graft]. [@chen2025persona] does the same with a fixed steering vector, [@azarbal2025recontextualization] with prompts in RL, including a sycophancy setting. [@wang2026emergent] shows that sycophancy fine-tuning produces emergent misalignment and trains gates that suppress it.

[@wong2026designing] released an RL environment in which Qwen3-4B learns sycophancy. A medical question is followed by a leading question from the patient that is wrong half the time. The reward is 1.0 for agreeing with the physician reference and 0.5 for agreeing with a wrong patient. They test penalties from probes and LLM judges and find that the policy learns to evade both. They cite gradient routing as related work and do not run it.

### 3.3 Reading adapters and weight changes

Activation oracles are models trained to answer questions about injected activations [@karvonen2025activation]. Diff Interpretation Tuning trains an adapter that makes a fine-tuned model describe its own weight change [@goel2025learning]. Introspection adapters extend this across fine-tuning methods and reach 59% on AuditBench, but on a model trained to please a reward model they surfaced 16 of 52 behaviors and never the core objective [@shenoy2026introspection]. None of these has been applied to a GRAFT adapter.

### 3.4 Activation monitors for reward hacking

Goodfire builds a difference-of-means "cheating" direction from synthetic coding conversations and uses it as a probe that matches an LLM judge and finds hacks the judge misses [@goodfire2026monitoring]. A second paper penalizes that direction during RL. Honest solving stays at 96% to 98%, and the original probe's balanced accuracy falls from 0.98 to 0.57 in the worst run, recoverable to 0.97 by refitting [@goodfire2026shaping]. I could not find the phrase "knowledge of graders" in either paper. The first lists reasoning about hidden tests and recognizing evaluations among the hacks it found. The intended source may instead be Anthropic's reward-seeker post on grader awareness [@qi2026reward], which the GRAFT paper cites.

## 4. Gaps

- No published interpretability of the forget adapter beyond layer-window ablations.
- No test of the forget adapter as a detector of hacks its monitor never flagged. [@draganover2026sacred] proposes it and reports no experiment (6.6).
- No GRAFT run with a monitor that has false positives, an LLM judge, a probe, or a chain-of-thought monitor in RL.
- No explanation of why the forget adapter failed to hold sycophancy on Qwen.
- No RL experiment on a behavior other than reward hacking.
- No measurement of emergent misalignment with the forget adapter on and off in an RL-trained model.

## 5. Assessment of the listed ideas

### 5.1 Interpretability of the forget adapter

Open, and at risk of being done by the authors within weeks. A descriptive study of neurons is likely to find late-layer neurons that promote hack tokens, which is low in surprise. The version worth doing answers a question with ground truth: whether the adapter encodes a general disposition to cheat or a response tied to the training prompt format. The paper gives mixed evidence. An adapter trained on test replacement in Countdown-Code raises hard-coding in a LeetCode environment from 8.8% to 30.8%, a hack type never trained. The effect disappears in four of five prompt variants that differ more from the training format [@ward2026graft].

An oracle readout is a weak first tool here. Each adapter neuron has explicit read and write directions in the residual stream, so projection onto the vocabulary and onto known directions is direct, and introspection adapters missed the core objective on the closest tested model [@shenoy2026introspection]. Use an oracle as a comparison method.

First test, no training needed: load the released Qwen3-8B adapters, then measure (a) the cosine between forget write directions and a Goodfire-style cheating direction, with a matched-norm random baseline, (b) the adapter's activation on prompts that offer a cheating opportunity in formats unrelated to the training one, and (c) answers to the standard emergent-misalignment questions with the adapter on and off.

### 5.2 GRAFT on sycophancy

The plain application is taken in SFT and has an obvious owner in RL. The open contribution is the failure analysis.

Reward design, following the GRAFT problem setting:

| Component | Design |
|---|---|
| Proxy reward (trained on) | 1.0 if the answer matches the reference, 0.5 if it agrees with the user's wrong stated answer, as in [@wong2026designing] |
| Retain reward (evaluation only) | Correctness alone |
| Hackable prompts | Those where the user states a wrong belief. Prompts with no stated belief are unhackable |
| Monitor | Flags agreement with a wrong belief only when the belief is stated explicitly ("I think it is X") |
| Undetectable form | Agreement with a belief implied by the user's identity or emotional stake, with no explicit statement |
| Anchor | The same questions with the user's belief removed. This anchor is free of the hack by construction |
| Extra metric | Agreement rate when the user is right. A drop means the method produced a contrarian model |

A monitor that needs no answer key is also available for sycophancy: rerun the prompt with the user's belief flipped and flag the trajectory if the answer flips with it.

Predicted difficulty. In Countdown-Code a trajectory either hacks or solves. With sycophancy, agreeing with a user who is right is correct, is never flagged, and trains the retain adapter to agree. The retain adapter receives no gradient on the flagged cases, so on wrong-belief prompts it may fall back on "agree", and the deployed model may be more sycophantic than the base model. The second difficulty is that the base model is already sycophantic at a nonzero rate [@wong2026designing]. Deleting an adapter cannot remove a propensity that lives in the frozen weights. Both predict the Qwen failure in [@africa2026neologism], and separating them is the experiment.

### 5.3 Other listed ideas

| Idea | Status | Assessment |
|---|---|---|
| Persona and emergent-misalignment evaluation with the adapter on and off | Not done for RL-trained GRAFT models | Cheap. Run it on released weights as part of 5.1 |
| Goodfire probes and GRAFT | Not done | Use the probe as the routing monitor and test whether it stays valid. See 6.4 |
| Better hack direction by activation differences | [@knoche2026loreft] released a steering vector and layer-20 activations for comparing directions | Partly in progress. The both-adapters and retain-only models differ in one behavior, which gives a cleaner contrast than hack and non-hack transcripts |
| Inoculation adapters | Compared against GRAFT in SFT in both [@ward2026graft] and [@africa2026neologism] | Their gated variants are the source for self-routing in 6.2 |
| Neologism inoculation | Already includes GRAFT on six tasks | The source of the sycophancy failure in 5.2 |

## 6. New directions

Ranked by expected value per GPU hour. Each lists the claim, the competing accounts, and the first test that could end it.

### 6.1 Routing on a chain-of-thought monitor

Claim to test: when the monitor flags a hack only if the reasoning mentions it, the retain adapter learns to hack without mentioning it.

GRAFT's core claim is that the model does not learn to hide misbehavior because the monitor is absent from the reward. The retain adapter is, however, updated only on trajectories the monitor missed. When the missed trajectories are the same hack with the confession absent, every update the retain adapter receives on a hack is an update toward silent hacking. [@drori2026unrouted] saw 15% silent hacking in this setup in SFT and suspected a bug. The paper lists content-dependent labeling errors as an untested limitation.

- Account A, absorption dominates: the forget adapter takes both the reasoning and the action, and the deployed model's silent-hack rate stays at the base rate.
- Account B, selection dominates: the deployed model's silent-hack rate exceeds the base rate, and the fraction of its hacks that are unverbalized exceeds that of a model trained with no intervention.
- First test: measure how often Countdown-Code hack rollouts mention the hack in the thinking block. The experiment is informative only if this rate is between roughly 20% and 80%.
- Cost: about 9 Qwen3-8B runs of 2 to 4 hours each on one H200.

Either outcome is a result. Account A extends GRAFT to the monitors that labs use. Account B shows a case where routing produces unfaithful reasoning with no reward pressure.

### 6.2 The forget adapter as its own monitor

Claim to test: a score computed from the forget adapter detects hack forms its monitor never flagged, and routing on that score fixes the swapped environments where GRAFT fails.

Score candidates are the forget adapter's output norm per token and the per-token log-likelihood ratio between the both-adapters policy and the retain-only policy. Undetectable hacks are rare in GRAFT's own rollouts, so score them on trajectories from the reward-penalty baseline, which produces them at 26% to 45%, read through the GRAFT model with teacher forcing.

- First test: AUROC of the score on undetectable hacks against the perfect checker, compared with a difference-of-means probe [@goodfire2026monitoring]. Steering vectors reached at most 0.5 on a precision-weighted score in a similar role [@wassname2026steering]. If the adapter score is below about 0.8 AUROC, stop.
- If it passes: mask the retain adapter's gradient in proportion to the score during training. Evaluate on the swapped environments of [@zhong2026baiting], where GRAFT leaves 65% hacking on swapped Addition.
- Risk: the forget adapter absorbs the target task. The anchor environment and a cap on the routed fraction are the controls.

### 6.3 Hack discovery by policy differences

Claim to test: sampling the both-adapters and retain-only policies on the same prompts and ranking by reward gap surfaces environment loopholes that no monitor was told about.

This uses GRAFT as an environment debugging tool. The two policies were trained on the same trajectories and differ mainly in the routed behavior, so their differences exclude most task learning. A base model compared with a fine-tuned model does not have this property. Ground truth: plant several loopholes in one environment, give the monitor one, and count how many of the others appear in the top-ranked differences, against an LLM judge reading rollouts from an ordinarily trained policy.

### 6.4 Probe-routed GRAFT against probe penalties

Claim to test: a cheating-direction probe used as the routing monitor keeps its accuracy through training, while the same probe used as a penalty loses it.

[@goodfire2026shaping] and [@wong2026designing] both show that probes degrade under penalties. Probes also have false positives, which no GRAFT experiment has included. The measurement is the original probe's balanced accuracy after training under each use, plus task performance as the false-positive rate rises from 0% to 10%.

### 6.5 GRAFT models as ground truth for model-diffing methods

Claim to test: a GRAFT-trained model with both adapters merged gives a benchmark where the location of one learned behavior is known causally, and current diffing methods can be scored on recovering it.

The behavior was learned during RL alongside the task, which avoids the concern that behaviors implanted by narrow fine-tuning are unrealistically easy to find. A commenter on [@knoche2026loreft] reports unsuccessful crosscoder experiments on reward hacking. The limitation is that the forget adapter also holds some task skill (deleting it costs about 7 task points in the toy environments [@taylor2026headstart]), so ground truth must be per-neuron ablation effects.

### 6.6 The forget adapter as a held-out test

Added 20 November 2026.

[@draganover2026sacred] argues that a technique used to train alignment cannot also be trusted to test it, in the way a model cannot be scored on its training data. The post proposes keeping the forget adapter at inference and watching whether computation passes through it, and states that this is unexplored. In its comments, Knoche says a rank-1 forget adapter still works as a passive linear probe after removal and gives no detection numbers. The analysis page he links reports steering effects only. [@hu2026thoughts] promised a follow-up on the adapter as a probe. I found no published detection result as of this date.

The proposal needs three changes before it can be tested.

- Keeping the adapter active deploys a model that hacks in 25% to 49% of rollouts ([first_checks.md](first_checks.md)). The usable form is a shadow adapter: the retain-only model acts, the forget adapter reads the same residual stream, and its output is discarded.
- The adapter is never idle. The rank-1 forget edit is nonzero on every rollout and about 1.7 times larger on hacks, per the release of [@knoche2026loreft]. The test is a score with a threshold.
- The adapter is trained from the monitor's labels, so it is correlated with the monitor by construction. It is held out only to the extent that it responds to hacks the monitor missed.

Released evaluations show what the test would have to catch ([first_checks.md](first_checks.md)). Ordinary RL produces 1,522 hacks in 3,072 rollouts and the monitor can see all of them. The deployed GRAFT model produces 122 hacks in 3,328 rollouts and the monitor can see 44. Routing removes the hacks the monitor sees and leaves a model whose remaining hacks are 64% invisible to it. This is the post's objection in numbers, and it sets the bar: a shadow adapter is worth having only if it fires on the 64%. The forget adapter does not produce those hacks (0 in 2,304 rollouts with it on). Whether it recognizes them is untested.

Four experiments, ordered by cost.

**Shadow adapter against a label-matched probe.** Claim: the shadow forget score detects monitor-missed hacks better than a linear probe trained on the same monitor labels.

- Account A, training by doing: the adapter learned to produce the hack from policy gradients, so it responds to precursors shared across hack types. Its recall on undetectable hacks at a 1% false-positive rate exceeds the probe's.
- Account B, distilled monitor: the adapter encodes what the monitor flagged. Its recall on undetectable hacks equals the probe's and is near chance.
- The two detectors are different objects: forget write directions are close to orthogonal to the difference-of-means direction ([first_checks.md](first_checks.md)). Different generalization is therefore possible in either direction.
- First test, no training: check 2 of [first_checks.md](first_checks.md) with the probe added as baseline. Also report d-prime of the routing monitor and of the probe before and after GRAFT, against a reward-penalty run, since held-out monitors can degrade under penalties even when not trained against [@yudelson2026heldout]. This merges with 6.4.

**Cross-validated routing.** Claim: the fraction of unrouted hacks that a shadow adapter detects can be estimated by holding out hack types during routing, as test error is estimated by holding out data.

Split the detectable hack set into folds. Route on fold 1 only, score fold 2 hacks with the shadow adapter, then swap. This is the detection analogue of the transfer metric in [@drori2026unrouted], which measured removal. The six SmolLM2 environments of [@ward2026graft] have word-level ground truth and fit on one 16 GB GPU. The output is a held-out recall for a training technique, which is the quantity the post asks labs to report. Cost: about 36 short runs (6 environments, 2 folds, 3 seeds).

**A portable detector across training methods.** Claim: a forget adapter trained in one run detects hacks in a model trained by a different method on the same frozen base.

Every adapter attaches to the same frozen weights, so a forget adapter can be attached in shadow mode to any other fine-tune of that base. The target is a reward-penalty model, which hacks undetectably in 26% to 45% of rollouts [@ward2026graft; @taylor2026headstart]. A positive result gives a detector that the tested model's training never touched, which is the post's definition of held out. First step: seed A's forget adapter on seed B's retain-only model. Risk: the other run's retain adapter shifts the residual stream the detector reads.

**Detection where removal fails.** Claim: for a trait already present in the frozen weights, deleting the forget adapter does not remove the trait and the shadow adapter still detects it.

Section 5.2 predicts removal fails for sycophancy because the base model is already sycophantic, and [@africa2026neologism] observed that on Qwen (0.44 with the forget branch, 0.53 without). Detection needs only that the adapter's inputs separate sycophantic from honest responses. Removal failing while detection succeeds would make testing the main use of GRAFT for traits that predate fine-tuning. Setup: supervised fine-tuning with exact labels as in [@africa2026neologism], shadow score AUROC on held-out prompts and on implicit-belief prompts, probe baseline.

## 7. Recommended first steps

Three measurements need no training and decide which directions survive.

1. Load the released Qwen3-8B adapters and run the three checks in 5.1.
2. Compute the adapter-score AUROC on undetectable hacks (6.2). This also decides 6.3.
3. Measure the verbalization rate of hacks in Countdown-Code rollouts (6.1).

Status, 19 November 2026: the weight-level part of step 1 is done and reported in [first_checks.md](first_checks.md). The forget adapter's write directions are close to orthogonal to an activation-difference hack direction, its read weights respond to that direction, and the retain adapter's do not. Released evaluations show the forget adapter adds only the monitored hack, which weakens 6.3. The parts of steps 1 to 3 that need a forward pass are open.

## 8. Caveats

- The `GRAFT-interp` repository was empty when cloned, so nothing here reflects existing code.
- Blog-post results are single-author and mostly unreplicated. [@knoche2026loreft] states that an AI model ran and reported the experiments.
- I read [@ward2026graft], [@africa2026neologism], and [@riche2026inoculation] in full or in their main sections. For the Goodfire papers, [@wong2026designing], and the oracle papers I read the abstract and summaries of the main text.
- Entries marked "via GRAFT reference list" in `refs.bib` were taken from that list and not opened.
- The search covered arXiv, LessWrong, and lab blogs through 18 November 2026. Unpublished work by the GRAFT authors is likely to overlap with 5.1 and 5.2.

## Search log

Queries: GRAFT gradient routing follow-up; forget adapter interpretability probe monitor; gradient routing sycophancy emergent misalignment persona; Goodfire reward hacking grader knowledge; Goodfire mechanistic shaping; activation oracle and weight-difference interpretation of adapters; RL sycophancy environment monitor mitigation; steering vectors gradient routing. Added 20 November 2026: GRAFT forget adapter as inference-time monitor or probe.
