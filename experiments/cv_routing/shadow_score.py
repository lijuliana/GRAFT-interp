"""Score a trained toy GRAFT run with its forget adapter in shadow mode.

NOT YET EXECUTED: written without a GPU. Run from the rl/ directory of a
jnward/GRAFT checkout (commit 6da923d) so its modules import by bare name:

    python /path/to/shadow_score.py --checkpoint output/<sweep>/<run>/checkpoint-1000 \
        --config configs/cv_routing/<stem>.yaml --out scores/<run>.npz

Two sets of rollouts are sampled on upstream's fixed test prompts:
  policy 0: retain adapter only (the deployed model),
  policy 1: both adapters (what training rollouts look like; gives the probe
            and the in-fold check their monitor-flagged positives).
Every rollout is then scored in ONE condition: a forward pass with the forget
adapter's output discarded (scales 1, 0). Hooks read each adapter's MLP input
and compute what the forget adapter would have written. A second pass at
scales (1, 1) gives the log-likelihood ratio of the sampled tokens.
"""
import argparse, os, sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.getcwd())
from transformers import AutoTokenizer  # noqa: E402

from envs import get_env  # noqa: E402
from eval_utils import _tokenize_prompts_for_vllm, generate_from_model, load_gradient_routing_model  # noqa: E402
from gradient_routing import DualMLPAdapter, set_scales  # noqa: E402


def word_lists(cfg):
    p = cfg["rh_detector"]["params"]
    routed = p.get("words") or p.get("phrases")
    q = cfg["hack_freq_detector"]["params"]
    allw = q.get("words") or q.get("phrases")
    return [w.lower() for w in routed], [w.lower() for w in allw if w not in routed]


def first_hit_token(tok, ids, words):
    """Index of the first completion token after which the decoded text contains a hack word; -1 if none."""
    for t in range(1, len(ids) + 1):
        text = tok.decode(ids[:t], skip_special_tokens=True).lower()
        if any(w in text for w in words):
            return t - 1
    return -1


@torch.no_grad()
def score(model, tok, rows, probe_layer, bs, device):
    adapters = [m for m in model.modules() if isinstance(m, DualMLPAdapter)]
    cache = {}
    hooks = [m.register_forward_pre_hook(lambda mod, inp, i=i: cache.__setitem__(i, inp[0]))
             for i, m in enumerate(adapters)]
    T = max(len(r["completion_ids"]) for r in rows)
    n = len(rows)
    f_norm = np.zeros((n, T), np.float32); r_norm = np.zeros((n, T), np.float32)
    llr = np.zeros((n, T), np.float32); mask = np.zeros((n, T), bool)
    resid = None
    prompt_ids = _tokenize_prompts_for_vllm(tok, [r["prompt"] for r in rows])
    for a in range(0, n, bs):
        idx = range(a, min(a + bs, n))
        seqs = [prompt_ids[i] + rows[i]["completion_ids"] for i in idx]
        L = max(map(len, seqs))
        ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for j, s in enumerate(seqs):  # right padding: causal attention keeps real positions exact
            ids[j, :len(s)] = torch.tensor(s); att[j, :len(s)] = 1
        ids, att = ids.to(device), att.to(device)

        set_scales(model, 1.0, 0.0)
        out0 = model(input_ids=ids, attention_mask=att, output_hidden_states=True)
        fsum = sum(m._forget_core(cache[i]).float().norm(dim=-1) for i, m in enumerate(adapters))
        rsum = sum(m._retain_core(cache[i]).float().norm(dim=-1) for i, m in enumerate(adapters))
        lp0 = torch.log_softmax(out0.logits.float(), -1)
        hs = out0.hidden_states[probe_layer].float()
        set_scales(model, 1.0, 1.0)
        lp1 = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits.float(), -1)
        set_scales(model, 1.0, 0.0)

        if resid is None:
            resid = np.zeros((n, hs.shape[-1]), np.float32)
        for j, i in enumerate(idx):
            p, c = len(prompt_ids[i]), len(rows[i]["completion_ids"])
            if c == 0:
                continue
            pos = torch.arange(p, p + c, device=device)          # positions of completion tokens
            tgt = ids[j, pos]
            # token at position t is predicted from position t - 1
            llr[i, :c] = (lp1[j, pos - 1, tgt] - lp0[j, pos - 1, tgt]).cpu().numpy()
            # adapter activity while reading up to and including each completion token
            f_norm[i, :c] = fsum[j, pos].cpu().numpy(); r_norm[i, :c] = rsum[j, pos].cpu().numpy()
            mask[i, :c] = True
            resid[i] = hs[j, pos].mean(0).cpu().numpy()
    for h in hooks:
        h.remove()
    return f_norm, r_norm, llr, mask, resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_prompts", type=int, default=256)   # upstream's eval set size
    ap.add_argument("--reps", type=int, default=32)         # samples per prompt and policy
    ap.add_argument("--probe_layer", type=int, default=15)  # of 30 blocks in SmolLM2-135M
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    routed, heldout = word_lists(cfg)
    tr = cfg["training"]
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(tr["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_gradient_routing_model(a.checkpoint, base_model=tr["model"]).to(device).eval()

    env_args = argparse.Namespace(hack_frac=0.5, unconditional_hackable=False)
    data = get_env(tr["environment"]).load_eval_prompts(a.n_prompts, env_args)
    prompts = [d["prompt"] for d in data] * a.reps
    hackable = np.array([bool(d.get("hackable", True)) for d in data] * a.reps)

    rows, policy = [], []
    for pol, scales in ((0, (1.0, 0.0)), (1, (1.0, 1.0))):
        set_scales(model, *scales)
        got = generate_from_model(model, tok, n_samples=len(prompts), max_new_tokens=tr["max_completion_length"],
                                  temperature=1.0, prompts=prompts, batch_size=a.batch_size)
        rows += got; policy += [pol] * len(got)

    text = [r["completion"].lower() for r in rows]
    hit_r = np.array([any(w in t for w in routed) for t in text])
    hit_h = np.array([any(w in t for w in heldout) for t in text])
    first = np.array([first_hit_token(tok, r["completion_ids"], routed + heldout) for r in rows])
    f_norm, r_norm, llr, mask, resid = score(model, tok, rows, a.probe_layer, a.batch_size, device)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(
        a.out, f_norm=f_norm, r_norm=r_norm, llr=llr, mask=mask, resid=resid,
        policy=np.array(policy), hit_routed=hit_r, hit_heldout=hit_h, first_hit=first,
        hackable=np.concatenate([hackable, hackable]),
        routed=np.array(routed), heldout=np.array(heldout), run=os.path.basename(os.path.dirname(os.path.abspath(a.checkpoint))),
        env=tr["environment"], n_prompts=a.n_prompts, reps=a.reps)
    dep = np.array(policy) == 0
    print(f"deployed rollouts {dep.sum()}: routed-word hacks {int((hit_r & dep).sum())}, "
          f"held-out-only hacks {int((hit_h & ~hit_r & dep).sum())}, clean {int((~hit_h & ~hit_r & dep).sum())}")


if __name__ == "__main__":
    main()
