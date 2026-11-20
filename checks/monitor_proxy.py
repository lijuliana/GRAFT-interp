"""Proxy for check 2: does the layer-21 forget adapter respond more strongly to
hack rollouts than clean ones? Input is the released rollout-mean layer-20
residual (SFT-base replay), passed through layer 21's RMSNorm gain. Missing:
layer-21 attention output, per-token inputs, upstream adapters."""
import sys, json
import numpy as np
import st_remote as st, small_data as sd

OUT = sys.argv[1]
REPRO = "https://huggingface.co/samuel500/graft-repro-adapters/resolve/main/"
BASE = "https://huggingface.co/samuel500/graft-qwen3-8b-countdown-sft/resolve/a2d54fdf0aa9fa6ee8e487777ee7d63adaf8ec48/model.safetensors"
SEEDS = [1, 9, 15, 16]; L = 21
RUNS = {f"graft_s{s}": REPRO + f"graft/cdhf100_gr_lccoh64_lr3_s{s}/model.safetensors" for s in SEEDS}
RUNS.update({f"noroute_s{s}": REPRO + f"graft-no-routing/cdhf100_noroute_anchor_lr1_s{s}/model.safetensors" for s in SEEDS})
rng = np.random.default_rng(0)

def auroc(pos, neg):
    x = np.concatenate([pos, neg]); r = np.argsort(np.argsort(x)) + 1.0
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

nat = sd.natural(); H, C = nat["hack"].astype(np.float64), nat["clean"].astype(np.float64)
bh = json.load(open(sys.argv[2]))
g = st.load_tensor(BASE, f"model.layers.{L}.post_attention_layernorm.weight", bh["hdr"], bh["n"]).astype(np.float64)
norm = lambda X: X / np.sqrt((X ** 2).mean(1, keepdims=True) + 1e-6) * g
XH, XC = norm(H), norm(C)
silu = lambda z: z / (1 + np.exp(-z))

# reference: difference-of-means probe, fit on even rows, scored on odd rows
d = H[::2].mean(0) - C[::2].mean(0)
ref = auroc(H[1::2] @ d, C[1::2] @ d)
out = {"n_hack": len(H), "n_clean": len(C), "diff_of_means_heldout_auroc": ref, "runs": {}}
print("diff-of-means held-out AUROC", round(ref, 4))

def score(G, U, Dn, X):
    return np.linalg.norm((silu(X @ G.T) * (X @ U.T)) @ Dn.T, axis=1)

for run, url in RUNS.items():
    n, hdr = st.header(url); r = {}
    for side in ("forget", "retain"):
        k = f"_orig_mod.model.layers.{L}.mlp."
        G, U, Dn = (st.load_tensor(url, k + f"{p}_{side}.weight", hdr, n).astype(np.float64) for p in ("gate", "up", "down"))
        sh, sc = score(G, U, Dn, XH), score(G, U, Dn, XC)
        nulls = []
        for _ in range(50):  # same weights, input coordinates shuffled
            p = rng.permutation(4096)
            nulls.append(auroc(score(G, U, Dn, XH[:, p]), score(G, U, Dn, XC[:, p])))
        nulls = np.array(nulls)
        r[side] = {"auroc": auroc(sh, sc), "ratio_hack_over_clean": float(sh.mean() / sc.mean()),
                   "null_auroc_mean": float(nulls.mean()), "null_auroc_min": float(nulls.min()), "null_auroc_max": float(nulls.max())}
    out["runs"][run] = r
    print(run, {s: {k: round(v, 3) for k, v in r[s].items()} for s in r}, flush=True)
json.dump(out, open(OUT + "/monitor_proxy.json", "w"))
