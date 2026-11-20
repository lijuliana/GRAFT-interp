"""Check 1(a): do GRAFT forget-adapter weights align with activation-difference
hack directions? Weights only; everything is read into memory.

Energy ratio E(M, v) = ||M^T v_hat||^2 / (||M||_F^2 / d). 1.0 = an isotropic
(random) direction. Null distribution: random unit vectors against the same M.
"""
import sys, json, time
import numpy as np
import st_remote as st, small_data as sd

OUT = sys.argv[1]
REPRO = "https://huggingface.co/samuel500/graft-repro-adapters/resolve/main/"
LOREFT = "https://huggingface.co/samuel500/graft-loreft-adapters/resolve/main/"
BASE = "https://huggingface.co/samuel500/graft-qwen3-8b-countdown-sft/resolve/a2d54fdf0aa9fa6ee8e487777ee7d63adaf8ec48/model.safetensors"
SEEDS = [1, 9, 15, 16]
RUNS = {f"graft_s{s}": REPRO + f"graft/cdhf100_gr_lccoh64_lr3_s{s}/model.safetensors" for s in SEEDS}
RUNS.update({f"noroute_s{s}": REPRO + f"graft-no-routing/cdhf100_noroute_anchor_lr1_s{s}/model.safetensors" for s in SEEDS})
RUNS.update({f"noint_s{s}": REPRO + f"no-intervention/cdhf100_dn_lccoh64_lr3_s{s}/model.safetensors" for s in SEEDS})
SIDES = lambda run: ("retain",) if run.startswith("noint") else ("forget", "retain")
NL, D = 36, 4096
rng = np.random.default_rng(0)
unit = lambda v: v / np.linalg.norm(v)

pps = sd.pps(); nat = sd.natural()
vL = {L: unit(pps["raw"][L].astype(np.float64)) for L in range(NL)}
dn = unit(nat["dir_dn"].astype(np.float64))
RAND = rng.standard_normal((D, 500)); RAND /= np.linalg.norm(RAND, axis=0)

bh = json.load(open(sys.argv[2])); bn, bhdr = bh["n"], bh["hdr"]
gain = {L: st.load_tensor(BASE, f"model.layers.{L}.post_attention_layernorm.weight", bhdr, bn).astype(np.float64) for L in range(NL)}
print("norm gains loaded", flush=True)

# rank-1 LoReFT forget write directions at the output of layer 20 (8 seeds)
lor = {}
for s in [1, 2, 3, 4, 5, 9, 15, 16]:
    u = LOREFT + f"mlp-loreft/cdhf100_gr_m64_lf1_L20_k8_s{s}/model.safetensors"
    lor[s] = unit(st.load_tensor(u, "_orig_mod.model.layers.20.loreft.rot_forget")[0].astype(np.float64))
print("loreft |cos| with dir_dn:", [round(float(abs(lor[s] @ dn)), 3) for s in lor], flush=True)

def energy(M, v):  # M: d x k (columns are directions), v: unit d-vector(s)
    return (np.linalg.norm(M.T @ v, axis=0) ** 2) / (np.linalg.norm(M) ** 2 / D)

res, top = {}, {}
for run, url in RUNS.items():
    t = time.time(); W = st.load_all(url)
    r = {}
    for L in range(NL):
        for side in SIDES(run):
            k = f"_orig_mod.model.layers.{L}.mlp."
            Dn = W[k + f"down_{side}.weight"].astype(np.float64)           # 4096 x 64, write
            G = W[k + f"gate_{side}.weight"].astype(np.float64) * gain[L]   # 64 x 4096, read (residual space up to rms scale)
            U = W[k + f"up_{side}.weight"].astype(np.float64) * gain[L]
            Uw, sv, _ = np.linalg.svd(Dn, full_matrices=False)
            null = energy(Dn, RAND)
            e = {
                "fro": float(np.linalg.norm(Dn)),
                "pr": float(sv.sum() ** 2 / (sv ** 2).sum()),           # participation ratio of singular values (1..64)
                "top1_share": float(sv[0] ** 2 / (sv ** 2).sum()),
                "E_write_ppsL": float(energy(Dn, vL[L])),
                "E_write_pps20": float(energy(Dn, vL[20])),
                "E_write_dn20": float(energy(Dn, dn)),
                "E_write_null95": float(np.quantile(null, 0.95)),
                "E_write_null99": float(np.quantile(null, 0.99)),
                "maxcos_write_ppsL": float(np.abs((Dn / (np.linalg.norm(Dn, axis=0) + 1e-12)).T @ vL[L]).max()),
                "maxcos_write_dn20": float(np.abs((Dn / (np.linalg.norm(Dn, axis=0) + 1e-12)).T @ dn).max()),
                "E_read_gate": float(energy(G.T, vL[max(L - 1, 0)])),
                "E_read_up": float(energy(U.T, vL[max(L - 1, 0)])),
                "E_read_null95": float(np.quantile(energy(G.T, RAND), 0.95)),
                "E_write_loreft": float(np.mean([energy(Dn, lor[s]) for s in lor])),
            }
            r[f"{L}.{side}"] = e
            top[(run, L, side)] = (Uw[:, :8].astype(np.float32), sv[:8].astype(np.float32))
        # overlap of forget and retain write subspaces within a run (top-8 each); chance = 8/4096
        if len(SIDES(run)) == 2:
            a, b = top[(run, L, "forget")][0], top[(run, L, "retain")][0]
            r[f"{L}.fr_overlap8"] = float(np.linalg.norm(a.T @ b) ** 2 / 8)
    res[run] = r
    json.dump(res, open(OUT + "/weights_check_partial.json", "w"))
    del W
    print(run, f"{time.time()-t:.0f}s", flush=True)

# cross-seed agreement of the top write direction and top-8 subspace
cross = {}
for arm in ("graft", "noroute", "noint"):
    for side in SIDES(arm):
        for L in range(NL):
            c1, c8 = [], []
            for i, s in enumerate(SEEDS):
                for t2 in SEEDS[i + 1:]:
                    a = top[(f"{arm}_s{s}", L, side)][0]; b = top[(f"{arm}_s{t2}", L, side)][0]
                    c1.append(abs(float(a[:, 0] @ b[:, 0]))); c8.append(float(np.linalg.norm(a.T @ b) ** 2 / 8))
            cross[f"{arm}.{side}.{L}"] = {"top1_abs_cos": float(np.mean(c1)), "top8_overlap": float(np.mean(c8))}
# cross-arm: graft forget vs no-intervention adapters (same seed), top-8
for L in range(NL):
    v = [float(np.linalg.norm(top[(f"graft_s{s}", L, "forget")][0].T @ top[(f"noint_s{s}", L, "retain")][0]) ** 2 / 8) for s in SEEDS]
    cross[f"graftF_vs_noint.{L}"] = float(np.mean(v))

json.dump({"res": res, "cross": cross}, open(OUT + "/weights_check.json", "w"))
np.savez_compressed(OUT + "/forget_top_dirs.npz", **{f"{run}.{L}.{side}.U": top[(run, L, side)][0] for run in RUNS for L in range(12, 24) for side in SIDES(run)},
                    **{f"{run}.{L}.{side}.s": top[(run, L, side)][1] for run in RUNS for L in range(12, 24) for side in SIDES(run)})
print("done")
