"""Project forget-adapter write directions through the final norm gain and the
unembedding (a direct 'logit lens' of the write direction). The unembedding is
read into memory in chunks; nothing is written to disk except a small JSON."""
import sys, json
import numpy as np
import st_remote as st, small_data as sd

OUT = sys.argv[1]
REPO = "https://huggingface.co/samuel500/graft-qwen3-8b-countdown-sft/resolve/a2d54fdf0aa9fa6ee8e487777ee7d63adaf8ec48/"
bh = json.load(open(sys.argv[2])); n, hdr = bh["n"], bh["hdr"]

v = hdr["lm_head.weight"]; s, e = v["data_offsets"]; base = 8 + n
CH = 48 << 20
WU = np.empty(v["shape"], dtype=np.float32); row = 0
for a in range(s, e, CH):
    b = min(a + CH, e)
    buf = st._get(REPO + "model.safetensors", base + a, base + b - 1)
    blk = (np.frombuffer(buf, dtype="<u2").astype(np.uint32) << 16).view(np.float32).reshape(-1, v["shape"][1])
    WU[row: row + len(blk)] = blk; row += len(blk)
    print(f"unembedding {100 * (b - s) / (e - s):.0f}%", flush=True)
assert row == v["shape"][0]
g = st.load_tensor(REPO + "model.safetensors", "model.norm.weight", hdr, n)
tok = json.loads(st._get(REPO + "tokenizer.json"))
id2tok = {i: t for t, i in tok["model"]["vocab"].items()}
for a in tok.get("added_tokens", []): id2tok[a["id"]] = a["content"]
show = lambda i: id2tok.get(int(i), f"<{i}>").replace("Ġ", "␣").replace("Ċ", "\\n")

def lens(d, k=12):
    lg = WU @ (g * (d / np.linalg.norm(d)).astype(np.float32))
    lg = lg - lg.mean()
    o = np.argsort(lg)
    return {"pos": [show(i) for i in o[::-1][:k]], "neg": [show(i) for i in o[:k]],
            "kurtosis": float(((lg / lg.std()) ** 4).mean())}

Z = np.load(OUT + "/forget_top_dirs.npz")
pps = sd.pps(); nat = sd.natural()
res = {"pps20": lens(pps["raw"][20]), "natural20": lens(nat["dir_dn"]), "pps30": lens(pps["raw"][30])}
rng = np.random.default_rng(0)
res["random"] = [lens(rng.standard_normal(4096))["kurtosis"] for _ in range(20)]
for run in [k for k in ("graft_s1", "graft_s9", "graft_s15", "graft_s16", "noroute_s9")]:
    for L in (14, 16, 18, 20, 22):
        for side in ("forget", "retain"):
            U = Z[f"{run}.{L}.{side}.U"]
            res[f"{run}.L{L}.{side}"] = [lens(U[:, j]) for j in range(2)]
json.dump(res, open(OUT + "/token_readout.json", "w"), ensure_ascii=False, indent=1)
print("done")
