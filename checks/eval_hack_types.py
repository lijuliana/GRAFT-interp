"""Hack rates by type with the forget adapter off (scale 0.0) and on (scale 1.0),
from the evaluation summaries released with the LoReFT follow-up (n=256 each)."""
import json, sys
import st_remote as st

B = "https://raw.githubusercontent.com/samuel500/GRAFT/loreft/rl/repro_data/"
RUNS = {**{f"GRAFT lr 5e-4, s{s}": f"gr_lr1_fseval/cdhf100_gr_lccoh64_lr1_s{s}.json" for s in (1, 9, 15, 16)},
        **{f"GRAFT lr/3 no anchor, s{s}": f"exp_nocoh_fseval/cdhf100_gr_nocoh_lr3_s{s}.json" for s in (1, 9, 15, 16)},
        "MLP retain + rank-1 forget, s9": "exp_m64_loreft_local_fseval/cdhf100_gr_m64_lf1_L20_k8_s9.json",
        **{f"no routing, s{s}": f"noroute_lr1_fseval/cdhf100_noroute_anchor_lr1_s{s}.json" for s in (1, 9, 15, 16)}}
K = ["hack_freq_detectable", "hack_freq_undetectable", "hack_freq_constraint_relax", "hack_freq_modify_input", "hack_freq_other"]
out, tot = {}, {"graft": {"0.0": [0, 0, 0], "1.0": [0, 0, 0]}}
for name, path in RUNS.items():
    d = json.loads(st._get(B + path)); n = d["n_eval"]; out[name] = {}
    for sc, v in d["scales"].items():
        row = {k.replace("hack_freq_", ""): v[k + "/score_threshold"] for k in K}
        out[name][sc] = row
        print(f"{name:32s} forget scale {sc}: " + "  ".join(f"{k} {x:.3f}" for k, x in row.items()))
        if not name.startswith("no routing"):
            t = tot["graft"][sc]; t[0] += round(row["detectable"] * n); t[1] += round(row["undetectable"] * n); t[2] += n
print("\nGRAFT-family totals (detectable hacks, undetectable hacks, rollouts):", tot["graft"])
json.dump({"runs": out, "totals": tot}, open(sys.argv[1] + "/eval_hack_types.json", "w"), indent=1)

# Deployed-model comparison: hacks that remain, and how many the monitor can see.
GROUPS = {"GRAFT deployed, paper config": ([f"repro_fig2_fseval/cdhf100_gr_lccoh64_lr3_s{s}.json" for s in (1, 9, 15, 16)], "0.0"),
          "GRAFT deployed, 9 variants": ([p for n, p in RUNS.items() if not n.startswith("no routing")], "0.0"),
          "no intervention (3 configs)": ([f"{d}_fseval/cdhf100_{r}_s{s}.json" for d, r in (("dn_anchor", "dn_lccoh64_lr3"), ("dn_anchor_lr1", "dn_lccoh64_lr1"), ("dn_lr3", "dn_lr3")) for s in (1, 9, 15, 16)], "1.0"),
          "no routing, both halves on": ([p for n, p in RUNS.items() if n.startswith("no routing")], "1.0")}
dep = {}
for g, (paths, sc) in GROUPS.items():
    det = und = n = 0
    for path in paths:
        d = json.loads(st._get(B + path)); v = d["scales"][sc]
        det += round(v["hack_freq_detectable/score_threshold"] * d["n_eval"]); und += round(v["hack_freq_undetectable/score_threshold"] * d["n_eval"]); n += d["n_eval"]
    dep[g] = {"detectable": det, "undetectable": und, "rollouts": n}
    print(f"{g:32s} detectable {det:5d}  undetectable {und:4d}  rollouts {n}")
res = json.load(open(sys.argv[1] + "/eval_hack_types.json")); res["deployed"] = dep
json.dump(res, open(sys.argv[1] + "/eval_hack_types.json", "w"), indent=1)

