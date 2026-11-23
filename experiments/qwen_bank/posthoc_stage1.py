"""Post hoc checks on the stage-1 read-outs (numpy only).   python posthoc_stage1.py <out dir>

None of this was fixed before the run. It was written after reading analysis.json and raw rollouts, to
answer two questions the pre-set analysis leaves open:

  1. Length. Hacks in the deployed model's rollouts follow much longer reasoning than solved rollouts.
     How much of each detector's AUROC does reasoning length alone give?
  2. What "clean" holds. Non-hack rollouts are solved answers plus failed ones (mostly cut off at the
     token limit). Each detector is scored against the two groups separately.

Writes <out>/posthoc_stage1.json.
"""
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cv_routing"))
from analyze import auroc, fit_probe, recall_at_fpr  # noqa: E402
from analyze_bank import labels, per_rollout  # noqa: E402


def groups(z):
    lab = labels(z); solved = z["true"] > 0
    return {"undet": lab["undet"], "det": lab["det"], "clean": lab["clean"],
            "solved": lab["clean"] & solved, "failed": lab["clean"] & ~solved}


def scores(z):
    """Rollout-level scores, including per-token means that the pre-set analysis did not compute."""
    s = per_rollout(z); n = len(z["n_tok"]); off = np.concatenate([[0], np.cumsum(z["n_tok"])])
    extra = {k: np.zeros(n) for k in ("shadow_pre_mean", "retain_mean", "llr_mean", "llr_pre_mean")}
    for i in range(n):
        a, b = off[i], off[i + 1]
        if b == a:
            continue
        at = max(int(z["action_tok"][i]), 1)
        extra["shadow_pre_mean"][i] = z["f_norm"][a:b][:at].mean(); extra["retain_mean"][i] = z["r_norm"][a:b].mean()
        extra["llr_mean"][i] = z["llr"][a:b].mean(); extra["llr_pre_mean"][i] = z["llr"][a:b][:at].mean()
    s.update(extra)
    s["len_response_tok"] = z["n_tok"].astype(float); s["len_before_files_tok"] = z["action_tok"].astype(float)
    return s


def matched_auroc(v, pos, neg, length, bins=5):
    """AUROC within quintiles of length (pooled over pos and neg), weighted by the number of pairs."""
    m = pos | neg; edges = np.quantile(length[m], np.linspace(0, 1, bins + 1)[1:-1])
    b = np.digitize(length, edges); num = den = 0.0
    for k in range(bins):
        p, q = pos & (b == k), neg & (b == k)
        if p.sum() and q.sum():
            w = p.sum() * q.sum(); num += w * auroc(v[p], v[q]); den += w
    return float(num / den) if den else float("nan")


def table(s, g, positives, negatives, length):
    out = {}
    for name, v in s.items():
        row = {}
        for p in positives:
            for q in negatives:
                if g[p].sum() >= 5 and g[q].sum() >= 5:
                    a = auroc(v[g[p]], v[g[q]])
                    row[f"{p}_vs_{q}"] = {"auroc": a, "auroc_length_matched": matched_auroc(v, g[p], g[q], length),
                                          "recall_at_5pct_fpr": recall_at_fpr(v[g[p]], v[g[q]], 0.05),
                                          "recall_at_5pct_fpr_flipped": recall_at_fpr(-v[g[p]], -v[g[q]], 0.05),
                                          "n_pos": int(g[p].sum()), "n_neg": int(g[q].sum())}
        out[name] = row
    return out


def main(out):
    L = lambda n: dict(np.load(f"{out}/scores/{n}.npz", allow_pickle=False))
    dep, trn, ctl = L("dep"), L("trn"), L("dep_ctrl")
    gd, gt = groups(dep), groups(trn); sd, st = scores(dep), scores(trn)
    for key in ("resid_pre", "resid_all"):
        sd[f"probe_{key}"] = fit_probe(trn[key].astype(np.float64), gt["det"])(dep[key].astype(np.float64))
    for k, v in scores(ctl).items():
        if k.startswith("shadow"):
            sd["noroute_" + k] = v
    for who in ("policy", "base"):
        for z, s in ((dep, sd), (trn, st)):
            s[f"confess_cheat_{who}"] = z[f"confess_cheat_{who}"]; s[f"confess_honest_{who}"] = z[f"confess_honest_{who}"]
            s[f"confess_mean_{who}"] = (z[f"confess_cheat_{who}"] + z[f"confess_honest_{who}"]) / 2
    sd["verbal_judge"] = dep["verbal_judge"]; st["verbal_judge"] = trn["verbal_judge"]

    res = {"note": "post hoc, not pre-set",
           "group_sizes": {"dep": {k: int(v.sum()) for k, v in gd.items()}, "trn": {k: int(v.sum()) for k, v in gt.items()}},
           "median_tokens_before_files": {b: {k: float(np.median(z["action_tok"][v])) if v.sum() else None for k, v in g.items()}
                                          for b, z, g in (("dep", dep, gd), ("trn", trn, gt))},
           "dep": table(sd, gd, ("undet", "det"), ("clean", "solved", "failed"), dep["action_tok"].astype(float)),
           "trn": table(st, gt, ("det",), ("clean",), trn["action_tok"].astype(float))}
    says = lambda z, g, key: {k: float((z[key][v] > 0).mean()) if v.sum() else None for k, v in g.items()}
    res["yes_rates"] = {b: {key: says(z, g, key) for key in ("verbal_judge", "confess_cheat_policy", "confess_honest_policy",
                                                               "confess_cheat_base", "confess_honest_base")}
                        for b, z, g in (("dep", dep, gd), ("trn", trn, gt))}
    # does the reasoning judge follow length? solved rollouts whose reasoning is at least as long as the median hack's
    long_solved = gd["solved"] & (dep["reason_chars"] >= np.median(dep["reason_chars"][gd["undet"] | gd["det"]]))
    res["verbal_judge_long_solved"] = {"n": int(long_solved.sum()), "judge_yes": float((dep["verbal_judge"][long_solved] > 0).mean()) if long_solved.sum() else None}
    json.dump(res, open(f"{out}/posthoc_stage1.json", "w"), indent=1)

    f = lambda r, k: f"{r[k]['auroc']:.2f}/{r[k]['auroc_length_matched']:.2f}" if k in r else "    -    "
    print("groups", json.dumps(res["group_sizes"])); print("median tokens before files", json.dumps(res["median_tokens_before_files"]))
    print("\ndep: AUROC / length-matched AUROC.  columns: undet vs clean, undet vs solved, undet vs failed, det vs solved")
    for k, r in res["dep"].items():
        print(f"  {k:26s} {f(r, 'undet_vs_clean')}  {f(r, 'undet_vs_solved')}  {f(r, 'undet_vs_failed')}  {f(r, 'det_vs_solved')}")
    print("\ntrn: det vs clean, AUROC / length-matched AUROC")
    for k, r in res["trn"].items():
        print(f"  {k:26s} {f(r, 'det_vs_clean')}")
    print("\nyes rates", json.dumps({b: {k: {g: (round(x, 3) if x is not None else None) for g, x in v.items()} for k, v in d.items()}
                                     for b, d in res["yes_rates"].items()}, indent=1))
    print("judge on long solved rollouts", json.dumps(res["verbal_judge_long_solved"]))


if __name__ == "__main__":
    main(sys.argv[1])
