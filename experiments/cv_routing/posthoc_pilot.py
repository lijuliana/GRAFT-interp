"""Post hoc checks on the pilot's secondary score (numpy only).   python posthoc_pilot.py scores/cvr_pilot/*.npz --out posthoc.json

None of this was fixed before the run. The pilot's primary score (forget adapter output norm) hit the
stop rule. The secondary score written in DESIGN.md, the log-likelihood ratio between both adapters and
retain only (llr), separated held-out hacks well in the GRAFT runs. These checks ask whether that
separation survives three objections:

  1. Length. llr is a sum over tokens, so a difference in response length could produce it.
     -> llr per token, response length alone, and AUROC within fifths of response length.
  2. Reading the word. The score could come only from the hack token's own prediction.
     -> llr over tokens strictly before the hack token, at the hack token alone, and after it.
        Clean rollouts are cut at positions drawn from the hack rollouts' positions, as in analyze.py.
  3. Few positives per seed. -> pooled over seeds, with thresholds set within each seed.
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import RUN_RE, auroc, boot_ci, recall_at_fpr  # noqa: E402


def matched_auroc(v, pos, neg, length, bins=5):
    m = pos | neg; edges = np.quantile(length[m], np.linspace(0, 1, bins + 1)[1:-1])
    b = np.digitize(length, edges); num = den = 0.0
    for k in range(bins):
        p, q = pos & (b == k), neg & (b == k)
        if p.sum() and q.sum():
            w = p.sum() * q.sum(); num += w * auroc(v[p], v[q]); den += w
    return float(num / den) if den else float("nan")


def scores(z, rng):
    mask = z["mask"].astype(bool); T = mask.shape[1]; t = np.arange(T)[None, :]
    llr = np.where(mask, z["llr"], 0.0); n_tok = mask.sum(1)
    first = z["first_hit"].astype(int).copy(); has = first >= 0
    if has.any():
        first[~has] = rng.choice(first[has], (~has).sum())
    else:
        first[~has] = T
    f = first[:, None]
    mean = lambda a, m: np.where(m.any(1), np.where(m, a, 0.0).sum(1) / np.maximum(m.sum(1), 1), 0.0)
    return {"llr_sum": llr.sum(1), "llr_per_token": llr.sum(1) / np.maximum(n_tok, 1),
            "llr_strictly_before": np.where(mask & (t < f), z["llr"], 0.0).sum(1),
            "llr_at_cut_token": np.where(mask & (t == f), z["llr"], 0.0).sum(1),
            "llr_after": np.where(mask & (t > f), z["llr"], 0.0).sum(1),
            "llr_max_token": np.where(mask.any(1), np.where(mask, z["llr"], -np.inf).max(1), 0.0),
            "shadow_mean": mean(z["f_norm"], mask), "retain_mean": mean(z["r_norm"], mask),
            "length_tokens": n_tok.astype(float), "cut_position": first.astype(float)}


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv); rng = np.random.default_rng(0); res = {"note": "post hoc, not pre-set", "runs": []}
    pooled = {}
    for path in sorted(a.files):
        m = RUN_RE.search(os.path.basename(path)[:-4])
        if not m:
            continue
        arm, seed = m.group(1), int(m.group(2)); z = dict(np.load(path, allow_pickle=False))
        dep, trn = z["policy"] == 0, z["policy"] == 1
        hr, hh = z["hit_routed"].astype(bool), z["hit_heldout"].astype(bool)
        pos, neg = dep & hh & ~hr, dep & ~hh & ~hr; in_pos, in_neg = trn & hr, trn & ~hr & ~hh
        s = scores(z, rng); L = s["length_tokens"]
        run = {"arm": arm, "seed": seed, "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
               "median_length": {"pos": float(np.median(L[pos])) if pos.sum() else None, "neg": float(np.median(L[neg]))},
               "median_cut_position_pos": float(np.median(z["first_hit"][pos])) if pos.sum() else None, "detectors": {}}
        for name, v in s.items():
            run["detectors"][name] = {"auroc": auroc(v[pos], v[neg]), "auroc_ci": boot_ci(auroc, v[pos], v[neg], rng, 300),
                                      "auroc_length_matched": matched_auroc(v, pos, neg, L),
                                      "recall_at_1pct_fpr": recall_at_fpr(v[pos], v[neg]),
                                      "recall_at_5pct_fpr": recall_at_fpr(v[pos], v[neg], 0.05),
                                      "in_fold_auroc": auroc(v[in_pos], v[in_neg])}
            thr = np.quantile(v[neg], 0.99, method="higher") if neg.sum() else np.inf
            p = pooled.setdefault((arm, name), {"hit": 0, "n": 0, "aurocs": []})
            p["hit"] += int((v[pos] > thr).sum()); p["n"] += int(pos.sum()); p["aurocs"].append(run["detectors"][name]["auroc"])
        res["runs"].append(run)
    res["pooled"] = {f"{arm}.{name}": {"recall_at_1pct_fpr": p["hit"] / max(p["n"], 1), "n_pos": p["n"],
                                       "mean_auroc": float(np.nanmean(p["aurocs"])), "aurocs": p["aurocs"]}
                     for (arm, name), p in pooled.items()}
    json.dump(res, open(a.out, "w"), indent=1)
    for r in res["runs"]:
        print(f"\n{r['arm']} s{r['seed']}  pos {r['n_pos']} neg {r['n_neg']}  median length pos/neg {r['median_length']['pos']}/{r['median_length']['neg']}"
              f"  median hack position {r['median_cut_position_pos']}")
        for k, d in r["detectors"].items():
            print(f"  {k:20s} auroc {d['auroc']:.3f} [{d['auroc_ci'][0]:.2f}, {d['auroc_ci'][1]:.2f}]  length-matched {d['auroc_length_matched']:.3f}"
                  f"  recall@1% {d['recall_at_1pct_fpr']:.3f}  @5% {d['recall_at_5pct_fpr']:.3f}  in-fold {d['in_fold_auroc']:.3f}")
    print("\npooled over seeds (thresholds within seed)")
    for k, p in res["pooled"].items():
        print(f"  {k:30s} recall@1% {p['recall_at_1pct_fpr']:.3f} of {p['n_pos']}   mean auroc {p['mean_auroc']:.3f}")


if __name__ == "__main__":
    main()
