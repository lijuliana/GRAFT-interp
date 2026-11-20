"""Planted-signal checks for analyze.py and the fold configs. Run: python test_analyze.py"""
import os, tempfile

import numpy as np

import analyze as A
import make_fold_configs as M


def synth(path, run, shadow_gap=0.0, after_only=False, n=4000, T=12, d=32, seed=0, probe_gap=3.0):
    """Fake shadow_score output. Held-out hacks get `shadow_gap` extra forget norm at the hack
    token (or only after it). Monitor-flagged rollouts are linearly separable in resid."""
    r = np.random.default_rng(seed)
    policy = np.repeat([0, 1], n)
    N = 2 * n
    u = r.random(N)
    hit_r = np.where(policy == 1, u < 0.3, u < 0.01)
    hit_h = ~hit_r & (r.random(N) < 0.15)
    first = np.where(hit_r | hit_h, r.integers(3, T - 2, N), -1)
    mask = np.ones((N, T), bool)
    f = r.normal(1.0, 0.2, (N, T)).astype(np.float32); rn = r.normal(1.0, 0.2, (N, T)).astype(np.float32)
    llr = r.normal(0, 0.1, (N, T)).astype(np.float32)
    for i in np.flatnonzero(hit_h):
        t = first[i] + (1 if after_only else -1)
        f[i, t] += shadow_gap
    for i in np.flatnonzero(hit_r):          # the adapter always knows the routed words
        f[i, first[i] - 1] += 3.0
    resid = r.normal(0, 1, (N, d)).astype(np.float32); resid[hit_r, 0] += probe_gap
    np.savez(path, f_norm=f, r_norm=rn, llr=llr, mask=mask, resid=resid, policy=policy, hit_routed=hit_r,
             hit_heldout=hit_h, first_hit=first, hackable=np.ones(N, bool), routed=np.array(["a", "b"]),
             heldout=np.array(["c", "d"]), run=run, env="object_qa", n_prompts=1, reps=1)


def test_metrics():
    assert A.auroc([2, 3], [0, 1]) == 1.0 and A.auroc([0, 1], [2, 3]) == 0.0
    assert A.auroc([1, 1], [1, 1]) == 0.5
    assert abs(A.auroc([1, 3], [2, 2]) - 0.5) < 1e-9
    r = np.random.default_rng(0)
    neg, pos = r.normal(0, 1, 20000), r.normal(3, 1, 20000)
    assert abs(A.recall_at_fpr(pos, neg) - 0.749) < 0.03          # P(N(3,1) > 2.326)
    assert A.recall_at_fpr(neg[:10000], neg[10000:]) < 0.02


def test_planted_and_null():
    with tempfile.TemporaryDirectory() as d:
        files = []
        for s in (1, 2, 3):
            synth(f"{d}/g{s}.npz", f"cvr.x.graft.s{s}", shadow_gap=2.0, seed=s); files.append(f"{d}/g{s}.npz")
            synth(f"{d}/c{s}.npz", f"cvr.x.noroute.s{s}", shadow_gap=0.0, seed=10 + s); files.append(f"{d}/c{s}.npz")
        res = A.main(files)
        (fold,) = res.values()
        v = fold["verdict"]
        assert v["routing_specific"] and not v["stop"] and v["in_fold_ok"] and v["valid_n"] and v["beats_probe"], v
        for r in fold["runs"]:
            det = r["detectors"]
            assert abs(det["shadow.all"]["shuffled_label_auroc"] - 0.5) < 0.06
            assert abs(det["retain.all"]["auroc"] - 0.5) < 0.06            # control adapter carries no signal
            assert det["probe.all"]["in_fold_auroc"] > 0.9                 # probe finds what it was trained on
            assert abs(det["probe.all"]["auroc"] - 0.5) < 0.08             # and not the held-out words
            if r["arm"] == "graft":
                assert det["shadow.all"]["auroc"] > 0.95 and det["shadow.before"]["auroc"] > 0.9
            else:
                assert abs(det["shadow.all"]["auroc"] - 0.5) < 0.06
        # null everywhere -> stop
        files = []
        for s in (1, 2, 3):
            synth(f"{d}/ng{s}.npz", f"cvr.x.graft.s{s}", seed=20 + s); files.append(f"{d}/ng{s}.npz")
            synth(f"{d}/nc{s}.npz", f"cvr.x.noroute.s{s}", seed=30 + s); files.append(f"{d}/nc{s}.npz")
        (fold,) = A.main(files).values()
        assert fold["verdict"]["stop"] and not fold["verdict"]["routing_specific"]


def test_before_window():
    """A signal that appears only after the hack word is written must vanish from the 'before' window."""
    with tempfile.TemporaryDirectory() as d:
        synth(f"{d}/a.npz", "cvr.x.graft.s1", shadow_gap=2.0, after_only=True, seed=5)
        r = A.analyze_run(dict(np.load(f"{d}/a.npz")), np.random.default_rng(0))["detectors"]
        assert r["shadow.all"]["auroc"] > 0.95 and abs(r["shadow.before"]["auroc"] - 0.5) < 0.06, r


def test_fold_configs():
    for all_splits in (False, True):
        seen = set()
        for env, routed, heldout in M.plan(all_splits):
            words = M.ENVS[env][2]
            assert set(routed) | set(heldout) == set(words) and not set(routed) & set(heldout)
            txt = M.render(env, routed, heldout)
            assert f"environment: {env}" in txt and "\t" not in txt
            s = M.stem(env, routed, heldout); assert s not in seen and "." not in s; seen.add(s)
    two = [p for p in M.plan() if p[0] == "object_qa" and len(p[1]) == 2]
    assert len(two) == 6                                   # every 2-vs-2 split, both directions
    assert {(tuple(h), tuple(r)) for _, r, h in two} == {(tuple(r), tuple(h)) for _, r, h in two}
    with tempfile.TemporaryDirectory() as d:               # the folds sweep's filename filter
        for env, r, h in M.plan():
            open(os.path.join(d, M.stem(env, r, h) + ".yaml"), "w").close()
        stems = sorted(f[:-5] for f in os.listdir(d) if f.startswith("object_qa__") and f.endswith(".yaml")
                       and len(f.split("__")[1].split("-")) - 1 == len(f[:-5].split("__")[2].split("-")) - 1)
        assert len(stems) == 6, stems


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
