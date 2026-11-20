import json, sys
import numpy as np
d = json.load(open(sys.argv[1])); res, cross = d["res"], d["cross"]
SEEDS = [1, 9, 15, 16]
def m(arm, L, side, key):
    v = [res[f"{arm}_s{s}"][f"{L}.{side}"][key] for s in SEEDS]
    return np.mean(v), np.min(v), np.max(v)
print("Per-layer means over 4 seeds. E = energy ratio along the hack direction (1.0 = random direction).")
print("L  | GRAFT forget: fro  PR  top1  E_ppsL (min-max)  E_dn20  E_loreft null95 | GRAFT retain: fro  PR  E_ppsL | noroute F E_ppsL  R E_ppsL | noint E_ppsL | F-R overlap8")
for L in range(36):
    f = lambda k: m("graft", L, "forget", k); r = lambda k: m("graft", L, "retain", k)
    ov = np.mean([res[f"graft_s{s}"][f"{L}.fr_overlap8"] for s in SEEDS])
    print(f"{L:2d} | {f('fro')[0]:5.2f} {f('pr')[0]:5.1f} {f('top1_share')[0]:4.2f}  {f('E_write_ppsL')[0]:6.2f} ({f('E_write_ppsL')[1]:.2f}-{f('E_write_ppsL')[2]:.2f})  {f('E_write_dn20')[0]:5.2f}  {f('E_write_loreft')[0]:5.2f}  {f('E_write_null95')[0]:4.2f} | {r('fro')[0]:5.2f} {r('pr')[0]:5.1f} {r('E_write_ppsL')[0]:6.2f} | {m('noroute',L,'forget','E_write_ppsL')[0]:6.2f} {m('noroute',L,'retain','E_write_ppsL')[0]:6.2f} | {m('noint',L,'retain','E_write_ppsL')[0]:6.2f} | {ov:.3f}")
print()
print("Read side (gate rows / up rows, weighted by the norm gain) along the steering direction of the previous layer")
for L in (5, 10, 14, 16, 17, 18, 19, 20, 21, 22, 25, 30, 35):
    print(L, "forget gate %.2f up %.2f | retain gate %.2f up %.2f | null95 %.2f" % (
        m("graft", L, "forget", "E_read_gate")[0], m("graft", L, "forget", "E_read_up")[0], m("graft", L, "retain", "E_read_gate")[0], m("graft", L, "retain", "E_read_up")[0],
        m("graft", L, "forget", "E_read_null95")[0]))
print()
print("Cross-seed agreement (mean over 6 seed pairs): |cos| of top write direction ; top-8 subspace overlap (chance 0.002)")
for L in (0, 5, 10, 14, 16, 17, 18, 19, 20, 21, 22, 25, 30, 35):
    g = lambda a, s: cross[f"{a}.{s}.{L}"]
    print(f"{L:2d}  graft F {g('graft','forget')['top1_abs_cos']:.3f} {g('graft','forget')['top8_overlap']:.3f} | graft R {g('graft','retain')['top1_abs_cos']:.3f} {g('graft','retain')['top8_overlap']:.3f} | noroute F {g('noroute','forget')['top1_abs_cos']:.3f} {g('noroute','forget')['top8_overlap']:.3f} R {g('noroute','retain')['top1_abs_cos']:.3f} {g('noroute','retain')['top8_overlap']:.3f} | noint {g('noint','retain')['top1_abs_cos']:.3f} {g('noint','retain')['top8_overlap']:.3f} | graftF-vs-noint {cross[f'graftF_vs_noint.{L}']:.3f}")
