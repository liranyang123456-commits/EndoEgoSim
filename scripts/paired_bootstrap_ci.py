"""Paired bootstrap 95% CIs for the mean Sim(3) ATE difference (EgoLite minus baseline), per cohort.

Re-uses the per-sequence loader and method/tag maps of scripts/audit_table_markers.py, so the
sequence pairing is identical to the Wilcoxon markers in Tables 2 and 3.
Resampling: sequences with replacement, 10,000 draws, seed 0, percentile interval.
Output: results/sota/paired_bootstrap_ci.json (+ compact printout for the real cohorts).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve()
src = (HERE.parent / "audit_table_markers.py").read_text(encoding="utf-8").split("\nres = {}")[0]
ns = {"__file__": str(HERE.parent / "audit_table_markers.py")}
exec(compile(src, "audit_table_markers_header", "exec"), ns)
S, EGO, METHODS, per_seq = ns["S"], ns["EGO"], ns["METHODS"], ns["per_seq"]

rng = np.random.default_rng(0)
B = 10000
out = {}
for m, cols in METHODS.items():
    for col, tag in cols.items():
        o, e = per_seq(tag), per_seq(EGO[col])
        k = sorted(set(o) & set(e))
        if len(k) < 5:
            continue
        a = np.array([o[x] for x in k])
        b = np.array([e[x] for x in k])
        d = b - a                                  # negative: EgoLite lower (better)
        idx = rng.integers(0, len(d), size=(B, len(d)))
        boots = d[idx].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out[f"{m}|{col}"] = {"n": len(k), "mean_diff": float(d.mean()), "ci95": [float(lo), float(hi)],
                             "egolite_lower_in": int((d < 0).sum()), "baseline_mean": float(a.mean()),
                             "egolite_mean_on_common": float(b.mean())}
json.dump(out, open(S / "paired_bootstrap_ci.json", "w", encoding="utf-8"), indent=1)

order = ["AF-SfMLearner", "EndoDAC", "Endo-FASt3r", "DROID-SLAM", "ORB-SLAM3", "8-point"]
for cols in (("SCARED_B", "StereoMIS_B", "C3VD_B"), ("SCARED_A", "StereoMIS_A", "C3VD_A", "EGO-Mo_A"),
             ("dev", "core", "ext")):
    print("== ", cols)
    for m in list(METHODS):
        cells = []
        for c in cols:
            v = out.get(f"{m}|{c}")
            cells.append("   --   " if v is None else
                         f"{v['mean_diff']:+6.2f} [{v['ci95'][0]:+6.2f},{v['ci95'][1]:+6.2f}] n={v['n']:>2} lower {v['egolite_lower_in']:>2}")
        print(f"{m:14s} | " + " | ".join(cells))
