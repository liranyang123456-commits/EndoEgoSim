"""Macro-mean rotational RPE (degrees) per method and cohort from the stored per-sequence records.

Uses the same method/tag maps as scripts/audit_table_markers.py.  Rotation error does not depend on the
scale, so it is comparable across monocular methods.  Gap 1 exists for every learned method; gaps 5 and 10
only where the records contain them.  ORB-SLAM3 and the 8-point chain are skipped: their stored rotation
errors (40-115 deg) are not plausible and use an unchecked pose convention.
Output: results/sota/rotation_rpe_summary.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve()
src = (HERE.parent / "audit_table_markers.py").read_text(encoding="utf-8").split("\nres = {}")[0]
ns = {"__file__": str(HERE.parent / "audit_table_markers.py")}
exec(compile(src, "hdr", "exec"), ns)
S, EGO, METHODS = ns["S"], ns["EGO"], ns["METHODS"]
SKIP = {"ORB-SLAM3", "8-point"}


def rot(tag):
    d = S / tag
    recs = []
    if (d / "summary.json").exists():
        js = json.load(open(d / "summary.json", encoding="utf-8"))
        recs = js.get("records") or js.get("summary", {}).get("records") or []
    for p in d.glob("*_record.json"):
        recs.append(json.load(open(p, encoding="utf-8")))
    out = {}
    for r in recs:
        sid = r.get("seq_id") or Path(r.get("seq_dir", "")).name
        vals = []
        for g in ("rpe_1", "rpe_5", "rpe_10"):
            vals.append(r.get(g, {}).get("rot_deg_mean") if isinstance(r.get(g), dict) else None)
        if vals[0] is not None:
            out[sid] = vals
    return out


res = {}
rows = {m: c for m, c in METHODS.items() if m not in SKIP}
rows["EgoLite"] = EGO
for m, cols in rows.items():
    for col, tag in cols.items():
        o = rot(tag)
        if len(o) < 5:
            continue
        cell = {"n": len(o)}
        for j, g in enumerate((1, 5, 10)):
            v = [x[j] for x in o.values() if x[j] is not None]
            cell[f"gap{g}"] = float(np.mean(v)) if len(v) == len(o) else None
        res[f"{m}|{col}"] = cell
json.dump(res, open(S / "rotation_rpe_summary.json", "w", encoding="utf-8"), indent=1)
for cols in (("SCARED_B", "StereoMIS_B", "C3VD_B"), ("SCARED_A", "StereoMIS_A", "C3VD_A", "EGO-Mo_A")):
    for col in cols:
        print("==", col)
        for m in rows:
            v = res.get(f"{m}|{col}")
            if v:
                f = lambda x: "  --  " if x is None else f"{x:6.3f}"
                print(f"  {m:14s} n={v['n']:2d}  gap1 {f(v['gap1'])}  gap5 {f(v['gap5'])}  gap10 {f(v['gap10'])} deg")
