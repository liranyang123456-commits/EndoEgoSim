"""Evaluate a label-ablation checkpoint like r_v2 and compare it with r_v2 (paired Wilcoxon on per-sequence ATE).

  python scripts/eval_label_ablation.py --ckpt results/egolite/r_v2_pairlab_full/last.pth \
      --prefix egolite_r_v2_pairlabfull --dec-prefix decoupling_r_v2_pairlabfull \
      --summary results/sota/pairlab_full_ablation_summary.json

Runs scripts/eval_egolite.py on the Table-7 cohorts (+ Abs test) and scripts/eval_decoupling.py on the Abs and R2S tests.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--prefix", required=True)
ap.add_argument("--dec-prefix", required=True)
ap.add_argument("--summary", required=True)
a = ap.parse_args()

LISTS = {"core265": "lists/sim_confirmatory265.txt", "scared": "lists/scared_official.txt",
         "c3vd": "lists/c3vd_full.txt", "simtest": "lists/simtest92.txt", "stereomis": "lists/stereomis.txt",
         "r2stest": "lists/r2s_test.txt", "abstest": "lists/abs_test94.txt"}
for name, lst in LISTS.items():
    tag = f"{a.prefix}_{name}"
    if (ROOT / "results/sota" / tag / "summary.json").exists():
        continue
    with open(ROOT / "results/egolite" / f"eval_{tag}.log", "w", encoding="utf-8") as f:
        subprocess.run([sys.executable, "scripts/eval_egolite.py", "--list", lst, "--ckpt", a.ckpt, "--tag", tag],
                       stdout=f, stderr=subprocess.STDOUT)
for name, lst in (("abs", "lists/abs_test94.txt"), ("r2s", "lists/r2s_test.txt")):
    out = ROOT / "results/egolite" / f"{a.dec_prefix}_{name}.json"
    if not out.exists():
        with open(ROOT / "results/egolite" / f"{a.dec_prefix}_{name}.log", "w", encoding="utf-8") as f:
            subprocess.run([sys.executable, "scripts/eval_decoupling.py", "--ckpt", a.ckpt, "--list", lst, "--out", str(out)],
                           stdout=f, stderr=subprocess.STDOUT)

src = (ROOT / "scripts/audit_table_markers.py").read_text(encoding="utf-8").split("\nres = {}")[0]
ns = {"__file__": str(ROOT / "scripts/audit_table_markers.py")}
exec(compile(src, "hdr", "exec"), ns)
per_seq = ns["per_seq"]
summary = {}
for name in LISTS:
    x0, y0 = per_seq(f"egolite_r_v2_{name}"), per_seq(f"{a.prefix}_{name}")
    k = sorted(set(x0) & set(y0))
    if len(k) < 5:
        print(name, "insufficient")
        continue
    x = np.array([x0[s] for s in k]); y = np.array([y0[s] for s in k])
    p = float(wilcoxon(y, x).pvalue) if np.any(x != y) else 1.0
    summary[name] = {"n": len(k), "r_v2": float(x.mean()), "ablation": float(y.mean()), "p": p}
    print(f"{name:10s} n={len(k):3d}  r_v2 {x.mean():7.3f}  ablation {y.mean():7.3f}  p={p:.4f}", flush=True)
dec = {}
for name in ("abs", "r2s"):
    p = ROOT / "results/egolite" / f"{a.dec_prefix}_{name}.json"
    if p.exists():
        js = json.load(open(p, encoding="utf-8"))
        ti = js["tissue_motion_image"]["ours"]
        dec[name] = {"median_px": ti["epe_px_median"], "auroc": ti["moving_vs_static_auroc"],
                     "cosine": ti["moving_direction_cosine"]}
        print(f"decoupling {name}: median {ti['epe_px_median']:.3f} px  AUROC {ti['moving_vs_static_auroc']:.3f}  cos {ti['moving_direction_cosine']:.3f}")
summary["decoupling"] = dec
json.dump(summary, open(ROOT / a.summary, "w", encoding="utf-8"), indent=1)
