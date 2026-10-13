"""Evaluate the per-pair-label ablation (results/egolite/r_v2_pairlab) like r_v2 and compare.

Waits for training to finish, then runs scripts/eval_egolite.py on the Table-5 cohorts (+ the Abs test) and
scripts/eval_decoupling.py on the Abs and R2S tests.  Prints mean ATE and paired Wilcoxon p against r_v2.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
CK = "results/egolite/r_v2_pairlab/last.pth"
def _train_done():
    # PowerShell's *> redirection writes UTF-16, so decode both ways
    raw = Path("results/egolite/r_v2_pairlab.log").read_bytes()
    txt = raw.decode("utf-16", errors="ignore") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", errors="ignore")
    return Path(CK).exists() and "done in" in txt


while not _train_done():
    time.sleep(30)

LISTS = {"core265": "lists/sim_confirmatory265.txt", "scared": "lists/scared_official.txt",
         "c3vd": "lists/c3vd_full.txt", "simtest": "lists/simtest92.txt", "stereomis": "lists/stereomis.txt",
         "r2stest": "lists/r2s_test.txt", "abstest": "lists/abs_test94.txt"}
for name, lst in LISTS.items():
    tag = f"egolite_r_v2_pairlab_{name}"
    if (ROOT / "results/sota" / tag / "summary.json").exists():
        continue
    with open(ROOT / "results/egolite" / f"eval_{tag}.log", "w", encoding="utf-8") as f:
        subprocess.run([sys.executable, "scripts/eval_egolite.py", "--list", lst, "--ckpt", CK, "--tag", tag],
                       stdout=f, stderr=subprocess.STDOUT)
for name, lst in (("abs", "lists/abs_test94.txt"), ("r2s", "lists/r2s_test.txt")):
    out = ROOT / "results/egolite" / f"decoupling_r_v2_pairlab_{name}.json"
    if not out.exists():
        with open(ROOT / "results/egolite" / f"decoupling_r_v2_pairlab_{name}.log", "w", encoding="utf-8") as f:
            subprocess.run([sys.executable, "scripts/eval_decoupling.py", "--ckpt", CK, "--list", lst, "--out", str(out)],
                           stdout=f, stderr=subprocess.STDOUT)

src = (ROOT / "scripts/audit_table_markers.py").read_text(encoding="utf-8").split("\nres = {}")[0]
ns = {"__file__": str(ROOT / "scripts/audit_table_markers.py")}
exec(compile(src, "hdr", "exec"), ns)
per_seq = ns["per_seq"]
summary = {}
for name in LISTS:
    a, b = per_seq(f"egolite_r_v2_{name}"), per_seq(f"egolite_r_v2_pairlab_{name}")
    k = sorted(set(a) & set(b))
    if len(k) < 5:
        print(name, "insufficient")
        continue
    x = np.array([a[s] for s in k]); y = np.array([b[s] for s in k])
    p = float(wilcoxon(y, x).pvalue) if np.any(x != y) else 1.0
    summary[name] = {"n": len(k), "r_v2": float(x.mean()), "pairlab": float(y.mean()), "p": p}
    print(f"{name:10s} n={len(k):3d}  r_v2 {x.mean():7.3f}  per-pair labels {y.mean():7.3f}  p={p:.4f}")
json.dump(summary, open(ROOT / "results/sota/pairlab_ablation_summary.json", "w"), indent=1)
print("decoupling json:", [p.name for p in (ROOT / "results/egolite").glob("decoupling_r_v2_pairlab_*.json")])
