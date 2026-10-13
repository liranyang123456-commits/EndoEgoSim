"""Inference-time sensitivity to the number of Gauss-Newton steps (the reported model is trained with 3).

Runs scripts/eval_egolite.py on the development list (92 sequences) and the Abs test (94 sequences)
with 1, 2, 5 and 8 steps, then prints mean/median Sim(3) ATE next to the stored 3-step and 0-step
(regression-only) results.  Summaries land in results/sota/egolite_r_v2_gn{n}_{set}/.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
SETS = {"simtest": "lists/simtest92.txt", "abstest": "lists/abs_test94.txt"}
CK = "results/egolite/r_v2/last.pth"
for n in (1, 2, 5, 8):
    for name, lst in SETS.items():
        tag = f"egolite_r_v2_gn{n}_{name}"
        if (ROOT / "results/sota" / tag / "summary.json").exists():
            continue
        log = ROOT / "results/egolite" / f"eval_egolite_r_v2_gn{n}_{name}.log"
        with open(log, "w", encoding="utf-8") as f:
            subprocess.run([sys.executable, "scripts/eval_egolite.py", "--list", lst, "--ckpt", CK, "--tag", tag,
                            "--gn-iters", str(n)], stdout=f, stderr=subprocess.STDOUT, check=False)


def load(tag):
    p = ROOT / "results/sota" / tag / "summary.json"
    if not p.exists():
        return None
    s = json.load(open(p, encoding="utf-8"))["summary"]
    return s["ate_sim3_rmse_mean"], s["ate_sim3_rmse_median"], s.get("net_ms_per_pair")


rows = {}
for n in (0, 1, 2, 3, 5, 8):
    r = {}
    for name in SETS:
        if n == 3:
            tag = {"simtest": "egolite_r_v2_simtest", "abstest": "egolite_r_v2_abstest"}[name]
        elif n == 0:
            tag = {"simtest": "egolite_r_v2_abl_nogn_simtest", "abstest": "egolite_r_v2_abl_nogn_abstest"}[name]
        else:
            tag = f"egolite_r_v2_gn{n}_{name}"
        r[name] = load(tag)
    rows[n] = r
    cells = []
    for name in SETS:
        v = r[name]
        cells.append("   n/a   " if v is None else f"{v[0]:.2f} ({v[1]:.2f})")
    print(f"GN steps {n}: dev {cells[0]} | Abs {cells[1]}")
json.dump({str(k): v for k, v in rows.items()}, open(ROOT / "results/sota/gn_steps_sensitivity.json", "w"), indent=1)
