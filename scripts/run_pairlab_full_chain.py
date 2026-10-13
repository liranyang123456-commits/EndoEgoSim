"""Label ablation over the whole label-dependent chain: stages 5 and 6 retrained with per-pair motion masks.

Stage 5': from results/egolite/s_big_ft/last.pth, 6,600 iterations, data E:A:R = 1:1:2, lr 1e-4  (as r_v1)
Stage 6': from stage 5', 2,200 iterations, data E:A:R = 2:1:1, lr 5e-5                           (as r_v2)
Everything but --pair-labels is identical to the commands in results/egolite/_pipeline_r2s.ps1 and _pipeline_r2.ps1
(same seed 0, hence the same clip sampling).  Stages 1-4 saw EndoEgoSim only, whose labels are per-pair masks anyway.
Then evaluates stage 6' with scripts/eval_label_ablation.py.  Keeps the machine awake while it runs.
"""
import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if os.name == "nt":
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED, reverted at exit

COMMON = ["--batch", "3", "--clip", "8", "--val-every", "1100", "--workers", "6", "--abs-depth", "--flow-net", "large",
          "--head-extra", "6", "--ctx", "--train-flow", "--val-abs", "sim_data_r2s/val", "--pair-labels"]
STAGES = [
    ("results/egolite/r_v1_pairlab", "6600", "1e-4", "results/egolite/s_big_ft/last.pth",
     "sim_data/train:1,sim_data_abs/train:1,sim_data_r2s/train:2"),
    ("results/egolite/r_v2_pairlab_full", "2200", "5e-5", "results/egolite/r_v1_pairlab/last.pth",
     "sim_data/train:2,sim_data_abs/train:1,sim_data_r2s/train:1"),
]
for out, iters, lr, init, data in STAGES:
    last = ROOT / out / "last.pth"
    if last.exists() and "done in" in (ROOT / (out + ".log")).read_text(encoding="utf-8", errors="ignore"):
        print("skip finished stage", out, flush=True)
        continue
    cmd = [sys.executable, "-u", "scripts/train_egolite.py", "--out", out, "--iters", iters, "--lr", lr, "--init", init,
           "--data", data] + COMMON
    t0 = time.time()
    with open(ROOT / (out + ".log"), "w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"{out}: rc={rc} in {time.time() - t0:.0f}s", flush=True)
    if rc != 0:
        sys.exit(f"stage failed: {out}")
rc = subprocess.run([sys.executable, "scripts/eval_label_ablation.py", "--ckpt", "results/egolite/r_v2_pairlab_full/last.pth",
                     "--prefix", "egolite_r_v2_pairlabfull", "--dec-prefix", "decoupling_r_v2_pairlabfull",
                     "--summary", "results/sota/pairlab_full_ablation_summary.json"],
                    stdout=open(ROOT / "results/egolite/pairlab_full_eval.log", "w", encoding="utf-8"),
                    stderr=subprocess.STDOUT).returncode
print("evaluation rc", rc, flush=True)
