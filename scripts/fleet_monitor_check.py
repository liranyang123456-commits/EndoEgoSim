"""舰队/训练监管快照：训练进度、渲染进度、进程存活、GPU、磁盘。

供 AGENT_LOOP_TICK 巡查调用，输出紧凑 JSON 报告。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time

ROOT = r"D:\ego_motiion_Camera"
TRAIN_TERM = r"C:\Users\lry\.cursor\projects\d-ego-motiion-Camera\terminals\111383.txt"
RENDER_DIR = os.path.join(ROOT, "sim_data_ego")
EVAL_DIRS = [
    ("phb_core265", os.path.join(ROOT, "results", "sota", "ours_v7phb_confirmatory265"), 265),
    ("phb_ext93", os.path.join(ROOT, "results", "sota", "ours_v7phb_confirmatory_extension93"), 93),
]
ML_DIR = os.path.join(ROOT, "results", "finetune", "ours_vggt_v7_ml")


def tail_lines(path: str, n: int = 400) -> list[str]:
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.readlines()[-n:]


def training_status() -> dict:
    lines = tail_lines(TRAIN_TERM)
    out = {"stage": None, "iter": None, "iters": None, "loss": None,
           "cam": None, "val_ate": None, "last_age_s": None,
           "finished": False, "error": None}
    if not lines:
        out["error"] = "terminal file missing"
        return out
    out["last_age_s"] = round(time.time() - os.path.getmtime(TRAIN_TERM))
    for ln in lines:
        m = re.search(r"\[(\d+)/(\d+)\]\s+loss=([-0-9.]+)\s+cam=([-0-9.]+)", ln)
        if m:
            out["iter"] = int(m.group(1))
            out["iters"] = int(m.group(2))
            out["loss"] = float(m.group(3))
            out["cam"] = float(m.group(4))
        v = re.search(r"\[val\].*?ATE\(Sim3\)=([-0-9.]+)", ln)
        if v:
            out["val_ate"] = float(v.group(1))
        if "core265 redo" in ln:
            out["stage"] = "core265"
        if "extension93 redo" in ln:
            out["stage"] = "extension93"
        if "ml smoke" in ln:
            out["stage"] = "ml_smoke"
        if "full ml training" in ln:
            out["stage"] = "ml_train"
        if "ml training done" in ln:
            out["stage"] = "ml_done"
            out["finished"] = True
        if "Traceback" in ln or "CUDA out of memory" in ln or "CUDA error" in ln:
            out["error"] = "traceback/OOM/CUDA in tail"
    if out["stage"] is None and out["iters"] == 400:
        out["stage"] = "ml_train"
    if out["stage"] is None:
        out["stage"] = "starting"
    return out


def render_status() -> dict:
    out = {"train": 0, "val": 0, "test": 0}
    for split in out:
        d = os.path.join(RENDER_DIR, split)
        if os.path.isdir(d):
            out[split] = len([x for x in os.listdir(d) if x.startswith("seq_")])
    out["total"] = sum(out[k] for k in ("train", "val", "test"))
    return out


def gpu_status() -> dict:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        mem, util = r.stdout.strip().split(",")
        return {"mem_mb": int(mem.strip().split()[0]), "util_pct": int(util.strip().rstrip("%"))}
    except Exception as e:
        return {"error": str(e)}


def disk_status() -> dict:
    import shutil
    free = shutil.disk_usage("D:\\").free / 2**30
    return {"D_free_GiB": round(free, 1)}


def python_procs() -> int:
    try:
        r = subprocess.run(
            ["powershell", "-c",
             "(Get-Process python -ErrorAction SilentlyContinue).Count"],
            capture_output=True, text=True, timeout=20)
        return int(r.stdout.strip() or 0)
    except Exception:
        return -1


def main() -> None:
    eval_prog = {}
    for name, d, total in EVAL_DIRS:
        n = len([f for f in os.listdir(d) if f.endswith("_est_c2w.txt")]) if os.path.isdir(d) else 0
        eval_prog[name] = f"{n}/{total}"
    report = {
        "ts": time.strftime("%H:%M:%S"),
        "training": training_status(),
        "frozen_eval": eval_prog,
        "render": render_status(),
        "gpu": gpu_status(),
        "disk": disk_status(),
        "python_procs": python_procs(),
    }
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
