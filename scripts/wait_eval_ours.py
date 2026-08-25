"""训练结束后自动评测 Ours-Single, 再重写对比总表。

等 results/finetune/ours_vggt/vggt_endo_ft.pth 落盘 (只在 400 iter 结束时写),
然后在同一 92 条、max-frames 64 协议上跑 VGGT。
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "results", "finetune", "ours_vggt", "vggt_endo_ft.pth")
PY = sys.executable


def main():
    print(f"等待 {CKPT}", flush=True)
    while not os.path.exists(CKPT):
        time.sleep(30)
    # 等写入结束
    last = -1
    for _ in range(20):
        sz = os.path.getsize(CKPT)
        if sz > 1e8 and sz == last:
            break
        last = sz
        time.sleep(5)
    print(f"发现 ckpt ({last/1e9:.2f} GB), 开始评测", flush=True)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    r = subprocess.run(
        [PY, os.path.join(ROOT, "scripts", "baseline_sota.py"),
         "--method", "vggt", "--list", os.path.join(ROOT, "lists", "simtest92.txt"),
         "--ckpt", CKPT, "--out", os.path.join(ROOT, "results", "sota"),
         "--tag", "ours_simtest", "--max-frames", "64"],
        cwd=ROOT, env=env)
    if r.returncode != 0:
        sys.exit(r.returncode)
    # 用新单模型替换 v2, 重做无 GT 中位数融合
    subprocess.check_call(
        [PY, os.path.join(ROOT, "scripts", "recompute_and_fuse.py")],
        cwd=ROOT, env=env)
    subprocess.check_call([PY, os.path.join(ROOT, "scripts", "compare_sota.py")],
                          cwd=ROOT, env=env)
    print("Ours-Single 评测完成, 总表已更新", flush=True)


if __name__ == "__main__":
    main()
