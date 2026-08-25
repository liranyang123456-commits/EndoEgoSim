"""光流格式转换: EndoEgoSim npz -> Middlebury .flo（标准格式, 供光流工具链使用）。

用法:
  python scripts/flow_convert.py --seq sim_data/train/seq_XXXX --out flows_flo/
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np


def write_flo(path: str, flow: np.ndarray) -> None:
    """Middlebury .flo: 'PIEH' + width + height (int32 LE) + float32 (H,W,2)。"""
    h, w = flow.shape[:2]
    with open(path, "wb") as f:
        f.write(b"PIEH")
        f.write(np.int32(w).tobytes())
        f.write(np.int32(h).tobytes())
        f.write(flow.astype(np.float32).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="序列目录")
    ap.add_argument("--out", default=None, help="输出目录(默认 <seq>/flow_flo)")
    args = ap.parse_args()
    out = args.out or os.path.join(args.seq, "flow_flo")
    os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.seq, "flow", "*.npz")))
    n = 0
    for p in files:
        flow = np.load(p)["flow"].astype(np.float32)
        flow[~np.isfinite(flow)] = 1e9  # .flo 无效值约定: 超大数
        write_flo(os.path.join(out, os.path.basename(p).replace(".npz", ".flo")), flow)
        n += 1
    print(f"已转换 {n} 个流文件 -> {out}")


if __name__ == "__main__":
    main()
