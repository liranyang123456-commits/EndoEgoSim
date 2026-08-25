"""为本地 C3VD / SCARED 建立路径索引 (不拷贝任何像素)。

用法:
  python scripts/index_real_datasets.py --out D:/ego_motiion_Camera/sim_data/real_refs

之后评测可直接:
  python scripts/baseline_sota.py --method vggt --seq "sim_data/real_refs/*" \\
      --protocol adaptive --max-frames 32 --max-step-mm 12
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from endosim.dataset.real_access import build_real_refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "sim_data", "real_refs"))
    ap.add_argument("--no-c3vd", action="store_true")
    ap.add_argument("--no-scared", action="store_true")
    args = ap.parse_args()
    results = build_real_refs(args.out, include_c3vd=not args.no_c3vd,
                              include_scared=not args.no_scared)
    print(f"已建立 {len(results)} 条路径引用序列 -> {args.out}")
    print("copied_pixels=False (帧仍在原始数据集路径)")
    lists_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "lists")
    os.makedirs(lists_dir, exist_ok=True)
    scared, c3vd = [], []
    for r in results:
        print(f"  {r['seq']}: {r['n_frames']}帧")
        line = r["dir"].replace("\\", "/")
        if r["seq"].startswith("scared_"):
            scared.append(line)
        else:
            c3vd.append(line)
    if scared:
        p = os.path.join(lists_dir, "scared_official.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(scared) + "\n")
        print(f"wrote {p} ({len(scared)})")
    if c3vd:
        p = os.path.join(lists_dir, "c3vd_full.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(c3vd) + "\n")
        print(f"wrote {p} ({len(c3vd)})")


if __name__ == "__main__":
    main()
