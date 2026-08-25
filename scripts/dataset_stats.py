"""数据集统计报告: 扫描 sim_data, 汇总各 split 的运动/形变/物体/纹理分布。

用法: python scripts/dataset_stats.py --root D:/ego_motiion_Camera/sim_data
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\ego_motiion_Camera\sim_data")
    args = ap.parse_args()

    print(f"数据集根目录: {args.root}\n")
    total_frames, total_size = 0, 0.0
    for split in ("train", "val", "test", "demo", "real_test"):
        d = os.path.join(args.root, split)
        if not os.path.isdir(d):
            continue
        metas = []
        for mp in sorted(glob.glob(os.path.join(d, "*", "meta.json"))):
            try:
                with open(mp, encoding="utf-8") as f:
                    metas.append(json.load(f))
            except Exception:
                pass
        if not metas:
            continue
        n = len(metas)
        frames = sum(m["n_frames"] for m in metas)
        size = sum(
            os.path.getsize(os.path.join(r, fn))
            for seq in os.listdir(d)
            for r, _, fs in os.walk(os.path.join(d, seq))
            for fn in fs) / 1e6
        total_frames += frames
        total_size += size
        motion = {}
        for m in metas:
            if "motion_type" in m:
                motion[m["motion_type"]] = motion.get(m["motion_type"], 0) + 1
        deform = [m["deformation"]["strength"] for m in metas
                  if m.get("deformation")]
        tex_real = sum(1 for m in metas if m.get("texture_source") == "real")
        steps = [m["motion_stats"]["step_mm_mean"] for m in metas
                 if "motion_stats" in m]
        rots = [m["motion_stats"]["rot_deg_mean"] for m in metas
                if "motion_stats" in m]
        refs = [np.mean([r for r in m.get("reference_fraction", []) if r is not None])
                for m in metas if m.get("reference_fraction")]
        markers = sum(1 for m in metas if m.get("has_marker"))
        print(f"=== {split}: {n} 序列, {frames} 帧, {size:.0f} MB ===")
        if motion:
            print(f"  运动类型: {motion}")
        if steps:
            print(f"  帧间平移: mean {np.mean(steps):.2f} mm (min {min(steps):.2f} / max {max(steps):.2f})")
            print(f"  帧间旋转: mean {np.mean(rots):.2f}° (min {min(rots):.2f} / max {max(rots):.2f})")
        if refs:
            print(f"  参照物比例: mean {np.mean(refs):.2f} "
                  f"(min {min(refs):.2f} / max {max(refs):.2f}) | <0.3的序列: "
                  f"{sum(1 for r in refs if r < 0.3)}/{len(refs)}")
        print(f"  标记物序列: {markers}/{n}")
        print(f"  形变序列: {len(deform)}/{n} (强度 mean {np.mean(deform):.2f})" if deform
              else "  形变序列: 0")
        print(f"  真实纹理: {tex_real}/{n}")
        objs = sum(len(json.load(open(os.path.join(args.root, split, m['seq_id'],
                'object_poses.json'), encoding='utf-8')))
               for m in metas
               if os.path.exists(os.path.join(args.root, split, m['seq_id'], 'object_poses.json')))
        print(f"  物体总数: {objs}")
        print()

    print(f"合计: {total_frames} 帧, {total_size/1000:.2f} GB")


if __name__ == "__main__":
    main()
