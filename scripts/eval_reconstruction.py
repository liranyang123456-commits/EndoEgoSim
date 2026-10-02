"""下游任务验证：轨迹误差 -> 组织表面重建误差（对称 Chamfer，mm）。

动机（审稿意见 R1-4）：论文主张服务于导航/拼接/重建，但只报了几何 ATE。
本脚本把每条序列的 GT 深度按"估计轨迹（Sim(3) 对齐后）"融合成组织点云，
与 GT 轨迹融合的点云比较对称 Chamfer 距离。深度固定为 GT，因此点云差异
完全来自轨迹质量——这是重建任务对位姿估计的下游度量。

仅静态支撑 (m=1) 与变形组织 (m=2) 像素参与融合；独立运动器械 (m=3) 剔除。

用法：
  python scripts/eval_reconstruction.py --list lists/simtest92.txt \
      --methods ours_v7_infer_simtest,droid_simtest,reloc3r_simtest,orb_e_c2w_simtest,ours_v6_simtest \
      --out results/recon/simtest92_recon.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from endosim.eval.metrics import align_trajectories, load_pose_txt
from endosim.eval.protocol import select_frame_indices


def load_intrinsics(seq_dir: str) -> np.ndarray:
    with open(os.path.join(seq_dir, "intrinsics.json"), encoding="utf-8") as f:
        intr = json.load(f)
    K = np.array(intr["K"], dtype=np.float64).reshape(3, 3)
    return K


def fuse_point_cloud(seq_dir: str, poses: np.ndarray, frame_idx: np.ndarray,
                     K: np.ndarray, per_frame: int = 1500,
                     voxel: float = 1.0, seed: int = 0) -> np.ndarray:
    """按给定 c2w 轨迹融合组织点云（世界系，mm）。"""
    rng = np.random.default_rng(seed)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pts_all = []
    for k, fi in enumerate(frame_idx):
        dp = os.path.join(seq_dir, "depth", f"{int(fi):06d}.png")
        mp = os.path.join(seq_dir, "motion_mask", f"{int(fi):06d}.png")
        if not os.path.isfile(dp):
            continue
        depth = cv2.imread(dp, cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue
        depth = depth.astype(np.float64)  # uint16 mm
        valid = depth > 0
        if os.path.isfile(mp):
            mask = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
            if mask is not None and mask.shape == depth.shape:
                valid &= (mask == 1) | (mask == 2)
        ys, xs = np.nonzero(valid)
        if len(xs) == 0:
            continue
        if len(xs) > per_frame:
            sel = rng.choice(len(xs), per_frame, replace=False)
            xs, ys = xs[sel], ys[sel]
        z = depth[ys, xs]
        x = (xs - cx) * z / fx
        y = (ys - cy) * z / fy
        pc = np.stack([x, y, z], axis=1)  # 相机系
        T = poses[k]
        pw = pc @ T[:3, :3].T + T[:3, 3]
        pts_all.append(pw)
    if not pts_all:
        return np.zeros((0, 3))
    pts = np.concatenate(pts_all, axis=0)
    if voxel > 0 and len(pts) > 0:
        keys = np.floor(pts / voxel).astype(np.int64)
        _, first = np.unique(keys, axis=0, return_index=True)
        pts = pts[np.sort(first)]
    return pts


def symmetric_chamfer(a: np.ndarray, b: np.ndarray) -> dict:
    if len(a) == 0 or len(b) == 0:
        return {"chamfer_mm": float("nan"), "acc_mm": float("nan"),
                "comp_mm": float("nan"), "n_a": len(a), "n_b": len(b)}
    tb = cKDTree(b)
    ta = cKDTree(a)
    d_ab, _ = tb.query(a, k=1)
    d_ba, _ = ta.query(b, k=1)
    return {
        "chamfer_mm": float(0.5 * (d_ab.mean() + d_ba.mean())),
        "acc_mm": float(d_ab.mean()),
        "comp_mm": float(d_ba.mean()),
        "acc_p95_mm": float(np.percentile(d_ab, 95)),
        "comp_p95_mm": float(np.percentile(d_ba, 95)),
        "n_a": len(a), "n_b": len(b),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--methods", required=True,
                    help="逗号分隔的 results/sota 目录名；第一个为参考方法")
    ap.add_argument("--sota-root", default="results/sota")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=64)
    ap.add_argument("--per-frame", type=int, default=1500)
    ap.add_argument("--voxel", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    seqs = [ln.strip() for ln in open(args.list, encoding="utf-8")
            if ln.strip() and not ln.startswith("#")]
    seqs = [s for s in seqs if os.path.isdir(s)]
    if args.limit > 0:
        seqs = seqs[:args.limit]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    records = []
    for si, seq_dir in enumerate(seqs):
        seq_id = os.path.basename(seq_dir.rstrip("/\\"))
        gt_all = load_pose_txt(os.path.join(seq_dir, "pose_c2w.txt"))
        idx = select_frame_indices(gt_all, protocol="uniform",
                                   max_frames=args.max_frames)
        K = load_intrinsics(seq_dir)
        gt_sel = gt_all[idx]
        gt_cloud = fuse_point_cloud(seq_dir, gt_sel, idx, K,
                                    per_frame=args.per_frame,
                                    voxel=args.voxel, seed=0)
        row = {"seq_id": seq_id, "n_frames": len(idx), "methods": {}}
        for tag in methods:
            est_path = os.path.join(args.sota_root, tag,
                                    f"{seq_id}_est_c2w.txt")
            if not os.path.isfile(est_path):
                row["methods"][tag] = {"chamfer_mm": None}
                continue
            est = np.loadtxt(est_path).reshape(-1, 4, 4)
            n = min(len(est), len(idx))
            est, gt_n = est[:n], gt_sel[:n]
            aligned, _ = align_trajectories(est, gt_n, with_scale=True)
            cloud = fuse_point_cloud(seq_dir, aligned, idx[:n], K,
                                     per_frame=args.per_frame,
                                     voxel=args.voxel, seed=0)
            row["methods"][tag] = symmetric_chamfer(cloud, gt_cloud)
        records.append(row)
        if (si + 1) % 10 == 0 or si + 1 == len(seqs):
            print(f"[recon] {si + 1}/{len(seqs)}")

    # 汇总：宏平均 + 配对差（相对第一个方法）
    summary = {}
    ref = methods[0]
    for tag in methods:
        vals = [r["methods"][tag]["chamfer_mm"] for r in records
                if r["methods"].get(tag, {}).get("chamfer_mm")]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        entry = {"n": len(vals)}
        if vals:
            entry["chamfer_mean_mm"] = float(np.mean(vals))
            entry["chamfer_median_mm"] = float(np.median(vals))
        if tag != ref:
            diffs = []
            for r in records:
                a = r["methods"].get(ref, {}).get("chamfer_mm")
                b = r["methods"].get(tag, {}).get("chamfer_mm")
                if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
                    diffs.append(b - a)  # 正 = 该方法比参考差
            if diffs:
                entry["paired_diff_mean_mm_vs_ref"] = float(np.mean(diffs))
                entry["win_rate_vs_ref"] = float(np.mean([d > 0 for d in diffs]))
        summary[tag] = entry

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"list": args.list, "methods": methods,
                   "max_frames": args.max_frames, "voxel": args.voxel,
                   "summary": summary, "records": records},
                  f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
