"""评测入口: 对一条/一批序列跑 ATE/RPE（可接任意估计方法）。

内置演示基线:
- identity: 估计=真值+固定漂移噪声（校验指标实现正确性）
- load: 从文件读估计轨迹 (pose_c2w.txt 格式)

用法:
  python scripts/evaluate.py --seq sim_data/train/seq_00000001 --baseline identity --noise 0.5
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from endosim.eval.metrics import evaluate_trajectory, load_pose_txt
from endosim.geometry.se3 import rot_trans, so3_exp


def drift_baseline(gt: np.ndarray, trans_noise_mm: float, rot_noise_deg: float,
                   drift_rate: float, seed: int = 0) -> np.ndarray:
    """真值 + 累积漂移 + 噪声 的伪估计（校验指标用）。"""
    rng = np.random.default_rng(seed)
    n = len(gt)
    est = gt.copy()
    for i in range(1, n):
        rel = np.linalg.inv(gt[i - 1]) @ gt[i]
        # 漂移: 每步加微小误差
        drift = rot_trans(so3_exp(rng.normal(0, np.deg2rad(rot_noise_deg * 0.1), 3)),
                          rng.normal(0, trans_noise_mm * 0.1, 3))
        rel = rel @ drift
        est[i] = est[i - 1] @ rel
    return est


def reference_stratified(seqs: list, evaluate_fn) -> None:
    """按参照物比例分桶评测: 量化'参照物可用度 ↔ egomotion精度'关系。"""
    import json
    import numpy as np
    buckets = {"低参照(<0.3)": [], "中参照(0.3-0.7)": [], "高参照(>0.7)": []}
    for s in seqs:
        meta_p = os.path.join(s, "meta.json")
        if not os.path.exists(meta_p):
            continue
        meta = json.load(open(meta_p, encoding="utf-8"))
        refs = [r for r in meta.get("reference_fraction", []) if r is not None]
        if not refs:
            continue
        rf = float(np.mean(refs))
        key = ("低参照(<0.3)" if rf < 0.3 else
               "中参照(0.3-0.7)" if rf < 0.7 else "高参照(>0.7)")
        buckets[key].append((s, rf))
    print("\n=== 参照物分层评测 ===")
    for key, items in buckets.items():
        if not items:
            continue
        ates = []
        for s, rf in items:
            gt = load_pose_txt(os.path.join(s, "pose_c2w.txt"))
            res = evaluate_fn(gt)
            ates.append(res["ate_se3"]["rmse"])
        print(f"  {key}: {len(items)}序列, ATE(SE3) rmse 均值 "
              f"{np.mean(ates):.4f} mm (参照比例均值 {np.mean([r for _, r in items]):.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="序列目录或glob")
    ap.add_argument("--baseline", default="identity",
                    choices=["identity", "drift", "load"])
    ap.add_argument("--est", default=None, help="load时的估计轨迹文件")
    ap.add_argument("--noise", type=float, default=0.3)
    ap.add_argument("--rot-noise", type=float, default=0.1)
    ap.add_argument("--drift", type=float, default=0.001)
    ap.add_argument("--by-reference", action="store_true",
                    help="按参照物比例分桶评测(演示: 用drift伪基线)")
    args = ap.parse_args()

    seqs = sorted(glob.glob(args.seq)) if any(c in args.seq for c in "*?[") else [args.seq]
    if args.by_reference:
        def eval_fn(gt):
            est = drift_baseline(gt, args.noise, args.rot_noise, args.drift)
            return evaluate_trajectory(est, gt)
        reference_stratified(seqs, eval_fn)
        return
    all_res = []
    for s in seqs:
        gt_path = os.path.join(s, "pose_c2w.txt")
        if not os.path.exists(gt_path):
            print(f"跳过 {s}: 无位姿文件")
            continue
        gt = load_pose_txt(gt_path)
        if args.baseline == "identity":
            est = gt.copy()
        elif args.baseline == "drift":
            est = drift_baseline(gt, args.noise, args.rot_noise, args.drift)
        else:
            from endosim.eval.metrics import load_pose_txt as lp
            est = lp(args.est)
        res = evaluate_trajectory(est, gt)
        all_res.append(res)
        print(f"\n=== {os.path.basename(s)} ({res['n_frames']}帧) ===")
        print(f"  ATE(SE3):  rmse={res['ate_se3']['rmse']:.4f} mm  "
              f"max={res['ate_se3']['max']:.4f} mm")
        print(f"  ATE(Sim3): rmse={res['ate_sim3']['rmse']:.4f} mm  "
              f"scale={res['ate_sim3']['scale']:.4f}")
        for g in (1, 5, 10):
            k = f"rpe_{g}"
            if k in res:
                r = res[k]
                print(f"  RPE(gap={g}): t={r['trans_mm_mean']:.4f} mm  "
                      f"r={r['rot_deg_mean']:.4f}°")

    if len(all_res) > 1:
        print(f"\n=== 汇总 ({len(all_res)} 序列) ===")
        print(f"  ATE(SE3) rmse 均值: {np.mean([r['ate_se3']['rmse'] for r in all_res]):.4f} mm")
        print(f"  RPE(1) t 均值: {np.mean([r['rpe_1']['trans_mm_mean'] for r in all_res]):.4f} mm")


if __name__ == "__main__":
    main()
