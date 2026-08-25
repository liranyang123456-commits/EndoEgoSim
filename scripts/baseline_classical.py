"""通用相机位姿基线 (对比协议 3): identity / 8-point Essential / RGB-D PnP。

不依赖学习权重, 在同一评测协议上给出经典方法下界。
8-point: ORB 匹配 + findEssentialMat + recoverPose, 链式积分, Sim3 对齐。
PnP: 用仿真度量深度 + ORB, 是 RGB-D 上界参考 (真实 SCARED 无逐帧深度则跳过)。

用法:
  python scripts/baseline_classical.py --method eight --list lists/simtest92.txt \\
      --out results/sota --tag eight_simtest --max-frames 64
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from endosim.eval.metrics import evaluate_trajectory, load_pose_txt, rpe
from endosim.eval.protocol import (list_color_frames, motion_stats_of_indices,
                                   select_frame_indices)
from endosim.geometry.se3 import rot_trans


def _finite_mean(xs):
    a = np.asarray(xs, float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def _finite_median(xs):
    a = np.asarray(xs, float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else float("nan")


def eval_with_scale_correction(est, gt):
    res = evaluate_trajectory(est, gt)
    s = res["ate_sim3"]["scale"]
    if not np.isfinite(s) or s <= 0:
        s = 1.0
        res["ate_sim3"]["scale"] = 1.0
    est_sc = est.copy()
    est_sc[:, :3, 3] *= s
    for g in (1, 5, 10):
        if f"rpe_{g}" in res:
            res[f"rpe_{g}"] = rpe(est_sc, gt, g)
    return res


def run_identity(n: int) -> np.ndarray:
    return np.repeat(np.eye(4)[None], n, axis=0)


def _orb_match(img0, img1, n_feat=2000):
    orb = cv2.ORB_create(n_feat)
    k0, d0 = orb.detectAndCompute(img0, None)
    k1, d1 = orb.detectAndCompute(img1, None)
    if d0 is None or d1 is None or len(k0) < 8 or len(k1) < 8:
        return None, None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    m = bf.match(d0, d1)
    if len(m) < 8:
        return None, None
    m = sorted(m, key=lambda x: x.distance)[:400]
    p0 = np.float32([k0[x.queryIdx].pt for x in m])
    p1 = np.float32([k1[x.trainIdx].pt for x in m])
    return p0, p1


def run_eight(frame_paths, K) -> np.ndarray:
    n = len(frame_paths)
    poses = [np.eye(4)]
    gray0 = cv2.cvtColor(cv2.imread(frame_paths[0]), cv2.COLOR_BGR2GRAY)
    for i in range(1, n):
        gray1 = cv2.cvtColor(cv2.imread(frame_paths[i]), cv2.COLOR_BGR2GRAY)
        p0, p1 = _orb_match(gray0, gray1)
        T = np.eye(4)
        if p0 is not None:
            E, mask = cv2.findEssentialMat(
                p0, p1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if E is not None:
                _, R, t, _ = cv2.recoverPose(E, p0, p1, K, mask=mask)
                T = rot_trans(R, t.reshape(3))
        poses.append(poses[-1] @ T)
        gray0 = gray1
    return np.stack(poses)


def run_pnp(frame_paths, depth_dir, K) -> np.ndarray | None:
    n = len(frame_paths)
    poses = [np.eye(4)]
    img0 = cv2.imread(frame_paths[0])
    if img0 is None:
        return None
    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    h, w = gray0.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    for i in range(1, n):
        dp = os.path.join(depth_dir, os.path.basename(frame_paths[i - 1]).replace(
            ".jpg", ".png"))
        if not os.path.exists(dp):
            stem = os.path.splitext(os.path.basename(frame_paths[i - 1]))[0]
            dp = os.path.join(depth_dir, f"{stem}.png")
        depth = cv2.imread(dp, cv2.IMREAD_UNCHANGED)
        gray1 = cv2.cvtColor(cv2.imread(frame_paths[i]), cv2.COLOR_BGR2GRAY)
        T = np.eye(4)
        if depth is not None:
            if depth.shape[:2] != gray0.shape:
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
            p0, p1 = _orb_match(gray0, gray1)
            if p0 is not None:
                obj, img = [], []
                for (u, v), (u2, v2) in zip(p0, p1):
                    ui, vi = int(round(u)), int(round(v))
                    if not (0 <= ui < w and 0 <= vi < h):
                        continue
                    z = float(depth[vi, ui])
                    if z <= 1e-3:
                        continue
                    obj.append([(u - cx) / fx * z, (v - cy) / fy * z, z])
                    img.append([u2, v2])
                if len(obj) >= 6:
                    ok, rvec, tvec, _ = cv2.solvePnPRansac(
                        np.float32(obj), np.float32(img), K, None,
                        flags=cv2.SOLVEPNP_EPNP)
                    if ok:
                        R, _ = cv2.Rodrigues(rvec)
                        # solvePnP: world(cam i) -> cam i+1, 即 T_cw of i+1 relative to i
                        T = np.linalg.inv(rot_trans(R, tvec.reshape(3)))
        poses.append(poses[-1] @ T)
        gray0 = gray1
    return np.stack(poses)


def stratified_summary(records):
    buckets = {"低参照(<0.3)": [], "中参照(0.3-0.7)": [], "高参照(>0.7)": []}
    for r in records:
        rf = r.get("reference_fraction")
        if rf is None:
            continue
        key = ("低参照(<0.3)" if rf < 0.3 else
               "中参照(0.3-0.7)" if rf < 0.7 else "高参照(>0.7)")
        buckets[key].append(r)
    out = []
    for key, items in buckets.items():
        if not items:
            continue
        out.append({
            "bucket": key, "n": len(items),
            "ate_se3_mean": _finite_mean([r["ate_se3"]["rmse"] for r in items]),
            "ate_sim3_mean": _finite_mean([r["ate_sim3"]["rmse"] for r in items]),
            "rpe1_t_mean": _finite_mean([r["rpe_1"]["trans_mm_mean"] for r in items]),
            "rpe1_r_mean": _finite_mean([r["rpe_1"]["rot_deg_mean"] for r in items]),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["identity", "eight", "pnp"])
    ap.add_argument("--seq", default=None)
    ap.add_argument("--list", default=None)
    ap.add_argument("--out", default="results/sota")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--max-frames", type=int, default=64)
    ap.add_argument("--protocol", default="uniform",
                    choices=["uniform", "consecutive", "stride", "adaptive"])
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-step-mm", type=float, default=12.0)
    args = ap.parse_args()

    if args.list:
        seqs = [ln.strip() for ln in open(args.list, encoding="utf-8")
                if ln.strip() and not ln.startswith("#")]
    else:
        seqs = sorted(glob.glob(args.seq or ""))
    seqs = [s for s in seqs if os.path.isdir(s)]
    tag = args.tag or args.method
    out_dir = os.path.join(args.out, tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[{args.method}] {len(seqs)} sequences")

    records, t0 = [], time.time()
    for i, seq_dir in enumerate(seqs):
        sid = os.path.basename(seq_dir.rstrip("/\\"))
        t_seq = time.time()
        try:
            frames = list_color_frames(seq_dir)
            gt_all = load_pose_txt(os.path.join(seq_dir, "pose_c2w.txt"))
            n = min(len(frames), len(gt_all))
            frames, gt_all = frames[:n], gt_all[:n]
            idx = select_frame_indices(
                gt_all, protocol=args.protocol, max_frames=args.max_frames,
                stride=args.stride, max_step_mm=args.max_step_mm)
            frame_paths = [frames[k] for k in idx]
            gt = gt_all[idx]
            intr = json.load(open(os.path.join(seq_dir, "intrinsics.json")))
            # 图像可能是原始分辨率; 8点用法与读图一致, 按第一帧尺寸缩放内参
            im0 = cv2.imread(frame_paths[0])
            h, w = im0.shape[:2]
            sx, sy = w / intr["width"], h / intr["height"]
            K = np.array([[intr["fx"] * sx, 0, intr["cx"] * sx],
                          [0, intr["fy"] * sy, intr["cy"] * sy],
                          [0, 0, 1.0]], np.float64)
            if args.method == "identity":
                est = run_identity(len(idx))
            elif args.method == "eight":
                est = run_eight(frame_paths, K)
            else:
                est = run_pnp(frame_paths, os.path.join(seq_dir, "depth"), K)
                if est is None:
                    raise RuntimeError("no depth")
            res = eval_with_scale_correction(est, gt)
            res["protocol_hop"] = motion_stats_of_indices(gt_all, idx)
            meta_p = os.path.join(seq_dir, "meta.json")
            if os.path.exists(meta_p):
                meta = json.load(open(meta_p, encoding="utf-8"))
                refs = [r for r in meta.get("reference_fraction", []) if r is not None]
                res["reference_fraction"] = float(np.mean(refs)) if refs else None
            res["seq_id"] = sid
            res["n_frames_used"] = int(len(idx))
            res["time_sec"] = round(time.time() - t_seq, 2)
            records.append(res)
            np.savetxt(os.path.join(out_dir, f"{sid}_est_c2w.txt"),
                       est.reshape(len(est), 16), fmt="%.6f")
            print(f"[{i+1}/{len(seqs)}] {sid}: ATE(Sim3)={res['ate_sim3']['rmse']:.3f}mm "
                  f"({res['time_sec']}s)")
        except Exception as e:
            print(f"[{i+1}/{len(seqs)}] {sid}: FAILED {e}")
            records.append({"seq_id": sid, "error": str(e)})

    ok = [r for r in records if "error" not in r]
    summary = {
        "method": args.method, "n_seq": len(ok), "n_failed": len(records) - len(ok),
        "protocol": {"max_frames": args.max_frames or "all", "name": args.protocol,
                     "stride": args.stride, "max_step_mm": args.max_step_mm},
        "ate_se3_rmse_mean": _finite_mean([r["ate_se3"]["rmse"] for r in ok]) if ok else None,
        "ate_sim3_rmse_mean": _finite_mean([r["ate_sim3"]["rmse"] for r in ok]) if ok else None,
        "ate_sim3_rmse_median": _finite_median([r["ate_sim3"]["rmse"] for r in ok]) if ok else None,
        "rpe1_trans_mean": _finite_mean([r["rpe_1"]["trans_mm_mean"] for r in ok]) if ok else None,
        "rpe1_rot_mean": _finite_mean([r["rpe_1"]["rot_deg_mean"] for r in ok]) if ok else None,
        "total_time_sec": round(time.time() - t0, 1),
    }
    if ok:
        summary["stratified"] = stratified_summary(ok)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, ensure_ascii=False, indent=1)
    print(f"\n=== {args.method} ATE(Sim3)={summary['ate_sim3_rmse_mean']} "
          f"median={summary['ate_sim3_rmse_median']} ===")
    for b in summary.get("stratified", []):
        print(f"  [{b['bucket']}] n={b['n']} ATE={b['ate_sim3_mean']:.3f}mm")


if __name__ == "__main__":
    main()
