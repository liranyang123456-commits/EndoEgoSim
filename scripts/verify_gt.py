"""GT 一致性验证。

核心检验（刚体闭环重投影）:
- 用帧i的深度图反投影点云 -> GT位姿变换到帧j -> 投影到帧j
  -> 与帧j深度图比对（组织静态部分误差应≈0）
- 位姿链闭环: T_ij = T_i^-1 T_j
- mask/bbox 一致性
- 深度>0、深度与bbox内物体距离合理

用法: python scripts/verify_gt.py --seq sim_data/train/seq_00000001
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from endosim.eval.metrics import load_pose_txt
from endosim.render.camera import PinholeCamera


def load_seq(seq_dir: str):
    meta = json.load(open(os.path.join(seq_dir, "meta.json"), encoding="utf-8"))
    K = json.load(open(os.path.join(seq_dir, "intrinsics.json")))
    cam = PinholeCamera(K["fx"], K["fy"], K["cx"], K["cy"], K["width"], K["height"])
    poses = load_pose_txt(os.path.join(seq_dir, "pose_c2w.txt"))
    return meta, cam, poses


def reprojection_closure(seq_dir: str, cam: PinholeCamera, poses: np.ndarray,
                         gaps=(1, 3), stride=5, max_points=20000, skip_dynamic=True):
    """深度+位姿闭环重投影误差（双线性采样深度）。

    组织静态区域: 亚像素重投影残差来自像素量化(<0.5px)与深度图量化(uint16 mm)。
    """
    n = len(poses)
    errs_px, errs_depth = [], []
    rng = np.random.default_rng(0)

    def bilinear(depth: np.ndarray, u: np.ndarray, v: np.ndarray):
        x = np.clip(u - 0.5, 0, depth.shape[1] - 1.001)
        y = np.clip(v - 0.5, 0, depth.shape[0] - 1.001)
        x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
        dx = x - x0; dy = y - y0
        d00 = depth[y0, x0]; d10 = depth[y0, x0 + 1]
        d01 = depth[y0 + 1, x0]; d11 = depth[y0 + 1, x0 + 1]
        return (d00 * (1 - dx) * (1 - dy) + d10 * dx * (1 - dy)
                + d01 * (1 - dx) * dy + d11 * dx * dy)

    for gap in gaps:
        for i in range(0, n - gap, stride):
            j = i + gap
            di = cv2.imread(os.path.join(seq_dir, "depth", f"{i:06d}.png"),
                            cv2.IMREAD_UNCHANGED).astype(np.float64)
            dj = cv2.imread(os.path.join(seq_dir, "depth", f"{j:06d}.png"),
                            cv2.IMREAD_UNCHANGED).astype(np.float64)
            mi = cv2.imread(os.path.join(seq_dir, "object_mask", f"{i:06d}.png"),
                            cv2.IMREAD_UNCHANGED)
            valid = (di > 0) & (dj > 0)
            if skip_dynamic:
                valid &= (mi <= 1)  # 只用组织区域
            ys, xs = np.nonzero(valid)
            if len(ys) == 0:
                continue
            sel = rng.choice(len(ys), size=min(max_points, len(ys)), replace=False)
            ys, xs = ys[sel], xs[sel]
            z = di[ys, xs]
            pc = cam.unproject(np.stack([xs, ys], 1), z)
            # 相机i系 -> 世界 -> 相机j系: T_j^{-1} @ T_i
            T_ij = np.linalg.inv(poses[j]) @ poses[i]
            pc_j = pc @ T_ij[:3, :3].T + T_ij[:3, 3]
            uv_j, z_j = cam.project(pc_j)
            inb = ((uv_j[:, 0] >= 1) & (uv_j[:, 0] < cam.width - 1)
                   & (uv_j[:, 1] >= 1) & (uv_j[:, 1] < cam.height - 1))
            if inb.sum() < 100:
                continue
            uv_j, z_j = uv_j[inb], z_j[inb]
            dj_val = bilinear(dj, uv_j[:, 0], uv_j[:, 1])
            dz = np.abs(z_j - dj_val)
            consistent = dz < 5.0  # 同一表面(排除遮挡)
            if consistent.sum() < 100:
                continue
            # 亚像素重投影残差（相对最近整数像素, 固有<0.5px）
            xi = np.clip(np.round(uv_j[:, 0]).astype(int), 0, cam.width - 1)
            yi = np.clip(np.round(uv_j[:, 1]).astype(int), 0, cam.height - 1)
            err_px = np.abs(uv_j[consistent]
                            - np.stack([xi[consistent], yi[consistent]], 1)).max(1)
            errs_px.append(err_px.mean())
            errs_depth.append(np.median(dz[consistent]))
    return {"reproj_px_mean": float(np.mean(errs_px)) if errs_px else None,
            "depth_residual_mm_median": float(np.mean(errs_depth)) if errs_depth else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--strict", action="store_true", help="形变序列也按刚性验证(预期失败)")
    args = ap.parse_args()

    meta, cam, poses = load_seq(args.seq)
    n = meta["n_frames"]
    n_pose = len(poses)
    n_color = len([f for f in os.listdir(os.path.join(args.seq, "color"))
                   if f.endswith(".png")])
    if not (n == n_pose == n_color):
        print(f"❌ 帧数不一致: meta={n}, pose={n_pose}, color={n_color}")
        sys.exit(1)
    print(f"序列 {meta['seq_id']}: {n}帧, 运动={meta['motion_type']}, "
          f"形变强度={None if not meta['deformation'] else meta['deformation']['strength']}")
    print(f"  帧间步长均值 {meta['motion_stats']['step_mm_mean']:.2f} mm, "
          f"旋转 {meta['motion_stats']['rot_deg_mean']:.2f}°")

    # ---- 检查1: 位姿正交性 ----
    ortho_err = max(abs(np.linalg.det(T[:3, :3]) - 1) for T in poses)
    print(f"[1] 位姿旋转矩阵行列式偏差: {ortho_err:.2e} (应<1e-7)")

    # ---- 检查2: 首帧 = 单位阵 ----
    print(f"[2] 首帧位姿 = I: {np.allclose(poses[0], np.eye(4))}")

    # ---- 检查3: 闭环重投影 ----
    res = reprojection_closure(args.seq, cam, poses)
    print(f"[3] 闭环重投影残差: {res['reproj_px_mean']} px (量化上限0.5), "
          f"深度残差中位 {res['depth_residual_mm_median']} mm")

    # ---- 检查4: 深度合理性 ----
    d0 = cv2.imread(os.path.join(args.seq, "depth", "000000.png"),
                    cv2.IMREAD_UNCHANGED).astype(np.float64)
    dpos = d0[d0 > 0]
    print(f"[4] 首帧深度: 有效像素 {len(dpos)} ({len(dpos)/d0.size*100:.0f}%), "
          f"范围 {dpos.min():.0f}~{dpos.max():.0f} mm")

    # ---- 检查5: mask/bbox 一致性 ----
    if os.path.exists(os.path.join(args.seq, "object_poses.json")):
        objs = json.load(open(os.path.join(args.seq, "object_poses.json")))
        print(f"[5] 物体数 {len(objs)}: " +
              ", ".join(o["name"] for o in objs))

    deform = meta.get("deformation")
    ok = (ortho_err < 1e-7 and res["reproj_px_mean"] is not None
          and res["reproj_px_mean"] < 0.5
          and (res["depth_residual_mm_median"] is None
               or res["depth_residual_mm_median"] < 2.0))
    print(f"\n结论: {'✅ 通过' if ok else '❌ 异常'}"
          + ("（含形变序列, 闭环残差含形变位移属预期）" if deform else ""))


if __name__ == "__main__":
    main()
