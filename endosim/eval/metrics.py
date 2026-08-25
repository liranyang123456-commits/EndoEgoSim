"""egomotion 评测指标: ATE RMSE / RPE（对齐 SCARED 惯例）。

支持:
- SE(3) 6-DoF 对齐（度量尺度方法）
- Sim(3) 7-DoF 对齐（单目 up-to-scale 方法）
- RPE: 平移(mm) + 旋转(deg), 可指定帧距
"""
from __future__ import annotations

import numpy as np

from ..geometry.se3 import relative, so3_log
from .align import align_trajectories


def ate_rmse(est: np.ndarray, gt: np.ndarray, with_scale: bool = False) -> dict:
    """绝对轨迹误差。

    est/gt: (N,4,4) c2w。返回 {rmse, mean, max, std}（mm）。
    """
    est_a, (s, R, t) = align_trajectories(est, gt, with_scale)
    err = np.linalg.norm(est_a[:, :3, 3] - gt[:, :3, 3], axis=1)
    return {"rmse": float(np.sqrt((err ** 2).mean())),
            "mean": float(err.mean()), "max": float(err.max()),
            "std": float(err.std()), "scale": float(s),
            "aligned": est_a}


def rpe(est: np.ndarray, gt: np.ndarray, gap: int = 1) -> dict:
    """相对位姿误差（不对齐，逐段比较）。

    返回 {trans_mm_mean/max, rot_deg_mean/max, trans_mm_rmse}。
    """
    n = min(len(est), len(gt))
    errs_t, errs_r = [], []
    for i in range(n - gap):
        Tij_est = relative(est[i], est[i + gap])
        Tij_gt = relative(gt[i], gt[i + gap])
        dT = np.linalg.inv(Tij_gt) @ Tij_est
        errs_t.append(np.linalg.norm(dT[:3, 3]))
        errs_r.append(np.rad2deg(np.linalg.norm(so3_log(dT[:3, :3]))))
    errs_t = np.asarray(errs_t)
    errs_r = np.asarray(errs_r)
    return {"trans_mm_mean": float(errs_t.mean()) if len(errs_t) else 0.0,
            "trans_mm_max": float(errs_t.max()) if len(errs_t) else 0.0,
            "trans_mm_rmse": float(np.sqrt((errs_t ** 2).mean())) if len(errs_t) else 0.0,
            "rot_deg_mean": float(errs_r.mean()) if len(errs_r) else 0.0,
            "rot_deg_max": float(errs_r.max()) if len(errs_r) else 0.0}


def evaluate_trajectory(est: np.ndarray, gt: np.ndarray,
                        rpe_gaps: tuple = (1, 5, 10)) -> dict:
    """一键评测: ATE(SE3+Sim3) + RPE(多帧距)。"""
    out = {"n_frames": len(gt)}
    out["ate_se3"] = {k: v for k, v in ate_rmse(est, gt, with_scale=False).items()
                      if k != "aligned"}
    out["ate_sim3"] = {k: v for k, v in ate_rmse(est, gt, with_scale=True).items()
                       if k != "aligned"}
    if not np.isfinite(out["ate_sim3"]["scale"]) or out["ate_sim3"]["scale"] <= 0:
        out["ate_sim3"]["scale"] = 1.0
    for g in rpe_gaps:
        if len(gt) > g:
            out[f"rpe_{g}"] = rpe(est, gt, g)
    return out


# ---------------------------------------------------------------------------
# 轨迹 IO
# ---------------------------------------------------------------------------

def load_pose_txt(path: str) -> np.ndarray:
    """读 pose_c2w.txt（每行16数, 行主序4x4）。"""
    P = np.loadtxt(path)
    if P.ndim == 1:
        P = P[None]
    return P.reshape(-1, 4, 4)


def load_tum_gt(path: str) -> np.ndarray:
    """读 TUM groundtruth.txt -> (N,4,4)。"""
    from ..geometry.se3 import R_from_quat
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            t, x, y, z, qx, qy, qz, qw = map(float, p[:8])
            R = R_from_quat([qx, qy, qz, qw])
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [x, y, z]
            rows.append(T)
    return np.stack(rows)
