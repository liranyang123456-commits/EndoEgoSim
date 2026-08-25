"""轨迹对齐: Umeyama (SE3 6-DoF / Sim3 7-DoF)。"""
from __future__ import annotations

import numpy as np


def umeyama_alignment(src: np.ndarray, dst: np.ndarray,
                      with_scale: bool = False):
    """最小二乘 src -> dst 的相似变换。

    返回 (s, R, t): dst ≈ s * R @ src + t
    src, dst: (N,3)
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    assert src.shape == dst.shape and src.shape[1] == 3
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_s = float((sc ** 2).sum() / max(len(src), 1))
    s = 1.0
    if with_scale and np.isfinite(var_s) and var_s > 1e-12:
        s_hat = float(np.trace(np.diag(D) @ S) / var_s)
        if np.isfinite(s_hat) and s_hat > 1e-12:
            s = s_hat
    t = mu_d - s * R @ mu_s
    return s, R, t


def align_trajectories(est: np.ndarray, gt: np.ndarray, with_scale: bool = False):
    """对齐估计轨迹到真值。返回 (对齐后估计轨迹, (s,R,t))。"""
    s, R, t = umeyama_alignment(est[:, :3, 3], gt[:, :3, 3], with_scale)
    sR = s * R
    n = len(est)
    aligned = np.zeros((n, 4, 4))
    for i in range(n):
        aligned[i, :3, :3] = sR @ est[i, :3, :3]
        aligned[i, :3, 3] = sR @ est[i, :3, 3] + t
    aligned[:, 3, :] = [0.0, 0.0, 0.0, 1.0]
    return aligned, (s, R, t)
