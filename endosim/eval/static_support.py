"""多帧静止支撑度：跨帧累积确认静止参照，作为位姿可信度信号。

单帧无法判定静止性——静止只在给定相机运动假设下才有定义。多帧观察可破此
循环：真正的静止结构在整个时间窗内对同一刚体运动假设保持一致，形变组织与
器械不会。

本模块不重估位姿，只回答"这一段的位姿假设可不可信"。当前论文把它作为
事后失效诊断；StereoMIS 实验显示它尚不能可靠选择候选位姿，因此不得把它
描述为已启用的路由或误差修正器。

判据用 Sampson 距离，对平移尺度不变，因此校验的是方向与旋转——恰好是
VGGT 坍缩时真正出错的量。
"""
from __future__ import annotations

import cv2
import numpy as np

from ..geometry.se3 import relative

LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def _load_gray(path: str, max_side: int) -> tuple[np.ndarray, float]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None, 1.0
    h, w = img.shape[:2]
    s = max_side / float(max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                         interpolation=cv2.INTER_AREA)
    else:
        s = 1.0
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), s


def track_forward(grays: list[np.ndarray], n_pts: int = 600,
                  fb_thresh: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    """在首帧撒点，前向 LK 逐帧跟踪，前后向一致性校验。

    返回 tracks (K,M,2) 与 valid (K,M)。
    """
    p0 = cv2.goodFeaturesToTrack(grays[0], maxCorners=n_pts, qualityLevel=0.01,
                                 minDistance=7, blockSize=7)
    if p0 is None or len(p0) < 8:
        return np.zeros((0, 0, 2), np.float32), np.zeros((0, 0), bool)
    K = len(grays)
    M = len(p0)
    tracks = np.zeros((K, M, 2), np.float32)
    valid = np.zeros((K, M), bool)
    tracks[0] = p0[:, 0, :]
    valid[0] = True
    cur = p0
    for t in range(1, K):
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(grays[t - 1], grays[t], cur, None,
                                              **LK_PARAMS)
        if nxt is None:
            break
        back, st_b, _ = cv2.calcOpticalFlowPyrLK(grays[t], grays[t - 1], nxt, None,
                                                 **LK_PARAMS)
        ok = st[:, 0].astype(bool)
        if back is not None:
            fb = np.linalg.norm(back[:, 0, :] - cur[:, 0, :], axis=1)
            ok &= (st_b[:, 0].astype(bool) & (fb < fb_thresh))
        tracks[t] = nxt[:, 0, :]
        valid[t] = valid[t - 1] & ok
        cur = nxt
    return tracks, valid


def _essential(T_rel: np.ndarray) -> np.ndarray:
    """由相对位姿构造本质矩阵（对平移尺度不敏感，归一化后使用）。"""
    R = T_rel[:3, :3]
    t = T_rel[:3, 3]
    n = np.linalg.norm(t)
    if n < 1e-12:
        return None
    t = t / n
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    return tx @ R


def sampson(p0: np.ndarray, p1: np.ndarray, E: np.ndarray,
            Kmat: np.ndarray) -> np.ndarray:
    """像素坐标下的 Sampson 距离（像素单位）。"""
    Kinv = np.linalg.inv(Kmat)
    x0 = np.concatenate([p0, np.ones((len(p0), 1))], 1) @ Kinv.T
    x1 = np.concatenate([p1, np.ones((len(p1), 1))], 1) @ Kinv.T
    Ex0 = x0 @ E.T
    Etx1 = x1 @ E
    num = np.sum(x1 * Ex0, axis=1) ** 2
    den = Ex0[:, 0] ** 2 + Ex0[:, 1] ** 2 + Etx1[:, 0] ** 2 + Etx1[:, 1] ** 2
    d_norm = num / np.maximum(den, 1e-12)
    f = 0.5 * (Kmat[0, 0] + Kmat[1, 1])
    return np.sqrt(np.maximum(d_norm, 0.0)) * f


def static_support(frame_paths: list[str], poses_c2w: np.ndarray,
                   Kmat: np.ndarray, max_side: int = 384,
                   n_pts: int = 600, resid_px: float = 2.0,
                   inst_probs: list[np.ndarray] | None = None) -> dict:
    """窗内多帧静止支撑度。

    一条轨迹只有在窗内**每一步**都被位姿假设解释（Sampson < resid_px）才计入
    持久静止支撑。support 低说明不存在能同时解释所有帧的刚体子集，即位姿假
    设不可信。
    """
    grays, scales = [], []
    for p in frame_paths:
        g, s = _load_gray(p, max_side)
        if g is None:
            return {"support": 0.0, "n_tracks": 0, "resid_med": float("inf"),
                    "reason": "load_fail"}
        grays.append(g)
        scales.append(s)
    s = scales[0]
    Ks = Kmat.copy().astype(np.float64)
    Ks[:2] *= s

    tracks, valid = track_forward(grays, n_pts=n_pts)
    if tracks.size == 0:
        return {"support": 0.0, "n_tracks": 0, "resid_med": float("inf"),
                "reason": "no_features"}

    if inst_probs is not None:
        h, w = grays[0].shape
        ip = cv2.resize(inst_probs[0], (w, h), interpolation=cv2.INTER_LINEAR)
        xy = np.round(tracks[0]).astype(int)
        xy[:, 0] = np.clip(xy[:, 0], 0, w - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, h - 1)
        valid[:] &= (ip[xy[:, 1], xy[:, 0]] < 0.3)[None, :]

    Kw = len(frame_paths)
    persist = valid[0].copy()
    resids, inl_fracs = [], []
    for t in range(1, Kw):
        # Epipolar geometry for tracks p_{t-1} -> p_t requires the transform
        # from the previous camera coordinates to the current camera
        # coordinates.  For c2w poses this is inv(P_t) @ P_{t-1}.
        T_cur_from_prev = relative(poses_c2w[t], poses_c2w[t - 1])
        E = _essential(T_cur_from_prev)
        m = valid[t]
        if E is None or m.sum() < 8:
            # 平移退化：无法用对极几何校验，该步不计入但也不判死
            continue
        d = np.full(len(persist), np.inf)
        d[m] = sampson(tracks[t - 1][m], tracks[t][m], E, Ks)
        resids.append(np.median(d[m]))
        inl_fracs.append(float(np.mean(d[m] < resid_px)))
        persist &= (d < resid_px)

    n_seed = int(valid[0].sum())
    n_pers = int(persist.sum())
    return {
        "support": float(n_pers / max(n_seed, 1)),
        "support_soft": float(np.mean(inl_fracs)) if inl_fracs else 0.0,
        "n_tracks": n_seed,
        "n_persist": n_pers,
        "resid_med": float(np.median(resids)) if resids else float("inf"),
        "n_steps": len(resids),
    }


def dominant_rigid_support(frame_paths: list[str], Kmat: np.ndarray,
                           max_side: int = 384, n_pts: int = 600,
                           resid_px: float = 2.0) -> dict:
    """不依赖位姿假设的版本：逐步 RANSAC 找占优刚体簇，再跨帧取交集。

    腔镜里绝对静止区域可能极少，占优簇提法能优雅退化。用于位姿完全不可用
    （global 窗退化）时仍能给出场景可跟踪性评估。
    """
    grays, scales = [], []
    for p in frame_paths:
        g, s = _load_gray(p, max_side)
        if g is None:
            return {"support": 0.0, "n_tracks": 0, "reason": "load_fail"}
        grays.append(g)
        scales.append(s)
    Ks = Kmat.copy().astype(np.float64)
    Ks[:2] *= scales[0]

    tracks, valid = track_forward(grays, n_pts=n_pts)
    if tracks.size == 0:
        return {"support": 0.0, "n_tracks": 0, "reason": "no_features"}

    persist = valid[0].copy()
    for t in range(1, len(grays)):
        m = valid[t]
        if m.sum() < 8:
            continue
        E, inl = cv2.findEssentialMat(tracks[t - 1][m], tracks[t][m], Ks,
                                      method=cv2.RANSAC, prob=0.999,
                                      threshold=resid_px)
        if E is None or inl is None:
            continue
        keep = np.zeros(len(persist), bool)
        idx = np.where(m)[0]
        keep[idx[inl.ravel().astype(bool)]] = True
        persist &= keep

    n_seed = int(valid[0].sum())
    return {"support": float(int(persist.sum()) / max(n_seed, 1)),
            "n_tracks": n_seed, "n_persist": int(persist.sum())}
